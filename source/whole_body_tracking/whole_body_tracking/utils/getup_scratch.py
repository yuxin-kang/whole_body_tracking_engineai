"""Audits and full-start evaluation for the independent get-up candidates."""

from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
import torch


def audit_scratch_initialization(runner):
    """Fail closed on an accidentally resumed optimizer, policy counter or normalizer."""
    counts = [int(norm.count.item()) for norm in (runner.obs_normalizer, runner.privileged_obs_normalizer)]
    state_count = len(runner.alg.optimizer.state_dict()["state"])
    if runner.current_learning_iteration != 0 or state_count or counts != [0, 0]:
        raise RuntimeError(f"Not a fresh initialization: iter={runner.current_learning_iteration}, "
                           f"optimizer_states={state_count}, normalization_counts={counts}")
    state = runner.alg.policy.state_dict()
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("Non-finite initial model")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return {"first_iteration": 0, "optimizer_state_count": state_count,
            "normalization_counts": counts, "initial_model_sha256": digest.hexdigest(),
            "initial_optimizer_lrs": [group["lr"] for group in runner.alg.optimizer.param_groups]}


def summarize_full_start(samples, step_dt):
    """Posture screening only: full bilateral foot support needs separate validation."""
    frames = np.asarray(samples["frame"])
    expected = np.arange(frames.shape[0])[:, None]
    uninterrupted = np.all(frames == expected, axis=0)
    window = (frames * step_dt >= 6.5 - 1e-6) & (frames * step_dt <= 7.5 + 1e-6)
    valid = ((samples["height"] >= .90) & (samples["tilt"] <= math.radians(15))
             & (samples["angular_speed"] <= .5))
    enough_window = window.sum(axis=0) >= round(1.0 / step_dt)
    passed = uninterrupted & enough_window & np.all(~window | valid, axis=0)
    transition = (frames * step_dt >= 3.0) & (frames * step_dt <= 5.5)
    streak = np.zeros(frames.shape[1], dtype=int)
    longest = streak.copy()
    for row in valid:
        streak = np.where(row, streak + 1, 0)
        longest = np.maximum(longest, streak)
    return {
        "num_envs_evaluated": frames.shape[1], "posture_pass_fraction": float(passed.mean()),
        "posture_pass_per_env": passed.tolist(), "uninterrupted_from_frame_zero": uninterrupted.tolist(),
        "maximum_height_per_env": samples["height"].max(axis=0).tolist(),
        "longest_posture_hold_s_per_env": (longest * step_dt).tolist(),
        "transition_orientation_error_rad": float(samples["orientation_error"][transition].mean()),
        "transition_position_error_m": float(samples["position_error"][transition].mean()),
        "note": "Deterministic actor, original noise/startup physics; posture screen is NOT verified bilateral standing success.",
    }


def learn_chunk(runner, updates):
    """Normalize the initial observation on re-entry to the installed RSL-RL learn().

    Upstream normalizes observations after env.step(), but not its initial
    get_observations(). This one-shot hook keeps populated normalization valid
    across evaluation boundaries without forking the upstream PPO loop.
    """
    original_reader = runner.env.get_observations

    def first_observations():
        runner.env.get_observations = original_reader
        obs, extras = original_reader()
        extras = {**extras, "observations": dict(extras["observations"])}
        for norm, key in ((runner.obs_normalizer, None), (runner.privileged_obs_normalizer, "critic")):
            was_training = norm.training
            norm.eval()
            try:
                with torch.no_grad():
                    if key is None:
                        obs = norm(obs)
                    else:
                        extras["observations"][key] = norm(extras["observations"][key])
            finally:
                norm.train(was_training)
        return obs, extras

    if runner.current_learning_iteration > 0:
        runner.env.get_observations = first_observations
    try:
        runner.learn(num_learning_iterations=updates, init_at_random_ep_len=False)
    finally:
        runner.env.get_observations = original_reader


def assert_preflight_result(summary, diagnostics, variant, expected_steps):
    """Require a valid training/evaluation procedure, not success of a random policy."""
    if not summary["uninterrupted_from_frame_zero"] or not all(summary["uninterrupted_from_frame_zero"]):
        raise RuntimeError("Preflight full-start trajectory reset or skipped reference frames")
    required_sources = range(3) if variant in {"c", "d"} else range(1)
    if any(diagnostics["reset_source_counts"][i] == 0 for i in required_sources):
        raise RuntimeError("Preflight did not exercise all required reset sources")
    if sum(diagnostics["physical_source_steps"]) != expected_steps:
        raise RuntimeError("Preflight physical step accounting mismatch")
    for name in ("peak_contact_force_n", "sampled_contact_impulse_proxy_ns", "max_reference_joint_limit_violation_rad"):
        if not math.isfinite(diagnostics[name]):
            raise RuntimeError(f"Non-finite preflight diagnostic: {name}")


@torch.inference_mode()
def evaluate_full_start(runner, output, iteration, num_reported_envs=16):
    """Evaluate between learn() chunks, then reset training without advancing its curriculum.

    Reuses the allocated simulator to avoid a second Isaac process on a 24GB GPU.
    All environments step, but only the first requested environments are reported.
    This intentionally resets training episodes at the evaluation boundary.
    """
    env = runner.env
    base = env.unwrapped
    command = base.command_manager.get_term("motion")
    count = min(num_reported_envs, base.num_envs)
    old_play = command._play_from_start
    old_diagnostics_enabled = getattr(env, "diagnostics_enabled", False)
    env.diagnostics_enabled = False
    old_step = base.common_step_counter
    old_sim_step = base._sim_step_counter
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
           np.random.get_state(), random.getstate())
    norm_counts = (int(runner.obs_normalizer.count), int(runner.privileged_obs_normalizer.count))
    output.mkdir(parents=True, exist_ok=False)
    samples = {}

    def restore_rng_and_counters():
        base.common_step_counter = old_step
        base._sim_step_counter = old_sim_step
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2])
        random.setstate(rng[3])

    try:
        command._play_from_start = True
        obs, _ = env.reset()
        policy = runner.get_inference_policy(device=runner.device)
        with torch.inference_mode():
            for _ in range(command.motion.time_step_total):
                q = command.robot_anchor_quat_w[:count]
                ref_q = command.anchor_quat_w[:count]
                tilt = torch.acos((1 - 2 * (q[:, 1].square() + q[:, 2].square())).clamp(-1, 1))
                angle = 2 * torch.acos((q * ref_q).sum(-1).abs().clamp(0, 1))
                values = {
                    "frame": command.time_steps[:count], "height": command.robot_anchor_pos_w[:count, 2],
                    "reference_height": command.anchor_pos_w[:count, 2], "tilt": tilt,
                    "orientation_error": angle,
                    "position_error": torch.linalg.vector_norm(
                        command.anchor_pos_w[:count] - command.robot_anchor_pos_w[:count], dim=-1),
                    "angular_speed": torch.linalg.vector_norm(command.robot_anchor_ang_vel_w[:count], dim=-1),
                    "joint_pos": command.robot_joint_pos[:count], "reference_joint_pos": command.joint_pos[:count],
                    "contact_forces_w": base.scene.sensors["contact_forces"].data.net_forces_w[:count],
                }
                for name, value in values.items():
                    if not torch.isfinite(value).all():
                        raise RuntimeError(f"Non-finite evaluation sample: {name}")
                    samples.setdefault(name, []).append(value.detach().cpu().numpy().copy())
                obs, _, _, _ = env.step(policy(obs))
        arrays = {name: np.stack(values) for name, values in samples.items()}
        summary = summarize_full_start(arrays, base.step_dt)
        summary.update(iteration=iteration, step_dt=base.step_dt,
                       contact_body_names=list(base.scene.sensors["contact_forces"].body_names),
                       joint_names=list(command.cfg.motion_joint_names))
        np.savez_compressed(output / "rollout.npz", **arrays)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        if norm_counts != (int(runner.obs_normalizer.count), int(runner.privileged_obs_normalizer.count)):
            raise RuntimeError("Evaluation changed normalization counts")
        return summary
    finally:
        command._play_from_start = old_play
        restore_rng_and_counters()
        try:
            env.reset()  # Sample the training reset from the saved, not evaluation, RNG stream.
        finally:
            restore_rng_and_counters()  # The reset itself must not leak evaluation RNG advancement.
            env.diagnostics_enabled = old_diagnostics_enabled
            runner.train_mode()
