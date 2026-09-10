"""Isaac AppLauncher preflight for the mixed-wall T800 kick tasks.

This is deliberately a construction/reset/rollout check, not a training smoke.
It writes its marker only after the mixed pose mask, reset event, target/critic
contract, material settings, and a complete finite rollout have all passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--motion-file", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 32:
    parser.error("mixed-wall preflight requires exactly --num-envs 32")
if not args.motion_file.is_file():
    parser.error(f"motion file is unavailable: {args.motion_file}")
if not args.usd_dir.is_dir():
    parser.error(f"USD directory is unavailable: {args.usd_dir}")
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import whole_body_tracking.tasks  # noqa: F401


ACTOR_OBS = 134
EXPECTED_BASE_CRITIC_OBS = 284
EXPECTED_CRITIC_OBS = EXPECTED_BASE_CRITIC_OBS + 3


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite {name}")


def _native_motion_settings(cfg) -> None:
    with np.load(args.motion_file, allow_pickle=False) as motion:
        frames = int(motion["joint_pos"].shape[0])
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if frames <= 0 or fps <= 0.0:
        raise RuntimeError(f"invalid motion metadata: frames={frames}, fps={fps}")
    cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    cfg.commands.motion.min_traj_duration = None
    cfg.commands.motion.bridge_frames = 0
    cfg.commands.motion.pd_stand_reset_ratio = 0.0
    cfg.episode_length_s = frames / fps
    return frames, fps


def _obs_tensor(observations, name: str) -> torch.Tensor:
    if isinstance(observations, dict):
        value = observations.get(name)
        if value is not None:
            return value
    raise RuntimeError(f"observation group {name!r} is missing")


def _check_absent_target_contact(base, absent_ids: torch.Tensor) -> dict:
    sensor = base.scene.sensors.get("target_contact")
    if sensor is None:
        raise RuntimeError("target_contact sensor is missing")
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None or force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
        raise RuntimeError(f"target_contact filtered force matrix is unresolved: {force_matrix}")
    absent_force = force_matrix[absent_ids]
    _finite(absent_force, "absent target_contact force")
    max_absent_force = float(absent_force.abs().max()) if absent_force.numel() else 0.0
    if max_absent_force > 1e-5:
        raise RuntimeError(f"absent target_contact is not zero: max_abs={max_absent_force}")
    return {
        "force_matrix_shape": list(force_matrix.shape),
        "absent_max_abs_force": max_absent_force,
    }


def _material_evidence(cfg, robot):
    terrain_material = cfg.scene.terrain.physics_material
    if (
        terrain_material.static_friction != 1.0
        or terrain_material.dynamic_friction != 1.0
        or terrain_material.friction_combine_mode != "multiply"
    ):
        raise RuntimeError("terrain material is not static=dynamic=1.0 with multiply combine")
    event = cfg.events.physics_material
    params = event.params
    if params.get("static_friction_range") != (1.0, 1.7):
        raise RuntimeError(f"unexpected static friction range: {params.get('static_friction_range')}")
    if params.get("dynamic_friction_range") != (1.0, 1.7):
        raise RuntimeError(f"unexpected dynamic friction range: {params.get('dynamic_friction_range')}")
    if params.get("make_consistent") is not True:
        raise RuntimeError("physics material randomization must set make_consistent=True")

    values = robot.root_physx_view.get_material_properties()
    _finite(values, "runtime material properties")
    if values.ndim < 2 or values.shape[-1] < 2:
        raise RuntimeError(f"unexpected runtime material property shape: {tuple(values.shape)}")
    static = values[..., 0]
    dynamic = values[..., 1]
    if bool((static < 1.0 - 1e-4).any()) or bool((static > 1.7 + 1e-4).any()):
        raise RuntimeError("runtime static friction is outside [1.0, 1.7]")
    if bool((dynamic < 1.0 - 1e-4).any()) or bool((dynamic > 1.7 + 1e-4).any()):
        raise RuntimeError("runtime dynamic friction is outside [1.0, 1.7]")
    if bool((dynamic > static + 1e-5).any()):
        raise RuntimeError("runtime dynamic friction exceeds static friction")
    return {
        "terrain_static_friction": terrain_material.static_friction,
        "terrain_dynamic_friction": terrain_material.dynamic_friction,
        "terrain_friction_combine_mode": terrain_material.friction_combine_mode,
        "randomized_static_range": list(params["static_friction_range"]),
        "randomized_dynamic_range": list(params["dynamic_friction_range"]),
        "make_consistent": params["make_consistent"],
        "runtime_static_min_max": [float(static.min()), float(static.max())],
        "runtime_dynamic_min_max": [float(dynamic.min()), float(dynamic.max())],
    }


def _probe_reference_contact(base, cfg, present_ids, absent_ids):
    """Check target reachability at reference frames, not policy success."""
    command = base.command_manager.get_term("motion")
    robot = base.scene["robot"]
    sensor = base.scene.sensors["target_contact"]
    term = base.reward_manager.get_term_cfg("target_contact")
    total = command.motion.time_step_total - 1
    start = round(term.params["phase_start"] * total)
    end = round(term.params["phase_end"] * total)
    env_ids = torch.arange(base.num_envs, device=base.device)
    old_range = command.cfg.joint_position_range
    command.cfg.joint_position_range = (0.0, 0.0)
    contacts = []
    try:
        for frame in range(start, end + 1):
            command.time_steps[:] = frame
            command._reset_envs_from_motion(env_ids)
            sensor.reset(env_ids)
            robot.set_joint_position_target(robot.data.joint_pos.clone())
            robot.set_joint_velocity_target(robot.data.joint_vel.clone())
            frame_peak = 0.0
            for _ in range(4):
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(base.physics_dt)
                matrix = sensor.data.force_matrix_w
                _finite(matrix, "reference probe target forces")
                frame_peak = max(frame_peak, float(torch.linalg.vector_norm(matrix[present_ids], dim=-1).max()))
                _check_absent_target_contact(base, absent_ids)
            contacts.append({"frame": frame, "present_peak_force_n": frame_peak})
    finally:
        command.cfg.joint_position_range = old_range
        base.reset()
    if max(row["present_peak_force_n"] for row in contacts) <= term.params["force_threshold"]:
        raise RuntimeError(f"Wall unreachable in reference strike window: {contacts}")
    return {"purpose": "reference-pose reachability, not learned-policy performance", "frames": contacts}


@torch.inference_mode()
def main() -> dict:
    spec = gym.spec(args.task)
    cfg = spec.kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_dir = str(args.usd_dir.resolve())
    cfg.scene.terrain.visual_material = None
    cfg.scene.contact_forces.debug_vis = False
    cfg.commands.motion.debug_vis = False
    frames, fps = _native_motion_settings(cfg)
    if cfg.scene.target is None:
        raise RuntimeError("mixed-wall task config has no target wall")
    configured_target_pos = torch.tensor(cfg.scene.target.init_state.pos, dtype=torch.float32)
    configured_target_rot = torch.tensor(cfg.scene.target.init_state.rot, dtype=torch.float32)

    env = None
    try:
        env = gym.make(args.task, cfg=cfg)
        base = env.unwrapped
        observations, _ = env.reset(seed=args.seed)
        actor = _obs_tensor(observations, "policy")
        critic = _obs_tensor(observations, "critic")
        if tuple(actor.shape) != (args.num_envs, ACTOR_OBS):
            raise RuntimeError(f"actor observation contract changed: {tuple(actor.shape)}")
        if tuple(critic.shape) != (args.num_envs, EXPECTED_CRITIC_OBS):
            raise RuntimeError(f"critic observation contract changed: {tuple(critic.shape)}")
        _finite(actor, "actor observations")
        _finite(critic, "critic observations")

        mask = getattr(base, "_target_wall_present", None)
        if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool or tuple(mask.shape) != (args.num_envs,):
            raise RuntimeError("_target_wall_present must be a bool tensor shaped (num_envs,)")
        expected_mask = (torch.arange(args.num_envs, device=mask.device) % 2) == 0
        if not torch.equal(mask, expected_mask):
            raise RuntimeError("mixed-wall mask is not deterministic even=present, odd=absent")

        target = base.scene["target"]
        local_target_pos = target.data.root_pos_w - base.scene.env_origins
        _finite(local_target_pos, "target poses")
        absent_ids = torch.arange(1, args.num_envs, 2, device=mask.device)
        present_ids = torch.arange(0, args.num_envs, 2, device=mask.device)
        if not torch.allclose(local_target_pos[absent_ids, 2], torch.full_like(local_target_pos[absent_ids, 2], -10.0), atol=1e-3):
            raise RuntimeError("absent target walls are not underground at local z=-10")
        if bool(torch.isclose(local_target_pos[present_ids, 2], torch.tensor(-10.0, device=mask.device), atol=1e-3).any()):
            raise RuntimeError("present target wall is underground")
        expected_center = configured_target_pos.to(local_target_pos)
        depth_range = cfg.events.reset_target.params.get("strike_depth_range", (0.025, 0.025))
        sampled_depth = 0.025 + expected_center[0] - local_target_pos[present_ids, 0]
        if not bool(((sampled_depth >= depth_range[0] - 1e-3) &
                     (sampled_depth <= depth_range[1] + 1e-3)).all()):
            raise RuntimeError("present wall depth is outside the configured random range")
        if not torch.allclose(local_target_pos[present_ids, 1:], expected_center[1:], atol=1e-3):
            raise RuntimeError(
                "present wall center differs from cfg.scene.target.init_state.pos: "
                f"expected={expected_center.tolist()} actual={local_target_pos[present_ids[0]].tolist()}"
            )
        target_quat = target.data.root_quat_w[present_ids]
        expected_quat = configured_target_rot.to(target_quat)
        if not torch.allclose(target_quat, expected_quat, atol=1e-3):
            raise RuntimeError("present wall orientation differs from cfg.scene.target.init_state.rot")

        target_obs = cfg.observations.critic.target_pos_b
        target_pos_b = target_obs.func(base, **target_obs.params)
        _finite(target_pos_b, "critic target position")
        if not torch.allclose(target_pos_b[absent_ids], torch.zeros_like(target_pos_b[absent_ids]), atol=1e-4):
            raise RuntimeError("critic target_pos_b is not zero for absent walls")
        target_contact = _check_absent_target_contact(base, absent_ids)

        subset = absent_ids[::2]
        base.reset(env_ids=subset)
        reset_local_target_pos = target.data.root_pos_w - base.scene.env_origins
        if not torch.equal(base._target_wall_present, expected_mask):
            raise RuntimeError("subset reset changed the mixed-wall mask")
        if not torch.allclose(reset_local_target_pos[subset, 2], torch.full_like(reset_local_target_pos[subset, 2], -10.0), atol=1e-3):
            raise RuntimeError("subset reset did not restore absent walls underground")
        target_contact.update(_check_absent_target_contact(base, absent_ids))

        material = _material_evidence(cfg, base.scene["robot"])
        reference_contact = _probe_reference_contact(base, cfg, present_ids, absent_ids)
        action_dim = base.action_manager.total_action_dim
        action = torch.zeros((args.num_envs, action_dim), device=base.device)
        _finite(action, "zero actions")
        steps = int(base.max_episode_length) + 2
        done_count = 0
        reward_min = float("inf")
        reward_max = float("-inf")
        for step in range(steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            _finite(reward, "rewards")
            _finite(action, "actions")
            _finite(_obs_tensor(next_obs, "policy"), "rollout actor observations")
            _finite(_obs_tensor(next_obs, "critic"), "rollout critic observations")
            reward_min = min(reward_min, float(reward.min()))
            reward_max = max(reward_max, float(reward.max()))
            done_count += int((terminated | truncated).sum())
            if step % 32 == 0 or step == steps - 1:
                target_contact.update(_check_absent_target_contact(base, absent_ids))
                contact_term = base.reward_manager.get_term_cfg("target_contact")
                contact_reward = contact_term.func(base, **contact_term.params)
                if bool((contact_reward[absent_ids] != 0).any()):
                    raise RuntimeError("Absent-wall environments received a contact reward")

        return {
            "passed": True,
            "task": args.task,
            "motion_file": str(args.motion_file.resolve()),
            "num_envs": args.num_envs,
            "motion_frames": frames,
            "motion_fps": fps,
            "actor_obs_shape": list(actor.shape),
            "critic_obs_shape": list(critic.shape),
            "mixed_wall_present_ids_even": True,
            "absent_wall_local_z": -10.0,
            "configured_wall_center": configured_target_pos.tolist(),
            "strike_depth_range_m": list(depth_range),
            "sampled_strike_depth_m": sampled_depth.tolist(),
            "configured_wall_quat_wxyz": configured_target_rot.tolist(),
            "subset_reset_count": int(subset.numel()),
            "rollout_steps": steps,
            "done_count": done_count,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "target_contact": target_contact,
            "material": material,
            "reference_contact_probe": reference_contact,
            "device": str(args.device),
        }
    finally:
        if env is not None:
            env.close()


try:
    evidence = main()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    app.close()
    raise
else:
    app.close()
