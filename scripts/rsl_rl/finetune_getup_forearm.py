"""Fine-tune the C get-up checkpoint with the four forearm ablations.

The source run is treated as an immutable configuration/checkpoint snapshot.
Each invocation owns one GPU and one F1--F4 variant.  Preflight mode restores
the complete PPO state, performs two updates, then exercises the complete
379-frame deterministic rollout and the new ground-filtered reward terms.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

from isaaclab.app import AppLauncher


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "logs/rsl_rl/t800_flat/2026-09-06_16-08-31_t800_getup_aggressive_c_scratch_3090_j9746"
CHECKPOINT_SHA256 = "42e1660dde180350f935cd8a72aa757b2c2b42c1b58e86f466a7a9565f037d90"
MOTION_SHA256 = "e5a1711e492c7fadac84741d11fb09f0ebaa8fe659b68ab3674cb6b8d482812d"
RESUME_ITERATION = 18_501
FULL_FRAMES = 379
ACTOR_OBS = 134
CRITIC_OBS = 284


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("f1", "f2", "f3", "f4"), required=True)
parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=5000, help="Additional PPO updates.")
parser.add_argument("--learning-rate", type=float, default=1.0e-5)
parser.add_argument("--preflight", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1:
    parser.error("Environment and iteration counts must be positive")
if args.preflight and (args.num_envs != 32 or args.iterations != 2):
    parser.error("Preflight requires exactly --num-envs 32 --iterations 2")
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.io import dump_pickle, dump_yaml, load_pickle
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.config.t800.getup_forearm_env_cfg import (
    apply_getup_forearm_experiment,
)
from whole_body_tracking.utils.getup_finetune import config_changes, restore_finetune_checkpoint
from whole_body_tracking.utils.getup_scratch import learn_chunk
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_scene_dict(value, key=None):
    """Make pre-creation and post-creation prim paths comparable."""
    if isinstance(value, dict):
        return {name: _normalize_scene_dict(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_normalize_scene_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_scene_dict(item) for item in value)
    if key == "prim_path" and isinstance(value, str):
        return value.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    return value


def _set_runtime_fields(cfg, num_envs, device, usd_dir, offset):
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    cfg.sim.device = device
    cfg.scene.robot.spawn.usd_dir = str(usd_dir.resolve())
    cfg.commands.motion.training_iteration_offset = float(offset)


def verify_registry(cfg, task, usd_dir, offset):
    registered = gym.spec(task).kwargs["env_cfg_entry_point"]()
    _set_runtime_fields(registered, cfg.scene.num_envs, cfg.sim.device, usd_dir, offset)
    registered.commands.motion.motion_file = cfg.commands.motion.motion_file
    registered.episode_length_s = cfg.episode_length_s
    registered.commands.motion.debug_vis = cfg.commands.motion.debug_vis
    registered.scene.contact_forces.debug_vis = cfg.scene.contact_forces.debug_vis
    registered.scene.terrain.visual_material = cfg.scene.terrain.visual_material
    registered.viewer = cfg.viewer
    for scene_cfg in (registered.scene, cfg.scene):
        scene_cfg.terrain.num_envs = scene_cfg.num_envs
        scene_cfg.terrain.env_spacing = scene_cfg.env_spacing
    # Scene expansion normally happens in InteractiveScene construction.  The
    # source pickle is post-expansion while the registry cfg is pre-expansion.
    source_dict = _normalize_scene_dict(cfg.to_dict())
    registered_dict = _normalize_scene_dict(registered.to_dict())
    changes = {key: value for key, value in config_changes(source_dict, registered_dict).items()
               if value != ["None", "None"]}
    if changes:
        raise RuntimeError(f"Registered F{task.rsplit('-F', 1)[-1].split('-')[0]} task differs from source: {changes}")


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite preflight value: {name}")


def _sensor_force(sensor):
    matrix = sensor.data.force_matrix_w
    if matrix is None or matrix.ndim != 4 or matrix.shape[1] != 1 or matrix.shape[2] == 0 or matrix.shape[3] != 3:
        raise RuntimeError(f"Ground sensor filter is empty or unresolved: {matrix}")
    _finite(matrix, "ground_contact_force_matrix")
    return matrix, torch.linalg.vector_norm(matrix, dim=-1).sum(dim=(1, 2))


def _resolve_scene_entity_params(params, scene):
    """Resolve copied SceneEntityCfg values without mutating the environment cfg."""
    for value in params.values():
        if hasattr(value, "resolve"):
            value.resolve(scene)


@torch.inference_mode()
def run_forearm_preflight(runner, output, variant):
    """Run the required full-start rollout while preserving training state."""
    env = runner.env
    base = env.unwrapped
    command = base.command_manager.get_term("motion")
    sensor_names = (
        "getup_ground_elbow_yaw_l", "getup_ground_wrist_end_l",
        "getup_ground_elbow_yaw_r", "getup_ground_wrist_end_r",
    )
    missing = [name for name in sensor_names if name not in base.scene.sensors]
    if missing:
        raise RuntimeError(f"Missing forearm ground sensors: {missing}")
    active_terms = set(base.reward_manager.active_terms)
    if "getup_forearm_vertical" not in active_terms:
        raise RuntimeError("getup_forearm_vertical is not an active reward term")
    vertical_term = base.reward_manager.get_term_cfg("getup_forearm_vertical")
    # IsaacLab omits zero-weight terms from RewardManager.active_terms.  The
    # slip ablation still needs a raw functional smoke check in F1/F2.
    slip_term = copy.deepcopy(base.cfg.rewards.getup_wrist_support_slip)
    _resolve_scene_entity_params(slip_term.params, base.scene)
    old_play = command._play_from_start
    old_diagnostics = getattr(env, "diagnostics_enabled", None)
    old_common = getattr(base, "common_step_counter", None)
    old_sim = getattr(base, "_sim_step_counter", None)
    norm_counts = (int(runner.obs_normalizer.count), int(runner.privileged_obs_normalizer.count))
    rng = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        np.random.get_state(),
        random.getstate(),
    )
    frames = []
    sensor_shapes = {}
    max_wrist_force = 0.0
    vertical_raw_max = 0.0
    slip_raw_max = 0.0
    vertical_weighted_max = 0.0
    slip_weighted_max = 0.0
    try:
        if old_diagnostics is not None:
            env.diagnostics_enabled = False
        command._play_from_start = True
        obs, _ = env.reset()
        policy = runner.get_inference_policy(device=runner.device)
        for _ in range(FULL_FRAMES):
            frame = command.time_steps.detach().clone()
            _finite(frame, "reference_frame")
            frames.append(frame.cpu().numpy())
            for name in sensor_names:
                matrix, force = _sensor_force(base.scene.sensors[name])
                sensor_shapes[name] = list(matrix.shape)
                if "wrist_end" in name:
                    max_wrist_force = max(max_wrist_force, float(force.max()))
            vertical = vertical_term.func(base, **vertical_term.params)
            slip = slip_term.func(base, **slip_term.params)
            _finite(vertical, "getup_forearm_vertical_raw")
            _finite(slip, "getup_wrist_support_slip_raw")
            vertical_raw_max = max(vertical_raw_max, float(vertical.abs().max()))
            slip_raw_max = max(slip_raw_max, float(slip.abs().max()))
            vertical_weighted_max = max(vertical_weighted_max, float((vertical * vertical_term.weight).abs().max()))
            slip_weighted_max = max(slip_weighted_max, float((slip * slip_term.weight).abs().max()))
            obs, _, _, _ = env.step(policy(obs))
            _finite(obs, "policy_observation")
        frame_array = np.stack(frames)
        expected = np.arange(FULL_FRAMES)[:, None]
        uninterrupted = bool(np.all(frame_array == expected))
        if not uninterrupted:
            raise RuntimeError("Preflight rollout did not remain on frames 0..378 for every environment")
        if max_wrist_force <= 0.0:
            raise RuntimeError("No nonzero wrist ground force was observed in the full rollout")
        if vertical_raw_max <= 0.0 or slip_raw_max <= 0.0:
            raise RuntimeError("Forearm reward raw scores were never nonzero")
        if variant in {"f3", "f4"} and slip_weighted_max <= 0.0:
            raise RuntimeError("Enabled wrist-slip reward remained identically zero")
        if norm_counts != (int(runner.obs_normalizer.count), int(runner.privileged_obs_normalizer.count)):
            raise RuntimeError("Preflight changed normalization counts")
        result = {
            "frames": FULL_FRAMES,
            "num_envs": base.num_envs,
            "uninterrupted_from_frame_zero": uninterrupted,
            "sensor_shapes": sensor_shapes,
            "max_wrist_ground_force": max_wrist_force,
            "vertical_raw_abs_max": vertical_raw_max,
            "vertical_weighted_abs_max": vertical_weighted_max,
            "slip_raw_abs_max": slip_raw_max,
            "slip_weighted_abs_max": slip_weighted_max,
            "vertical_weight": float(vertical_term.weight),
            "slip_weight": float(slip_term.weight),
        }
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "forearm_preflight_frames.npz", frame=frame_array)
        (output / "forearm_preflight.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        command._play_from_start = old_play
        base.common_step_counter = old_common
        if old_sim is not None:
            base._sim_step_counter = old_sim
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2])
        random.setstate(rng[3])
        try:
            # Consume the restored training reset stream, then restore counters
            # and RNG once more so evaluation does not advance curriculum state.
            env.reset()
        finally:
            base.common_step_counter = old_common
            if old_sim is not None:
                base._sim_step_counter = old_sim
            torch.set_rng_state(rng[0])
            if rng[1] is not None:
                torch.cuda.set_rng_state_all(rng[1])
            np.random.set_state(rng[2])
            random.setstate(rng[3])
            if old_diagnostics is not None:
                env.diagnostics_enabled = old_diagnostics
            runner.train_mode()


def _manifest_url():
    run_id = os.environ.get("WANDB_RUN_ID")
    entity = os.environ.get("WANDB_ENTITY", "gkmbgm-northeastern-university")
    project = os.environ.get("WANDB_PROJECT", "urkl-t800")
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}" if run_id else None


class ForearmRunner(MotionOnPolicyRunner):
    """Reject non-finite PPO losses and keep launch status auditable."""

    def log(self, locs, *log_args, **kwargs):
        if any(not math.isfinite(float(value)) for value in locs["loss_dict"].values()):
            raise RuntimeError("Non-finite forearm PPO loss")
        super().log(locs, *log_args, **kwargs)
        iteration = int(locs["it"])
        if iteration == RESUME_ITERATION or iteration % 250 == 0:
            self.forearm_manifest.update(
                status="preflight_training" if args.preflight else "training",
                last_iteration=iteration,
            )
            self.forearm_manifest_path.write_text(json.dumps(self.forearm_manifest, indent=2))


def prepare():
    source = args.source_run.resolve()
    output = args.output.resolve()
    checkpoint = source / "model_18500.pt"
    if not source.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing source run/checkpoint: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from source run")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Source model_18500.pt SHA256 does not match the authorized C checkpoint")
    cfg = load_pickle(str(source / "params/env.pkl"))
    agent = load_pickle(str(source / "params/agent.pkl"))
    source_hashes = {
        name: sha256(source / "params" / name) for name in ("env.pkl", "agent.pkl")
    }
    motion = Path(cfg.commands.motion.motion_file).resolve()
    if sha256(motion) != MOTION_SHA256:
        raise ValueError("Source configuration does not reference the verified C motion")
    if getattr(cfg.commands.motion, "aggressive_variant", None) != "c":
        raise ValueError("Source configuration is not the C aggressive curriculum")
    if int(agent.seed) != 42:
        raise ValueError(f"Source seed must be 42, got {agent.seed}")
    baseline = cfg.to_dict()
    apply_getup_forearm_experiment(cfg, args.variant)
    experiment_changes = config_changes(baseline, cfg.to_dict())
    if not experiment_changes:
        raise ValueError("Forearm experiment produced no configuration delta")
    allowed_prefixes = (
        "scene.getup_ground_",
        "rewards.getup_forearm_vertical",
        "rewards.getup_wrist_support_slip",
    )
    unexpected = [path for path in experiment_changes
                  if not path.startswith(allowed_prefixes)]
    required_paths = {
        "scene.getup_ground_elbow_yaw_l",
        "scene.getup_ground_wrist_end_l",
        "scene.getup_ground_elbow_yaw_r",
        "scene.getup_ground_wrist_end_r",
        "rewards.getup_forearm_vertical",
        "rewards.getup_wrist_support_slip",
    }
    if unexpected or not required_paths <= set(experiment_changes):
        raise ValueError(
            f"Forearm experiment changed outside the sensor/reward allowlist: "
            f"unexpected={unexpected}, missing={sorted(required_paths - set(experiment_changes))}"
        )
    if getattr(cfg.commands.motion, "aggressive_variant", None) != "c":
        raise ValueError("Forearm experiment changed the C curriculum")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_dir = str(args.usd_dir.resolve())
    cfg.commands.motion.training_iteration_offset = float(RESUME_ITERATION)
    if cfg.commands.motion.play_from_start or cfg.commands.motion.resample_at_motion_end:
        # C training must retain its adaptive reset curriculum and non-cyclic clip.
        raise ValueError("Unexpected C motion reset/end configuration")
    agent.device = args.device
    agent.resume = False
    agent.load_run = ""
    agent.load_checkpoint = ""
    agent.run_name = output.name
    agent.max_iterations = args.iterations
    agent.save_interval = 250
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.schedule = "fixed"
    agent.logger = "tensorboard" if args.preflight else "wandb"
    agent.wandb_project = "urkl-t800"
    task = f"Tracking-Flat-T800-GetUp-Aggressive-C-Forearm-{args.variant.upper()}-v0"
    verify_registry(cfg, task, args.usd_dir, RESUME_ITERATION)
    return source, output, checkpoint, motion, cfg, agent, task, source_hashes, experiment_changes


def main():
    source, output, checkpoint, motion, cfg, agent, task, source_hashes, experiment_changes = prepare()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "finetune_manifest.json"
    manifest = {
        "status": "initializing",
        "variant": args.variant,
        "task": task,
        "source_run": str(source),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_snapshot_sha256": source_hashes,
        "motion": str(motion),
        "motion_sha256": sha256(motion),
        "experiment_diff": experiment_changes,
        "training_iteration_offset": RESUME_ITERATION,
        "source_checkpoint_iteration": 18_500,
        "resume_iteration": RESUME_ITERATION,
        "num_envs": args.num_envs,
        "additional_iterations": args.iterations,
        "save_interval": 250,
        "seed": 42,
        "learning_rate": args.learning_rate,
        "schedule": "fixed",
        "preflight": args.preflight,
        "wandb_url": _manifest_url(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "implementation_sha256": {
            "scripts/rsl_rl/finetune_getup_forearm.py": sha256(REPO / "scripts/rsl_rl/finetune_getup_forearm.py"),
            "scripts/slurm/finetune_t800_getup_forearm_4x3090.sbatch": sha256(
                REPO / "scripts/slurm/finetune_t800_getup_forearm_4x3090.sbatch"
            ),
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/getup_forearm_env_cfg.py": sha256(
                REPO / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/getup_forearm_env_cfg.py"
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    dump_pickle(str(output / "params/env.pkl"), cfg)
    dump_pickle(str(output / "params/agent.pkl"), agent)
    env = None
    try:
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
        # This must remain an actor/critic contract invariant for all F variants.
        obs, extras = env.get_observations()
        critic = extras["observations"]["critic"]
        if tuple(obs.shape[1:]) != (ACTOR_OBS,) or tuple(critic.shape[1:]) != (CRITIC_OBS,):
            raise RuntimeError(f"Unexpected actor/critic widths: {tuple(obs.shape)} / {tuple(critic.shape)}")
        command = env.unwrapped.command_manager.get_term("motion")
        if command.motion.time_step_total != FULL_FRAMES or command.motion.fps != 50:
            raise RuntimeError("Unexpected source motion frame count/fps")
        runner = ForearmRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        runner.forearm_manifest = manifest
        runner.forearm_manifest_path = manifest_path
        restored = restore_finetune_checkpoint(runner, checkpoint, args.learning_rate)
        if restored["first_iteration"] != RESUME_ITERATION:
            raise RuntimeError(f"Checkpoint resume iteration mismatch: {restored}")
        manifest.update(
            status="ready_to_preflight" if args.preflight else "ready_to_train",
            restored=restored,
            observation_widths={"actor": ACTOR_OBS, "critic": CRITIC_OBS},
            registered_task_control_config_verified=True,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
        runner.logger_type = agent.logger
        runner.save(str(output / "model_init.pt"))
        if args.preflight:
            learn_chunk(runner, 2)
            if runner.current_learning_iteration != RESUME_ITERATION + 1:
                raise RuntimeError(f"Unexpected preflight final iteration: {runner.current_learning_iteration}")
            preflight = run_forearm_preflight(runner, output, args.variant)
            if sha256(checkpoint) != CHECKPOINT_SHA256 or sha256(motion) != MOTION_SHA256:
                raise RuntimeError("Source checkpoint or motion changed during preflight")
            manifest.update(
                status="preflight_completed",
                completed_updates=2,
                final_iteration=runner.current_learning_iteration,
                preflight=preflight,
            )
        else:
            learn_chunk(runner, args.iterations)
            expected_final = RESUME_ITERATION + args.iterations - 1
            if runner.current_learning_iteration != expected_final:
                raise RuntimeError(f"Unexpected final iteration: {runner.current_learning_iteration}")
            if not all(torch.isfinite(value).all() for value in runner.alg.policy.state_dict().values()):
                raise RuntimeError("Non-finite final policy")
            if any(group["lr"] != args.learning_rate for group in runner.alg.optimizer.param_groups):
                raise RuntimeError("Fixed fine-tuning learning rate changed")
            final_checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
            if not final_checkpoint.is_file():
                raise RuntimeError(f"Missing final checkpoint: {final_checkpoint}")
            if sha256(checkpoint) != CHECKPOINT_SHA256 or sha256(motion) != MOTION_SHA256:
                raise RuntimeError("Source checkpoint or motion changed during training")
            manifest.update(
                status="completed",
                completed_updates=args.iterations,
                final_iteration=runner.current_learning_iteration,
                final_checkpoint=str(final_checkpoint),
                final_checkpoint_sha256=sha256(final_checkpoint),
            )
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[FOREARM_FINETUNE] status={manifest['status']} variant={args.variant} output={output}", flush=True)
        if runner.writer is not None:
            runner.writer.flush()
        if not args.preflight:
            import wandb
            wandb.finish()
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    finally:
        if env is not None:
            env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    app.close()
    raise
else:
    app.close()
