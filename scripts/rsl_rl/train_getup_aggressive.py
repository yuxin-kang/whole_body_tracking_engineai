"""Train A/B/C/D from fresh policy, optimizer and normalization state."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=tuple("abcd"), required=True)
parser.add_argument("--source-run", type=Path, required=True, help="Trusted configuration source only, never model weights.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--preflight", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1:
    parser.error("Environment and iteration counts must be positive")
if args.preflight and args.iterations != 24:
    parser.error("Preflight requires 24 updates: three 8-update stage checks")
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_pickle, dump_yaml, load_pickle
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.config.t800.getup_aggressive_env_cfg import apply_getup_aggressive_experiment
from whole_body_tracking.tasks.tracking.mdp.getup_aggressive_commands import validate_aggressive_reference_clip
from whole_body_tracking.utils.getup_finetune import config_changes
from whole_body_tracking.utils.getup_scratch import (
    assert_preflight_result, audit_scratch_initialization, evaluate_full_start, learn_chunk,
)
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_registry(cfg, task):
    registered = gym.spec(task).kwargs["env_cfg_entry_point"]()
    registered.scene.num_envs = cfg.scene.num_envs
    registered.seed = cfg.seed
    registered.sim.device = cfg.sim.device
    registered.scene.robot.spawn.usd_dir = cfg.scene.robot.spawn.usd_dir
    registered.commands.motion.motion_file = cfg.commands.motion.motion_file
    registered.episode_length_s = cfg.episode_length_s
    registered.commands.motion.debug_vis = cfg.commands.motion.debug_vis
    registered.scene.contact_forces.debug_vis = cfg.scene.contact_forces.debug_vis
    registered.scene.terrain.visual_material = cfg.scene.terrain.visual_material
    registered.viewer = cfg.viewer
    for scene_cfg in (registered.scene, cfg.scene):
        scene_cfg.robot.prim_path = scene_cfg.robot.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
        scene_cfg.contact_forces.prim_path = scene_cfg.contact_forces.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
        scene_cfg.terrain.num_envs = scene_cfg.num_envs
        scene_cfg.terrain.env_spacing = scene_cfg.env_spacing
    changes = {k: v for k, v in config_changes(cfg.to_dict(), registered.to_dict()).items() if v != ["None", "None"]}
    if changes:
        raise ValueError(f"Registered Play task differs from training: {changes}")


def prepare_configs():
    source = args.source_run.resolve()
    snapshot_hashes = {name: sha256(source / "params" / name) for name in ("env.pkl", "agent.pkl")}
    expected = {"env.pkl": "36f539ff9e573d9a664fe9e3326a93db69717e76c7d2fe2159d3bf709d255bf1",
                "agent.pkl": "eaa932e20744d71399826532b01f3031bf385a6a12c0a46923ed5b527ce5c8d5"}
    if snapshot_hashes != expected:
        raise ValueError(f"Unexpected source configuration snapshots: {snapshot_hashes}")
    cfg = load_pickle(str(source / "params/env.pkl"))
    agent = load_pickle(str(source / "params/agent.pkl"))
    motion = Path(cfg.commands.motion.motion_file).resolve()
    motion_hash = sha256(motion)
    if motion_hash != "e5a1711e492c7fadac84741d11fb09f0ebaa8fe659b68ab3674cb6b8d482812d":
        raise ValueError("Motion is not the verified faint_prone_getup_03 clip")
    before = cfg.to_dict()
    apply_getup_aggressive_experiment(cfg, args.variant)
    changes = config_changes(before, cfg.to_dict())
    allowed = ("rewards.", "commands.motion.", "terminations.")
    if not changes or any(not path.startswith(allowed) for path in changes):
        raise ValueError(f"Unexpected non-experiment config changes: {changes}")
    if cfg.commands.motion.motion_file != before["commands"]["motion"]["motion_file"]:
        raise ValueError("Experiment changed the reference motion")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_dir = str(args.usd_dir.resolve())
    agent.device = args.device
    agent.resume = False
    agent.load_run = ""
    agent.load_checkpoint = ""
    agent.run_name = args.output.name
    agent.max_iterations = args.iterations
    agent.save_interval = 500
    agent.logger = "tensorboard" if args.preflight else "wandb"
    agent.wandb_project = "urkl-t800"
    if (agent.seed, agent.num_steps_per_env, agent.algorithm.learning_rate, agent.algorithm.schedule) != (42, 48, .001, "adaptive"):
        raise ValueError("Unexpected from-scratch PPO configuration")
    task = f"Tracking-Flat-T800-GetUp-Aggressive-{args.variant.upper()}-v0"
    verify_registry(cfg, task)
    manifest = {
        "status": "initializing", "variant": args.variant, "task": task,
        "configuration_source": str(source), "source_checkpoint": None, "resume": False,
        "motion": str(motion), "motion_sha256": motion_hash, "source_snapshot_sha256": snapshot_hashes,
        "experiment_changes": changes, "num_envs": args.num_envs, "num_steps_per_env": 48,
        "planned_updates": args.iterations, "save_interval": 500, "evaluation_interval_updates": 5000,
        "seed": agent.seed, "schedule": agent.algorithm.schedule, "preflight": args.preflight,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "registered_task_control_config_verified": True,
        "wandb_url": None if args.preflight else
            f"https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/{os.environ.get('WANDB_RUN_ID')}",
    }
    repo = Path(__file__).resolve().parents[2]
    tracked_sources = [
        "scripts/rsl_rl/train_getup_aggressive.py", "scripts/slurm/train_t800_getup_4x3090_aggressive.sbatch",
        "source/whole_body_tracking/whole_body_tracking/utils/getup_scratch.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/getup_aggressive_commands.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/getup_aggressive_rewards.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/getup_aggressive_env_cfg.py",
    ]
    manifest["implementation_sha256"] = {name: sha256(repo / name) for name in tracked_sources}
    return cfg, agent, task, manifest


class AuditedRunner(MotionOnPolicyRunner):
    def log(self, locs, *log_args, **kwargs):
        if any(not math.isfinite(float(value)) for value in locs["loss_dict"].values()):
            raise RuntimeError("Non-finite PPO loss")
        super().log(locs, *log_args, **kwargs)
        if locs["it"] % 500 == 0:
            self.manifest.update(status="training", last_update=int(locs["it"]),
                                 current_learning_rate=self.alg.learning_rate)
            self.manifest_path.write_text(json.dumps(self.manifest, indent=2))


class DiagnosticVecEnv(RslRlVecEnvWrapper):
    """Small preflight-only instrumentation; formal runs have no per-step diagnostic sync."""

    def __init__(self, env):
        self.diagnostics_enabled = False
        super().__init__(env)
        self.diagnostics = {
            "reset_source_counts": [0, 0, 0], "physical_source_steps": [0, 0, 0],
            "reset_frames": [], "peak_contact_force_n": 0.0,
            "sampled_contact_impulse_proxy_ns": 0.0, "max_reference_joint_limit_violation_rad": 0.0,
        }
        self.diagnostics_enabled = True

    def reset(self):
        result = super().reset()
        if self.diagnostics_enabled:
            cmd = self.unwrapped.command_manager.get_term("motion")
            source = getattr(cmd, "reset_source", torch.zeros_like(cmd.time_steps))
            counts = torch.bincount(source, minlength=3).tolist()
            self.diagnostics["reset_source_counts"] = [a + b for a, b in zip(self.diagnostics["reset_source_counts"], counts)]
            self.diagnostics["reset_frames"].append(cmd.time_steps.tolist())
            limits = cmd.robot.data.soft_joint_pos_limits
            if cmd.robot_joint_indexes is not None:
                limits = limits[:, cmd.robot_joint_indexes]
            violation = torch.maximum(cmd.joint_pos - limits[..., 1], limits[..., 0] - cmd.joint_pos).clamp_min(0)
            self.diagnostics["max_reference_joint_limit_violation_rad"] = max(
                self.diagnostics["max_reference_joint_limit_violation_rad"], float(violation.max()))
        return result

    def step(self, actions):
        if self.diagnostics_enabled:
            cmd = self.unwrapped.command_manager.get_term("motion")
            source = getattr(cmd, "reset_source", torch.zeros_like(cmd.time_steps))
            counts = torch.bincount(source, minlength=3).tolist()
            self.diagnostics["physical_source_steps"] = [a + b for a, b in zip(self.diagnostics["physical_source_steps"], counts)]
        result = super().step(actions)
        if self.diagnostics_enabled:
            forces = self.unwrapped.scene.sensors["contact_forces"].data.net_forces_w
            if not torch.isfinite(forces).all():
                raise RuntimeError("Non-finite preflight contact force")
            magnitudes = torch.linalg.vector_norm(forces, dim=-1)
            self.diagnostics["peak_contact_force_n"] = max(self.diagnostics["peak_contact_force_n"], float(magnitudes.max()))
            self.diagnostics["sampled_contact_impulse_proxy_ns"] += float(magnitudes.sum()) * self.unwrapped.step_dt
            if torch.any(self.unwrapped.reset_terminated & self.unwrapped.reset_time_outs):
                raise RuntimeError("Preflight failure and timeout labels overlap")
        return result


def run_learning(runner, output):
    completed = 0
    while completed < args.iterations:
        chunk = min(8 if args.preflight else 5000, args.iterations - completed)
        command = runner.env.unwrapped.command_manager.get_term("motion")
        if args.preflight:
            if args.variant in "cd":
                command.training_iteration_offset = (
                    (0, 10000, 20000)[completed // 8] - runner.env.unwrapped.common_step_counter / 48
                )
            with torch.inference_mode():
                runner.env.reset()
        runner.current_learning_iteration = completed
        learn_chunk(runner, chunk)
        completed += chunk
        if runner.current_learning_iteration != completed - 1:
            raise RuntimeError("PPO update counter skipped or repeated an iteration")
        if not all(torch.isfinite(value).all() for value in runner.alg.policy.state_dict().values()):
            raise RuntimeError("Non-finite trained model")
        result = evaluate_full_start(runner, output / "evaluations" / f"update_{completed:06d}", completed - 1)
        if args.preflight:
            assert_preflight_result(result, runner.env.diagnostics, args.variant, completed * 48 * args.num_envs)
            runner.manifest["preflight_diagnostics"] = runner.env.diagnostics
        print(f"[SCRATCH EVAL] updates={completed} posture_pass={result['posture_pass_fraction']}", flush=True)
        if runner.writer is not None:
            for key in ("posture_pass_fraction", "transition_orientation_error_rad", "transition_position_error_m"):
                runner.writer.add_scalar(f"Eval_full_start/{key}", result[key], completed - 1)
        runner.manifest["last_evaluation"] = result
        runner.manifest["completed_updates"] = completed
        runner.manifest_path.write_text(json.dumps(runner.manifest, indent=2))


def main():
    cfg, agent, task, manifest = prepare_configs()
    output = args.output.resolve()
    source = args.source_run.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from the trusted source run")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "scratch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    dump_pickle(str(output / "params/env.pkl"), cfg)
    dump_pickle(str(output / "params/agent.pkl"), agent)
    print(f"[SCRATCH] output={output} task={task} checkpoint=None resume=False", flush=True)
    wrapper = DiagnosticVecEnv if args.preflight else RslRlVecEnvWrapper
    env = None
    try:
        env = wrapper(gym.make(task, cfg=cfg))
        # Reference-state sampling consumes a different amount of RNG in C/D.
        # Reset the network seed so all four candidates start from the same weights.
        torch.manual_seed(agent.seed)
        torch.cuda.manual_seed_all(agent.seed)
        runner = AuditedRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        runner.manifest, runner.manifest_path = manifest, manifest_path
        manifest.update(audit_scratch_initialization(runner))
        obs, extras = env.get_observations()
        if obs.shape[1] != 134 or extras["observations"]["critic"].shape[1] != 284:
            raise RuntimeError("Actor/critic observation contract changed")
        command = env.unwrapped.command_manager.get_term("motion")
        if command.motion.time_step_total != 379 or command.motion.fps != 50:
            raise RuntimeError("Unexpected reference motion dimensions")
        if not validate_aggressive_reference_clip(command):
            raise RuntimeError("Non-finite reference trajectory")
        gpu_apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,gpu_name", "--format=csv,noheader"], text=True)
        gpu_rows = [line for line in gpu_apps.splitlines() if line.split(",")[0].strip() == str(os.getpid())]
        if len(gpu_rows) != 1:
            raise RuntimeError(f"Cannot verify one physical GPU for this process: {gpu_rows}")
        manifest.update(status="ready_to_train", observation_widths=[134, 284], gpu_process=gpu_rows[0], pid=os.getpid())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        runner.logger_type = agent.logger
        runner.save(str(output / "model_init.pt"))
        print(f"[SCRATCH] initialization_verified={manifest}", flush=True)
        run_learning(runner, output)
        final_checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError("Missing final checkpoint")
        if sha256(Path(manifest["motion"])) != manifest["motion_sha256"]:
            raise RuntimeError("Reference motion changed during training")
        manifest.update(status="completed", final_checkpoint=str(final_checkpoint))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if runner.writer is not None:
            runner.writer.flush()
        if not args.preflight:
            import wandb
            wandb.finish()
        print(f"[SCRATCH] completed={output}", flush=True)
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
