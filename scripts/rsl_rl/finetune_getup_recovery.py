"""Run the two bounded recovery ablations from the trusted j9519 snapshots.

The environment and PPO configuration come from the source run, not the
currently modified Smooth task defaults. No source checkpoint is overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("ori", "ori_velgate"), required=True)
parser.add_argument("--source-run", type=Path, required=True)
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
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_pickle, dump_yaml, load_pickle
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.config.t800.getup_recovery_env_cfg import (
    apply_getup_recovery_experiment,
)
from whole_body_tracking.utils.getup_finetune import (
    assert_reward_only_changes,
    config_changes,
    restore_finetune_checkpoint,
)
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source = args.source_run.resolve()
    output = args.output.resolve()
    checkpoint = source / "model_29999.pt"
    if output == source or source in output.parents:
        raise ValueError("Finetuning output must be separate from the source run")
    output.mkdir(parents=True, exist_ok=False)
    # Trusted, locally produced snapshots only.
    cfg = load_pickle(str(source / "params/env.pkl"))
    agent = load_pickle(str(source / "params/agent.pkl"))
    motion = Path(cfg.commands.motion.motion_file).resolve()
    expected = {
        "checkpoint": "29d3d84f47f62b5d8e1c7e9de281cc21ddfb5da575dce869f70257cb6dd8015d",
        "motion": "e5a1711e492c7fadac84741d11fb09f0ebaa8fe659b68ab3674cb6b8d482812d",
    }
    actual = {"checkpoint": sha256(checkpoint), "motion": sha256(motion)}
    if actual != expected:
        raise ValueError(f"Unexpected source checkpoint/motion: {actual}")
    if (cfg.rewards.motion_global_anchor_pos.weight, cfg.rewards.motion_global_anchor_ori.weight,
            cfg.rewards.getup_anchor_height.weight, cfg.rewards.getup_transition_joint_pos.weight) != (.5, .5, 1.25, 1.5):
        raise ValueError("Source is not the original j9519 reward configuration")
    baseline = cfg.to_dict()
    apply_getup_recovery_experiment(cfg, args.variant)
    reward_changes = config_changes(baseline, cfg.to_dict())
    assert_reward_only_changes(reward_changes, args.variant)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed  # Same seed=42 and startup distribution in both arms.
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_dir = str(args.usd_dir.resolve())
    if not cfg.commands.motion.play_from_start or cfg.commands.motion.resample_at_motion_end:
        raise ValueError("Expected frame-zero resets and native non-cyclic episodes")
    agent.device = args.device
    agent.resume = True
    agent.load_run = source.name
    agent.load_checkpoint = checkpoint.name
    agent.run_name = output.name
    agent.max_iterations = args.iterations
    agent.save_interval = 250
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.schedule = "fixed"
    agent.logger = "tensorboard" if args.preflight else "wandb"
    task = "Tracking-Flat-T800-GetUp-Recovery-" + ("Ori" if args.variant == "ori" else "OriVelGate") + "-v0"
    # The registered Play task must reconstruct the same control environment.
    # Normalize only launch-time/visual settings before checking equivalence.
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
    # InteractiveScene._add_entities_from_cfg performs these expansions;
    # the source was saved after scene creation, the registry is pre-creation.
    for scene_cfg in (registered.scene, cfg.scene):
        scene_cfg.robot.prim_path = scene_cfg.robot.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
        scene_cfg.contact_forces.prim_path = scene_cfg.contact_forces.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
        scene_cfg.terrain.num_envs = scene_cfg.num_envs
        scene_cfg.terrain.env_spacing = scene_cfg.env_spacing
    registry_changes = config_changes(cfg.to_dict(), registered.to_dict())
    # Missing disabled terms and explicit None are operationally equivalent.
    registry_changes = {k: v for k, v in registry_changes.items() if v != ["None", "None"]}
    if registry_changes:
        raise ValueError(f"Registered Play task differs from snapshot experiment: {registry_changes}")
    manifest = {
        "status": "initializing", "variant": args.variant, "task": task,
        "source_run": str(source), "source_checkpoint": str(checkpoint), "motion": str(motion),
        "sha256": actual, "reward_changes": reward_changes,
        "num_envs": args.num_envs, "additional_iterations": args.iterations,
        "seed": agent.seed, "save_interval": agent.save_interval,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preflight": args.preflight,
        "registered_task_control_config_verified": True,
        "source_snapshot_sha256": {
            name: sha256(source / "params" / name) for name in ("env.pkl", "agent.pkl")
        },
    }
    manifest_path = output / "finetune_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    dump_pickle(str(output / "params/env.pkl"), cfg)
    dump_pickle(str(output / "params/agent.pkl"), agent)
    print(f"[FINETUNE] output={output} task={task} changes={reward_changes}", flush=True)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    try:
        runner = MotionOnPolicyRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        restored = restore_finetune_checkpoint(runner, checkpoint, args.learning_rate)
        manifest.update(restored)
        manifest["status"] = "ready_to_train"
        manifest["last_planned_iteration"] = restored["first_iteration"] + args.iterations - 1
        obs, extras = env.get_observations()
        if obs.shape[1] != 134 or extras["observations"]["critic"].shape[1] != 284:
            raise RuntimeError("Original actor/critic observation contract changed")
        manifest["observation_widths"] = [134, 284]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[FINETUNE] verified_resume={restored}", flush=True)
        # Upstream initializes this in learn(); the initial local snapshot is earlier.
        runner.logger_type = agent.logger
        runner.save(str(output / "model_init.pt"))
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        if not all(torch.isfinite(value).all() for value in runner.alg.policy.state_dict().values()):
            raise RuntimeError("Non-finite trained policy")
        if runner.alg.learning_rate != args.learning_rate or any(
            group["lr"] != args.learning_rate for group in runner.alg.optimizer.param_groups
        ):
            raise RuntimeError("Finetune learning rate changed unexpectedly")
        final_checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError("Expected final checkpoint was not saved")
        if {"checkpoint": sha256(checkpoint), "motion": sha256(motion)} != actual:
            raise RuntimeError("Source checkpoint or motion was modified")
        manifest.update(status="completed", final_checkpoint=str(final_checkpoint))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if runner.writer is not None:
            runner.writer.flush()
        if not args.preflight:
            import wandb
            wandb.finish()
        print(f"[FINETUNE] completed={output}", flush=True)
    finally:
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
