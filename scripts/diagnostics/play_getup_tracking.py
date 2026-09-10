"""Replay a trusted local training snapshot without exporting or changing a policy.

Samples are PRE-ACTION states: each sample's reference frame, robot state and
instantaneous rewards refer to the same simulation instant. Auto-reset states
are separated into episodes instead of being mistaken for falls.
"""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--checkpoint", default="model_29999.pt")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd-dir", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1:
    parser.error("--episodes must be positive")
args.enable_cameras = args.video
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.io import dump_yaml, load_pickle
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.rsl_rl_checkpoint import load_rsl_rl_checkpoint_for_inference


def main():
    run = args.run.resolve()
    checkpoint = run / args.checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "rollout.npz").exists():
        raise FileExistsError(f"Refusing to replace an existing rollout: {args.output}")

    # These are trusted, locally produced snapshots, not arbitrary downloaded pickles.
    cfg = load_pickle(str(run / "params/env.pkl"))
    runner_cfg = load_pickle(str(run / "params/agent.pkl"))
    motion = Path(cfg.commands.motion.motion_file).resolve()
    hashes = {"checkpoint": sha256(checkpoint), "motion": sha256(motion)}
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_dir = str(args.usd_dir.resolve())
    # Visual-only overrides avoid remote debug assets and keep the camera fixed.
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.terrain.visual_material = None
    cfg.viewer.origin_type = "world"
    cfg.viewer.asset_name = None
    cfg.viewer.body_name = None
    cfg.viewer.eye = (-2.6, -2.6, 1.7)
    cfg.viewer.lookat = (0.0, 0.15, 0.65)
    cfg.viewer.resolution = (960, 720)
    runner_cfg.device = args.device
    runner_cfg.seed = args.seed
    if not cfg.commands.motion.play_from_start or cfg.commands.motion.resample_at_motion_end:
        raise ValueError("This diagnostic requires the saved start-frame/non-resampling setup")
    dump_yaml(str(args.output / "effective_env.yaml"), cfg)
    dump_yaml(str(args.output / "effective_agent.yaml"), runner_cfg)

    raw_env = gym.make(
        "Tracking-Flat-T800-GetUp-Smooth-v0", cfg=cfg,
        render_mode="rgb_array" if args.video else None,
    )
    base = raw_env.unwrapped
    if args.video:
        raw_env = gym.wrappers.RecordVideo(
            raw_env, video_folder=str(args.output / "video"),
            step_trigger=lambda step: step == 0,
            video_length=base.max_episode_length * args.episodes,
            name_prefix="j9519-tracking", disable_logger=True,
        )
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    load_rsl_rl_checkpoint_for_inference(runner, str(checkpoint))
    source = torch.load(checkpoint, map_location=args.device, weights_only=False)["model_state_dict"]
    loaded = runner.alg.policy.state_dict()
    actor_keys = [key for key in source if key.startswith("actor.")]
    if not actor_keys or any(key not in loaded or not torch.equal(source[key], loaded[key]) for key in actor_keys):
        raise RuntimeError("Actor checkpoint verification failed")
    policy = runner.get_inference_policy(device=args.device)
    observations = env.get_observations()[0]
    command = base.command_manager.get_term("motion")
    robot = base.scene["robot"]
    contact = base.scene.sensors["contact_forces"]
    masses = robot.root_physx_view.get_masses().to(base.device)
    joint_names = list(command.cfg.motion_joint_names)
    reward_names = list(base.reward_manager.active_terms)
    reward_cfgs = [base.reward_manager.get_term_cfg(name) for name in reward_names]
    metadata = {
        "run": str(run), "checkpoint": str(checkpoint), "motion": str(motion),
        "sha256": hashes, "seed": args.seed, "episodes_requested": args.episodes,
        "step_dt": base.step_dt, "max_episode_length": base.max_episode_length,
        "sampling": "pre_action", "joint_names": joint_names,
        "body_names": list(robot.body_names), "contact_body_names": list(contact.body_names),
        "torque_joint_names": list(robot.joint_names),
        "reward_names": reward_names, "reward_weights": [term.weight for term in reward_cfgs],
        "actor_weights_verified": len(actor_keys),
        "observation_noise": cfg.observations.policy.enable_corruption,
        "note": "Native timeout, original rewards and physics randomization; repeated episodes share startup material/COM randomization.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    samples = {}
    episode = 0

    def record(name, tensor):
        value = tensor.detach().cpu().numpy().copy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        samples.setdefault(name, []).append(value)

    try:
        with torch.inference_mode():
            for step in range(base.max_episode_length * args.episodes + args.episodes):
                record("episode", episode)
                record("frame", command.time_steps[0])
                record("episode_step", base.episode_length_buf[0])
                for name, value in {
                    "ref_joint_pos": command.joint_pos,
                    "joint_pos": command.robot_joint_pos,
                    "ref_joint_vel": command.joint_vel,
                    "joint_vel": command.robot_joint_vel,
                    "ref_anchor_pos": command.anchor_pos_w,
                    "anchor_pos": command.robot_anchor_pos_w,
                    "ref_anchor_quat": command.anchor_quat_w,
                    "anchor_quat": command.robot_anchor_quat_w,
                    "ref_anchor_lin_vel": command.anchor_lin_vel_w,
                    "anchor_lin_vel": command.robot_anchor_lin_vel_w,
                    "ref_anchor_ang_vel": command.anchor_ang_vel_w,
                    "anchor_ang_vel": command.robot_anchor_ang_vel_w,
                    "body_pos": robot.data.body_pos_w,
                    "body_quat": robot.data.body_quat_w,
                    "body_lin_vel": robot.data.body_lin_vel_w,
                    "contact_force": contact.data.net_forces_w,
                    "joint_torque": robot.data.applied_torque,
                }.items():
                    record(name, value[0])
                com = (robot.data.body_com_pos_w * masses[..., None]).sum(dim=1) / masses.sum(dim=1, keepdim=True)
                com_velocity = (robot.data.body_com_lin_vel_w * masses[..., None]).sum(dim=1) / masses.sum(dim=1, keepdim=True)
                record("com_pos", com[0])
                record("com_lin_vel", com_velocity[0])
                record("reward_raw", torch.stack([term.func(base, **term.params)[0] for term in reward_cfgs]))
                actions = policy(observations)
                record("action", actions[0])
                observations, _, done, _ = env.step(actions)
                record("done_after_action", done[0])
                if step % 100 == 0:
                    print(f"[DIAG] sample={step} episode={episode} frame={int(command.time_steps[0])}", flush=True)
                if bool(done[0]):
                    print(f"[DIAG] completed episode={episode} steps={int(samples['episode_step'][-1]) + 1}", flush=True)
                    episode += 1
                    if episode >= args.episodes:
                        break
        metadata["episodes_completed"] = episode
        metadata["samples"] = len(samples.get("frame", []))
        if {"checkpoint": sha256(checkpoint), "motion": sha256(motion)} != hashes:
            raise RuntimeError("Input checkpoint or motion changed during replay")
    finally:
        if samples:
            np.savez_compressed(args.output / "rollout.npz", **{key: np.stack(values) for key, values in samples.items()})
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        env.close()
    print(f"[DIAG] finished: {args.output}", flush=True)


try:
    main()
except BaseException:
    # Kit shutdown may exit the process, so print the failure before closing.
    import traceback
    traceback.print_exc()
    app.close()
    raise
else:
    app.close()
