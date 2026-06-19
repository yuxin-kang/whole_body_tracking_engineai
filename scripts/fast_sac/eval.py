from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FastSAC G1 checkpoint inside Isaac-only tasks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion_file", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402
from whole_body_tracking.rl.fast_sac import FastSACAgent  # noqa: E402
from whole_body_tracking.rl.fast_sac.artifacts import evaluate_policy, write_json_artifact  # noqa: E402
from whole_body_tracking.rl.fast_sac.runner import split_actor_critic_obs  # noqa: E402


def _load_checkpoint(path: str, agent: FastSACAgent) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    agent.load_checkpoint_state(payload)


def _get_num_actions(env) -> int:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_manager"):
        return int(unwrapped.action_manager.total_action_dim)
    return int(gym.spaces.flatdim(unwrapped.single_action_space))


@hydra_task_config(args_cli.task, "fast_sac_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    agent_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = agent_cfg.device
    env_cfg.seed = args_cli.seed
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    env_cfg.commands.motion.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    fast_sac_obs = split_actor_critic_obs(obs)
    agent = FastSACAgent(
        actor_obs_dim=int(fast_sac_obs.actor.shape[-1]),
        critic_obs_dim=int(fast_sac_obs.critic.shape[-1]),
        action_dim=_get_num_actions(env),
        cfg=agent_cfg,
    )
    checkpoint_path = str(Path(args_cli.checkpoint).expanduser().resolve())
    _load_checkpoint(checkpoint_path, agent)
    summary = evaluate_policy(
        env=env,
        agent=agent,
        split_obs=split_actor_critic_obs,
        num_episodes=args_cli.episodes,
        horizon=args_cli.horizon,
        deterministic=not args_cli.stochastic,
        task=args_cli.task,
        seed=args_cli.seed,
        motion_file=args_cli.motion_file,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=checkpoint_path,
    )
    output_path = Path(args_cli.output) if args_cli.output else Path(checkpoint_path).with_name("eval_summary.json")
    write_json_artifact(output_path, summary)
    print(f"[FastSAC Eval] wrote {output_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
