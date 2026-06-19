from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-full G1 with the local FastSAC runner.")
    parser.add_argument("--task", default="Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--motion_file", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--updates_per_step", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=None)
    parser.add_argument("--curve_interval", type=int, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--save_replay_buffer", action="store_true", default=False)
    parser.add_argument("--eval_episodes", type=int, default=16)
    parser.add_argument("--eval_horizon", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--logger", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default=None)
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
from whole_body_tracking.rl.fast_sac import FastSACAgent, FastSACReplayBuffer  # noqa: E402
from whole_body_tracking.rl.fast_sac.artifacts import (  # noqa: E402
    build_run_metadata,
    build_training_metrics_payload,
    evaluate_policy,
    write_json_artifact,
)
from whole_body_tracking.rl.fast_sac.runner import split_actor_critic_obs  # noqa: E402
from whole_body_tracking.rl.fast_sac.wandb_logging import (  # noqa: E402
    WandbLoggerConfig,
    create_wandb_logger,
)


def _update_agent_cfg(agent_cfg) -> None:
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_steps is not None:
        agent_cfg.max_steps = args_cli.max_steps
    if args_cli.warmup_steps is not None:
        agent_cfg.warmup_steps = args_cli.warmup_steps
    if args_cli.batch_size is not None:
        agent_cfg.batch_size = args_cli.batch_size
    if args_cli.updates_per_step is not None:
        agent_cfg.updates_per_step = args_cli.updates_per_step
    if args_cli.checkpoint_interval is not None:
        agent_cfg.checkpoint_interval = args_cli.checkpoint_interval
    if args_cli.resume_checkpoint is not None:
        agent_cfg.resume_checkpoint = args_cli.resume_checkpoint
    if args_cli.save_replay_buffer:
        agent_cfg.save_replay_buffer = True


def _make_log_dir(agent_cfg) -> Path:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = Path("logs") / "fast_sac" / agent_cfg.experiment_name / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _save_checkpoint(
    log_dir: Path,
    step: int,
    agent: FastSACAgent,
    replay_buffer: FastSACReplayBuffer,
    save_replay_buffer: bool,
) -> None:
    payload = agent.checkpoint_state()
    payload["step"] = step
    payload["replay_buffer"] = replay_buffer.state_dict(include_storage=save_replay_buffer)
    checkpoint_path = log_dir / f"model_{step}.pt"
    torch.save(payload, checkpoint_path)
    print(f"[FastSAC] checkpoint step={step} path={checkpoint_path}", flush=True)


def _log_progress(step: int, total_steps: int, replay_size: int, updates: int) -> None:
    print(
        f"[FastSAC] progress step={step}/{total_steps} replay_size={replay_size} updates={updates}",
        flush=True,
    )


def _require_positive_interval(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_positive_count(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class CurveLogger:
    def __init__(self, log_dir: Path):
        self.csv_path = log_dir / "training_curve.csv"
        self.jsonl_path = log_dir / "training_curve.jsonl"
        self._fieldnames = [
            "step",
            "total_steps",
            "updates",
            "replay_size",
            "reward_mean",
            "terminated_rate",
            "truncated_rate",
            "actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha",
            "mean_q",
            "target_q_mean",
        ]
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._csv_file.flush()

    def write(self, row: dict[str, float | int | None]) -> None:
        payload = {key: row.get(key) for key in self._fieldnames}
        self._writer.writerow(payload)
        self._csv_file.flush()
        self._jsonl_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        self._csv_file.close()
        self._jsonl_file.close()


def _load_checkpoint(path: str, agent: FastSACAgent, replay_buffer: FastSACReplayBuffer) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    agent.load_checkpoint_state(payload)
    if "replay_buffer" in payload:
        replay_buffer.load_state_dict(payload["replay_buffer"])
    return int(payload.get("step", 0))


def _get_num_actions(env) -> int:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_manager"):
        return int(unwrapped.action_manager.total_action_dim)
    return int(gym.spaces.flatdim(unwrapped.single_action_space))


def _replace_obs_for_env_ids(obs, replacement_obs, env_ids: torch.Tensor):
    if env_ids.numel() == 0:
        return obs
    if isinstance(obs, dict):
        return {key: _replace_obs_for_env_ids(value, replacement_obs[key], env_ids) for key, value in obs.items()}
    if torch.is_tensor(obs):
        corrected = obs.clone()
        corrected[env_ids] = replacement_obs[env_ids]
        return corrected
    return obs


def _manager_based_step_with_terminal_obs(env, action: torch.Tensor):
    unwrapped = env.unwrapped
    if not all(
        hasattr(unwrapped, name)
        for name in (
            "action_manager",
            "recorder_manager",
            "termination_manager",
            "reward_manager",
            "observation_manager",
            "command_manager",
            "event_manager",
            "_reset_idx",
        )
    ):
        next_obs, reward, terminated, truncated, extras = env.step(action)
        return next_obs, next_obs, reward, terminated, truncated, extras

    unwrapped.action_manager.process_action(action.to(unwrapped.device))
    unwrapped.recorder_manager.record_pre_step()
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()

    for _ in range(unwrapped.cfg.decimation):
        unwrapped._sim_step_counter += 1
        unwrapped.action_manager.apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        unwrapped.recorder_manager.record_post_physics_decimation_step()
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.episode_length_buf += 1
    unwrapped.common_step_counter += 1
    unwrapped.reset_buf = unwrapped.termination_manager.compute()
    unwrapped.reset_terminated = unwrapped.termination_manager.terminated
    unwrapped.reset_time_outs = unwrapped.termination_manager.time_outs
    unwrapped.reward_buf = unwrapped.reward_manager.compute(dt=unwrapped.step_dt)

    if len(unwrapped.recorder_manager.active_terms) > 0:
        unwrapped.obs_buf = unwrapped.observation_manager.compute()
        unwrapped.recorder_manager.record_post_step()

    reset_env_ids = unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    time_limit_env_ids = (unwrapped.reset_time_outs & ~unwrapped.reset_terminated).nonzero(as_tuple=False).squeeze(-1)
    terminal_obs = None
    if time_limit_env_ids.numel() > 0:
        terminal_obs = unwrapped.observation_manager.compute(update_history=False)

    if reset_env_ids.numel() > 0:
        unwrapped.recorder_manager.record_pre_reset(reset_env_ids)
        unwrapped._reset_idx(reset_env_ids)
        if unwrapped.sim.has_rtx_sensors() and unwrapped.cfg.num_rerenders_on_reset > 0:
            for _ in range(unwrapped.cfg.num_rerenders_on_reset):
                unwrapped.sim.render()
        unwrapped.recorder_manager.record_post_reset(reset_env_ids)

    unwrapped.command_manager.compute(dt=unwrapped.step_dt)
    if "interval" in unwrapped.event_manager.available_modes:
        unwrapped.event_manager.apply(mode="interval", dt=unwrapped.step_dt)
    unwrapped.obs_buf = unwrapped.observation_manager.compute(update_history=True)

    next_obs = unwrapped.obs_buf
    replay_next_obs = next_obs
    if terminal_obs is not None:
        replay_next_obs = _replace_obs_for_env_ids(next_obs, terminal_obs, time_limit_env_ids)
    return next_obs, replay_next_obs, unwrapped.reward_buf, unwrapped.reset_terminated, unwrapped.reset_time_outs, unwrapped.extras


@hydra_task_config(args_cli.task, "fast_sac_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    _update_agent_cfg(agent_cfg)
    torch.manual_seed(agent_cfg.seed)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = agent_cfg.device
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.seed = agent_cfg.seed

    log_dir = _make_log_dir(agent_cfg)
    progress_interval = _require_positive_interval(
        "progress_interval",
        args_cli.progress_interval or min(max(agent_cfg.checkpoint_interval // 10, 1), 1000),
    )
    curve_interval = _require_positive_interval("curve_interval", args_cli.curve_interval or progress_interval)
    (log_dir / "params").mkdir(exist_ok=True)
    write_json_artifact(log_dir / "params" / "agent.json", agent_cfg.to_dict())
    write_json_artifact(
        log_dir / "params" / "run.json",
        build_run_metadata(
            task=args_cli.task,
            motion_file=args_cli.motion_file,
            resume_checkpoint=agent_cfg.resume_checkpoint,
            seed=agent_cfg.seed,
        ),
    )
    wandb_logger = create_wandb_logger(
        WandbLoggerConfig(
            logger=args_cli.logger,
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            mode=args_cli.wandb_mode,
            group=args_cli.wandb_group,
            tags=args_cli.wandb_tags,
            run_id=args_cli.wandb_run_id,
            resume=args_cli.wandb_resume,
            name=agent_cfg.run_name or log_dir.name,
            directory=log_dir,
            config={
                "task": args_cli.task,
                "motion_file": args_cli.motion_file,
                "num_envs": env_cfg.scene.num_envs,
                "seed": agent_cfg.seed,
                "agent": agent_cfg.to_dict(),
            },
        )
    )
    atexit.register(wandb_logger.finish)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    obs, _ = env.reset()
    fast_sac_obs = split_actor_critic_obs(obs)
    num_envs = int(env.unwrapped.num_envs)
    action_dim = _get_num_actions(env)

    agent = FastSACAgent(
        actor_obs_dim=int(fast_sac_obs.actor.shape[-1]),
        critic_obs_dim=int(fast_sac_obs.critic.shape[-1]),
        action_dim=action_dim,
        cfg=agent_cfg,
    )
    replay_buffer = FastSACReplayBuffer(
        capacity=agent_cfg.replay_size,
        actor_obs_dim=int(fast_sac_obs.actor.shape[-1]),
        critic_obs_dim=int(fast_sac_obs.critic.shape[-1]),
        action_dim=action_dim,
        device="cpu",
    )

    obs_split = fast_sac_obs
    agent.observe(obs_split.actor, obs_split.critic)
    metrics: dict[str, float] = {}
    updates = 0
    start_step = 0
    curve_logger = CurveLogger(log_dir)
    if agent_cfg.resume_checkpoint:
        start_step = _load_checkpoint(agent_cfg.resume_checkpoint, agent, replay_buffer)
        print(
            f"[FastSAC] resumed checkpoint={agent_cfg.resume_checkpoint} start_step={start_step}",
            flush=True,
        )
    if args_cli.train_steps is not None:
        trained_steps_target = _require_positive_count("train_steps", args_cli.train_steps)
        agent_cfg.max_steps = start_step + trained_steps_target
    else:
        trained_steps_target = agent_cfg.max_steps - start_step
    if trained_steps_target <= 0:
        raise ValueError(
            f"FastSAC train target must add positive steps; start_step={start_step} max_steps={agent_cfg.max_steps}"
        )
    print(
        f"[FastSAC] start task={args_cli.task} run_name={agent_cfg.run_name!r} max_steps={agent_cfg.max_steps} "
        f"train_steps={trained_steps_target} start_step={start_step} num_envs={num_envs} "
        f"warmup_steps={agent_cfg.warmup_steps} checkpoint_interval={agent_cfg.checkpoint_interval} "
        f"progress_interval={progress_interval} curve_interval={curve_interval}",
        flush=True,
    )

    try:
        for step in range(start_step + 1, agent_cfg.max_steps + 1):
            agent.observe(obs_split.actor, obs_split.critic)
            if step <= agent_cfg.warmup_steps:
                action = torch.empty((num_envs, action_dim), device=agent.device).uniform_(-1.0, 1.0)
            else:
                action = agent.act(obs_split.actor)

            next_obs, replay_next_obs, reward, terminated, truncated, _ = _manager_based_step_with_terminal_obs(env, action)
            next_split = split_actor_critic_obs(next_obs)
            replay_next_split = split_actor_critic_obs(replay_next_obs)
            replay_buffer.add_batch(
                obs_split.actor.detach().cpu(),
                obs_split.critic.detach().cpu(),
                action.detach().cpu(),
                reward.detach().cpu(),
                terminated.detach().cpu(),
                truncated.detach().cpu(),
                replay_next_split.actor.detach().cpu(),
                replay_next_split.critic.detach().cpu(),
            )

            obs_split = next_split
            if replay_buffer.size >= agent_cfg.batch_size and step > agent_cfg.warmup_steps:
                for _ in range(agent_cfg.updates_per_step):
                    metrics = agent.update(replay_buffer.sample(agent_cfg.batch_size))
                    updates += 1

            if step % progress_interval == 0:
                _log_progress(step, agent_cfg.max_steps, replay_buffer.size, updates)

            if step % curve_interval == 0:
                curve_row = {
                    "step": step,
                    "total_steps": agent_cfg.max_steps,
                    "updates": updates,
                    "replay_size": replay_buffer.size,
                    "reward_mean": float(reward.detach().mean().cpu()),
                    "terminated_rate": float(terminated.detach().float().mean().cpu()),
                    "truncated_rate": float(truncated.detach().float().mean().cpu()),
                    "actor_loss": metrics.get("actor_loss"),
                    "critic_loss": metrics.get("critic_loss"),
                    "alpha_loss": metrics.get("alpha_loss"),
                    "alpha": metrics.get("alpha"),
                    "mean_q": metrics.get("mean_q"),
                    "target_q_mean": metrics.get("target_q_mean"),
                }
                curve_logger.write(curve_row)
                wandb_logger.log(curve_row, step=step)

            if step % max(agent_cfg.checkpoint_interval, 1) == 0:
                _save_checkpoint(log_dir, step, agent, replay_buffer, agent_cfg.save_replay_buffer)
    finally:
        curve_logger.close()

    _save_checkpoint(log_dir, agent_cfg.max_steps, agent, replay_buffer, agent_cfg.save_replay_buffer)
    final_checkpoint = log_dir / f"model_{agent_cfg.max_steps}.pt"
    write_json_artifact(
        log_dir / "params" / "run.json",
        build_run_metadata(
            task=args_cli.task,
            motion_file=args_cli.motion_file,
            resume_checkpoint=agent_cfg.resume_checkpoint,
            seed=agent_cfg.seed,
            checkpoint_path=str(final_checkpoint.resolve()),
        ),
    )
    write_json_artifact(
        log_dir / "metrics.json",
        build_training_metrics_payload(
            steps=agent_cfg.max_steps,
            updates=updates,
            metrics={
                "start_step": start_step,
                "trained_steps": agent_cfg.max_steps - start_step,
                **metrics,
            },
        ),
    )
    summary = {
        "final_step": agent_cfg.max_steps,
        "checkpoint_path": str(final_checkpoint.resolve()),
        "metrics_path": str((log_dir / "metrics.json").resolve()),
    }
    if not args_cli.skip_eval:
        print(f"[FastSAC] evaluating checkpoint={final_checkpoint}", flush=True)
        eval_summary = evaluate_policy(
            env=env,
            agent=agent,
            split_obs=split_actor_critic_obs,
            num_episodes=args_cli.eval_episodes,
            horizon=args_cli.eval_horizon,
            deterministic=True,
            task=args_cli.task,
            seed=agent_cfg.seed,
            motion_file=args_cli.motion_file,
            checkpoint_path=str(final_checkpoint.resolve()),
            resume_checkpoint=agent_cfg.resume_checkpoint,
        )
        write_json_artifact(log_dir / "eval_summary.json", eval_summary)
        summary["eval_summary_path"] = str((log_dir / "eval_summary.json").resolve())
        for key, value in eval_summary.items():
            if isinstance(value, int | float):
                summary[f"eval/{key}"] = value
        print(f"[FastSAC] wrote eval_summary path={log_dir / 'eval_summary.json'}", flush=True)
    wandb_logger.update_summary(summary)
    wandb_logger.finish()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
