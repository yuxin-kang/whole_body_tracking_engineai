from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.rl.fast_sac.recipe_contract import get_fastsac_recipe_contract
from whole_body_tracking.tasks.tracking.config.g1.paper_contract import get_paper_equivalence_contract


def build_run_metadata(
    *,
    task: str,
    motion_file: str | None,
    resume_checkpoint: str | None,
    seed: int,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    contract = get_paper_equivalence_contract()
    recipe_contract = get_fastsac_recipe_contract()
    return {
        "algorithm": "FastSAC",
        "task": task,
        "seed": int(seed),
        "motion_file": motion_file,
        "resume_checkpoint": resume_checkpoint,
        "checkpoint_path": checkpoint_path,
        "paper_equivalence_contract": contract,
        "fast_sac_recipe_contract": recipe_contract,
    }


def build_training_metrics_payload(*, steps: int, updates: int, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "algorithm": "FastSAC",
        "steps": int(steps),
        "updates": int(updates),
        **metrics,
    }


def _zero_episode_tensor(count: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.float32, device=device)


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


def _snapshot_command_metrics(env, command_name: str) -> dict[str, torch.Tensor]:
    command = env.unwrapped.command_manager.get_term(command_name)
    if hasattr(command, "_update_metrics"):
        command._update_metrics()
    joint_position_abs_error_max = torch.amax(torch.abs(command.joint_pos - command.robot_joint_pos), dim=1).clone()
    tracked_body_orientation_error = quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w)
    tracked_body_orientation_error_max = torch.amax(tracked_body_orientation_error, dim=1).clone()
    tracked_body_orientation_error_mean = torch.mean(tracked_body_orientation_error, dim=1).clone()
    return {
        "joint_position_abs_error_max": joint_position_abs_error_max,
        "tracked_body_orientation_error_max": tracked_body_orientation_error_max,
        "tracked_body_orientation_error_mean": tracked_body_orientation_error_mean,
    }


def step_with_terminal_metrics(env, action: torch.Tensor, command_name: str):
    unwrapped = env.unwrapped
    required = (
        "action_manager",
        "recorder_manager",
        "termination_manager",
        "reward_manager",
        "observation_manager",
        "command_manager",
        "event_manager",
        "_reset_idx",
    )
    if not all(hasattr(unwrapped, name) for name in required):
        missing = [name for name in required if not hasattr(unwrapped, name)]
        raise RuntimeError(
            "FastSAC Isaac-only eval requires manager-based terminal metric capture; "
            f"missing runtime features: {missing}"
        )

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
    terminal_metrics = _snapshot_command_metrics(env, command_name)

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
    if terminal_obs is not None:
        next_obs = _replace_obs_for_env_ids(next_obs, terminal_obs, time_limit_env_ids)
    return (
        next_obs,
        unwrapped.reward_buf,
        unwrapped.reset_terminated,
        unwrapped.reset_time_outs,
        unwrapped.extras,
        terminal_metrics,
    )


def evaluate_policy(
    *,
    env,
    agent,
    split_obs,
    num_episodes: int,
    horizon: int | None,
    deterministic: bool,
    task: str,
    seed: int,
    motion_file: str | None,
    checkpoint_path: str | None,
    resume_checkpoint: str | None,
) -> dict[str, Any]:
    if int(num_episodes) <= 0:
        raise ValueError(f"num_episodes must be positive for Isaac-only evaluation, got {num_episodes}")

    contract = get_paper_equivalence_contract()
    thresholds = contract["evaluation"]["success_rate"]
    joint_threshold = float(thresholds["joint_position_error_threshold_rad"])
    orientation_threshold = float(thresholds["tracked_body_orientation_error_threshold_rad"])

    command_name = "motion"
    num_envs = int(env.unwrapped.num_envs)
    episodes_remaining = int(num_episodes)
    stats_device = agent.device
    collected_returns: list[float] = []
    collected_lengths: list[int] = []
    collected_orientation: list[float] = []
    collected_smoothness: list[float] = []
    success_count = 0
    failure_by_joint = 0
    failure_by_orientation = 0
    failure_by_fall = 0
    truncated_count = 0
    terminated_count = 0
    horizon_completed_count = 0

    while episodes_remaining > 0:
        batch_size = min(num_envs, episodes_remaining)
        obs, _ = env.reset()
        obs_split = split_obs(obs)
        active = torch.zeros(num_envs, dtype=torch.bool, device=stats_device)
        active[:batch_size] = True
        episode_return = _zero_episode_tensor(num_envs, stats_device)
        episode_length = torch.zeros(num_envs, dtype=torch.long, device=stats_device)
        orientation_sum = _zero_episode_tensor(num_envs, stats_device)
        smoothness_sum = _zero_episode_tensor(num_envs, stats_device)
        smoothness_steps = torch.zeros(num_envs, dtype=torch.long, device=stats_device)
        failed = torch.zeros(num_envs, dtype=torch.bool, device=stats_device)
        failed_joint = torch.zeros(num_envs, dtype=torch.bool, device=stats_device)
        failed_orientation = torch.zeros(num_envs, dtype=torch.bool, device=stats_device)
        failed_fall = torch.zeros(num_envs, dtype=torch.bool, device=stats_device)
        prev_action = None
        step_limit = horizon or int(getattr(env.unwrapped, "max_episode_length", 0) or 0) or 1

        for _ in range(step_limit):
            action = agent.act(obs_split.actor, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _, terminal_metrics = step_with_terminal_metrics(
                env, action, command_name
            )
            joint_error = terminal_metrics["joint_position_abs_error_max"]
            orientation_error = terminal_metrics["tracked_body_orientation_error_max"]
            orientation_error_mean = terminal_metrics["tracked_body_orientation_error_mean"]
            action_delta = torch.zeros(num_envs, device=stats_device)
            if prev_action is not None:
                action_delta = torch.linalg.norm(action - prev_action, dim=1)
                smoothness_sum[active] += action_delta[active]
                smoothness_steps[active] += 1
            orientation_sum[active] += orientation_error_mean[active]
            episode_return[active] += reward[active]
            episode_length[active] += 1

            failed_joint |= joint_error > joint_threshold
            failed_orientation |= orientation_error > orientation_threshold
            failed_fall |= terminated
            failed |= failed_joint | failed_orientation | failed_fall

            done = (terminated | truncated) & active
            finished_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            for env_id in finished_ids:
                length = int(episode_length[env_id].item())
                collected_returns.append(float(episode_return[env_id].item()))
                collected_lengths.append(length)
                collected_orientation.append(float(orientation_sum[env_id].item() / max(length, 1)))
                collected_smoothness.append(
                    float(smoothness_sum[env_id].item() / max(int(smoothness_steps[env_id].item()), 1))
                )
                success = not bool(failed[env_id].item())
                success_count += int(success)
                failure_by_joint += int(failed_joint[env_id].item())
                failure_by_orientation += int(failed_orientation[env_id].item())
                failure_by_fall += int(failed_fall[env_id].item())
                truncated_count += int(truncated[env_id].item())
                terminated_count += int(terminated[env_id].item())
            active[done] = False
            prev_action = action
            obs_split = split_obs(next_obs)
            if not torch.any(active):
                break

        if torch.any(active):
            unfinished_ids = active.nonzero(as_tuple=False).squeeze(-1).tolist()
            for env_id in unfinished_ids:
                length = int(episode_length[env_id].item())
                collected_returns.append(float(episode_return[env_id].item()))
                collected_lengths.append(length)
                collected_orientation.append(float(orientation_sum[env_id].item() / max(length, 1)))
                collected_smoothness.append(
                    float(smoothness_sum[env_id].item() / max(int(smoothness_steps[env_id].item()), 1))
                )
                success = not bool(failed[env_id].item())
                success_count += int(success)
                failure_by_joint += int(failed_joint[env_id].item())
                failure_by_orientation += int(failed_orientation[env_id].item())
                failure_by_fall += int(failed_fall[env_id].item())
                horizon_completed_count += 1

        episodes_remaining -= batch_size

    total = len(collected_returns)
    success_rate = float(success_count / max(total, 1))
    orientation_error_mean = float(sum(collected_orientation) / max(total, 1))
    smoothness_mean = float(sum(collected_smoothness) / max(total, 1))
    episode_return_mean = float(sum(collected_returns) / max(total, 1))
    episode_length_mean = float(sum(collected_lengths) / max(total, 1))
    summary = {
        "algorithm": "FastSAC",
        "task": task,
        "seed": int(seed),
        "motion_file": motion_file,
        "checkpoint_path": checkpoint_path,
        "resume_checkpoint": resume_checkpoint,
        "paper_equivalence_contract": contract,
        "success_rate": success_rate,
        "orientation_error_mean": orientation_error_mean,
        "smoothness_mean": smoothness_mean,
        "episode_return_mean": episode_return_mean,
        "episode_length_mean": episode_length_mean,
        "contract": contract,
        "episodes": {
            "requested": int(num_episodes),
            "completed": total,
            "horizon": int(horizon) if horizon is not None else None,
            "deterministic": bool(deterministic),
        },
        "metrics": {
            "success_rate": success_rate,
            "orientation_error_mean": orientation_error_mean,
            "smoothness_mean": smoothness_mean,
            "episode_return_mean": episode_return_mean,
            "episode_length_mean": episode_length_mean,
        },
        "termination_summary": {
            "terminated_count": int(terminated_count),
            "truncated_count": int(truncated_count),
            "horizon_completed_count": int(horizon_completed_count),
            "failure_by_joint_error_count": int(failure_by_joint),
            "failure_by_orientation_error_count": int(failure_by_orientation),
            "failure_by_fall_count": int(failure_by_fall),
        },
    }
    return summary


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
