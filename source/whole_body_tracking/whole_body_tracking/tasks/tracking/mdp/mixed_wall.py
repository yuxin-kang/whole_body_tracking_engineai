"""Wall/no-wall cohorts for kick tracking with a shared actor observation ABI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import target_position_b
from .rewards import target_contact_force_reward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def wall_present_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep exactly half of an even-sized batch in each cohort across resets.

    Fixed environment IDs avoid a survival bias from re-sampling wall presence
    whenever just one cohort fails early. A one-environment Play has a wall;
    use two environments to compare both cases side by side.
    """
    if not hasattr(env, "_target_wall_present"):
        env._target_wall_present = torch.arange(env.num_envs, device=env.device) % 2 == 0
    return env._target_wall_present


def reset_mixed_wall_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    absent_height: float = -10.0,
    strike_depth_range: tuple[float, float] = (0.025, 0.025),
) -> None:
    """Sample inward depth from the reference foot envelope for yaw-zero walls.

    Default wall poses are calibrated to 0.025 m reference overlap. This is
    a geometric target depth, not physical penetration of the simulated foot.
    """
    if absent_height > -5.0:
        raise ValueError("Inactive walls must remain well below the floor")
    low, high = strike_depth_range
    if not 0.0 <= low <= high:
        raise ValueError("strike_depth_range must be nonnegative and ordered")
    target = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    present = wall_present_mask(env)[env_ids]
    state = target.data.default_root_state[env_ids].clone()
    depth = torch.empty(len(env_ids), device=env.device).uniform_(low, high)
    state[present, 0] += 0.025 - depth[present]
    state[~present, 2] = absent_height
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 7:] = 0.0
    target.write_root_state_to_sim(state, env_ids=env_ids)


def mixed_wall_target_position_b(
    env: ManagerBasedRLEnv,
    asset_name: str = "target",
    command_name: str = "motion",
) -> torch.Tensor:
    """Zero the privileged target vector when no reachable wall exists."""
    position = target_position_b(env, asset_name=asset_name, command_name=command_name)
    return torch.where(wall_present_mask(env)[:, None], position, torch.zeros_like(position))


def mixed_wall_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    phase_start: float,
    phase_end: float,
    force_scale: float = 25.0,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Only reward filtered wall contact in environments containing a wall."""
    reward = target_contact_force_reward(
        env, command_name, sensor_cfg, phase_start, phase_end, force_scale, force_threshold,
    )
    return torch.where(wall_present_mask(env), reward, torch.zeros_like(reward))
