from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_joint_indexes(command: MotionCommand, joint_names: list[str] | None) -> list[int]:
    if joint_names is None:
        return list(range(command.joint_pos.shape[1]))
    if command.cfg.motion_joint_names is None:
        raise ValueError("joint_names require MotionCommandCfg.motion_joint_names to be set")
    return [command.cfg.motion_joint_names.index(name) for name in joint_names]


def _get_phase_window_mask(command: MotionCommand, phase_start: float, phase_end: float) -> torch.Tensor:
    if phase_end <= phase_start:
        raise ValueError(f"Invalid phase window: [{phase_start}, {phase_end}]")
    denom = max(command.motion.time_step_total - 1, 1)
    phase = command.time_steps.to(dtype=torch.float32) / float(denom)
    return ((phase >= phase_start) & (phase <= phase_end)).to(dtype=torch.float32)


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def _select_sensor_body_forces(forces: torch.Tensor, body_ids) -> torch.Tensor:
    if body_ids == slice(None):
        return forces
    index = torch.as_tensor(body_ids, dtype=torch.long, device=forces.device)
    return torch.index_select(forces, dim=-2, index=index)


def self_collision_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    data = contact_sensor.data
    if hasattr(data, "net_forces_w_history") and data.net_forces_w_history is not None:
        forces = _select_sensor_body_forces(data.net_forces_w_history, sensor_cfg.body_ids)
        force_mag = torch.norm(forces, dim=-1)
        hit = (force_mag > force_threshold).any(dim=-1)
        return hit.sum(dim=-1).float()
    if hasattr(data, "force_history") and data.force_history is not None:
        forces = _select_sensor_body_forces(data.force_history, sensor_cfg.body_ids)
        force_mag = torch.norm(forces, dim=-1)
        hit = (force_mag > force_threshold).any(dim=-1)
        return hit.sum(dim=-1).float()
    if hasattr(data, "net_forces_w"):
        forces = _select_sensor_body_forces(data.net_forces_w, sensor_cfg.body_ids)
        force_mag = torch.norm(forces, dim=-1)
        return (force_mag > force_threshold).sum(dim=-1).float()
    return torch.zeros(env.num_envs, device=env.device)


def penalty_relative_shoulder_high(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    shoulder_indexes = getattr(command, "shoulders_indexes", [])
    if not shoulder_indexes:
        return torch.zeros(command.num_envs, device=command.device)
    error = command.body_pos_relative_w[:, shoulder_indexes, 2] - command.robot_body_pos_w[:, shoulder_indexes, 2]
    return torch.sum(torch.square(error), dim=-1)


def penalty_relative_root_orientation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    root_index = getattr(command, "root_index", getattr(command, "motion_root_body_index", 0))
    return quat_error_magnitude(
        command.body_quat_relative_w[:, root_index],
        command.robot_body_quat_w[:, root_index],
    ) ** 2


def penalty_xy_rate_before_stand(
    env: ManagerBasedRLEnv,
    command_name: str,
    stand_threshold: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    shoulder_indexes = getattr(command, "shoulders_indexes", [])
    if not shoulder_indexes or not hasattr(command, "prev_anchor_pos"):
        return torch.zeros(command.num_envs, device=command.device)
    drift_error = torch.norm(command.prev_anchor_pos[:, :2] - command.robot_anchor_pos_w[:, :2], dim=1)
    height_diff = torch.norm(
        command.body_pos_relative_w[:, shoulder_indexes, 2] - command.robot_body_pos_w[:, shoulder_indexes, 2],
        dim=-1,
    )
    return torch.where(height_diff > stand_threshold, drift_error, torch.zeros_like(drift_error))


def penalty_action_rate_before_stand(
    env: ManagerBasedRLEnv,
    command_name: str,
    stand_threshold: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    shoulder_indexes = getattr(command, "shoulders_indexes", [])
    if not shoulder_indexes:
        return torch.zeros(command.num_envs, device=command.device)
    height_diff = torch.norm(
        command.body_pos_relative_w[:, shoulder_indexes, 2] - command.robot_body_pos_w[:, shoulder_indexes, 2],
        dim=-1,
    )
    action_delta = env.action_manager.action - env.action_manager.prev_action
    action_rate = torch.sum(torch.square(action_delta), dim=1)
    return torch.where(height_diff > stand_threshold, action_rate, torch.zeros_like(action_rate))


def penalty_electrical_power_cost(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = asset_cfg.joint_ids

    torque = None
    for attr_name in ("applied_torque", "computed_torque", "joint_effort"):
        if hasattr(asset.data, attr_name):
            torque = getattr(asset.data, attr_name)
            break
    if torque is None:
        return torch.zeros(asset.data.joint_vel.shape[0], device=asset.device)

    tau = torque[:, joint_ids]
    qd = asset.data.joint_vel[:, joint_ids]
    mech = -tau * qd - 150.0
    mech_pos = torch.clamp(mech, min=0.0)
    return torch.sum((mech_pos / 500.0) ** 2, dim=1)


def reward_center_of_mass(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_com: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_indexes = getattr(command, "feet_indexes", [])
    if len(feet_indexes) < 2:
        return torch.zeros(command.num_envs, device=command.device)

    asset: Articulation = env.scene[asset_cfg.name]
    # Paper-aligned: whole-body mass-weighted center of mass (was pelvis root_com_pos_w).
    masses = asset.data.default_mass.to(asset.data.body_com_pos_w.device)  # (N, num_bodies)
    body_com_xy = asset.data.body_com_pos_w[:, :, :2]  # (N, num_bodies, 2)
    com_xy = (masses.unsqueeze(-1) * body_com_xy).sum(dim=1) / masses.sum(dim=1, keepdim=True)
    left_foot_z = command.robot_body_pos_w[:, feet_indexes[0], 2]
    right_foot_z = command.robot_body_pos_w[:, feet_indexes[1], 2]
    single_support = torch.abs(left_foot_z - right_foot_z) > 0.05

    left_foot_xy = command.robot_body_pos_w[:, feet_indexes[0], :2]
    right_foot_xy = command.robot_body_pos_w[:, feet_indexes[1], :2]
    lower_foot_xy = torch.where((left_foot_z > right_foot_z).unsqueeze(-1), right_foot_xy, left_foot_xy)
    # Paper-aligned CoM kernel: exp(-||.|| / sigma^2) on the linear norm.
    error = torch.norm(com_xy - lower_foot_xy, dim=-1)
    return torch.exp(-error / (sigma_com**2)) * single_support


def close_feet_penalty(env: ManagerBasedRLEnv, command_name: str, min_distance: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_indexes = getattr(command, "feet_indexes", [])
    if len(feet_indexes) < 2:
        return torch.zeros(command.num_envs, device=command.device)

    left_xy = command.robot_body_pos_w[:, feet_indexes[0], :2]
    right_xy = command.robot_body_pos_w[:, feet_indexes[1], :2]
    distance = torch.norm(left_xy - right_xy, dim=-1)
    # Paper-aligned: linear hinge max(0, min_distance - d) (was squared).
    return torch.clamp(min_distance - distance, min=0.0)


def feet_slip_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_indexes = getattr(command, "feet_indexes", [])
    if len(feet_indexes) == 0:
        return torch.zeros(command.num_envs, device=command.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    contact_mask = contact_force_z > force_threshold
    # Paper-aligned feet_slip: sum_f sqrt(||v_f^xy||), i.e. (vx^2+vy^2)^0.25 (was squared speed).
    foot_xy_speed = torch.norm(command.robot_body_lin_vel_w[:, feet_indexes, :2], dim=-1)
    return torch.sum(torch.sqrt(foot_xy_speed) * contact_mask.to(foot_xy_speed.dtype), dim=-1)


def action_rate_l2_by_joint_names(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_names: list[str],
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    action_delta = env.action_manager.action[:, joint_indexes] - env.action_manager.prev_action[:, joint_indexes]
    return torch.sum(torch.square(action_delta), dim=1)


def motion_global_anchor_xy_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]), dim=-1)
    return torch.exp(-error / std**2)


def phase_motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.square(command.joint_pos[:, joint_indexes] - command.robot_joint_pos[:, joint_indexes])
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward * _get_phase_window_mask(command, phase_start=phase_start, phase_end=phase_end)


def phase_motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_relative_body_position_error_exp(
        env=env,
        command_name=command_name,
        std=std,
        body_names=body_names,
    )
    return reward * _get_phase_window_mask(command, phase_start=phase_start, phase_end=phase_end)


def phase_motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_global_body_linear_velocity_error_exp(
        env=env,
        command_name=command_name,
        std=std,
        body_names=body_names,
    )
    return reward * _get_phase_window_mask(command, phase_start=phase_start, phase_end=phase_end)


def phase_motion_global_anchor_xy_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_global_anchor_xy_velocity_error_exp(
        env=env,
        command_name=command_name,
        std=std,
    )
    return reward * _get_phase_window_mask(command, phase_start=phase_start, phase_end=phase_end)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def support_foot_com_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    std: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    root_com_xy = asset.data.root_com_pos_w[:, :2]

    contact_force_norm = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    contact_mask = (contact_force_norm > force_threshold).to(foot_pos_w.dtype)

    support_pos_xy = torch.sum(foot_pos_w * contact_mask.unsqueeze(-1), dim=1)
    support_count = contact_mask.sum(dim=1, keepdim=True)
    fallback_midpoint_xy = foot_pos_w.mean(dim=1)
    support_pos_xy = torch.where(
        support_count > 0.0,
        support_pos_xy / support_count.clamp_min(1.0),
        fallback_midpoint_xy,
    )

    error = torch.sum(torch.square(root_com_xy - support_pos_xy), dim=-1)
    return torch.exp(-error / std**2)
