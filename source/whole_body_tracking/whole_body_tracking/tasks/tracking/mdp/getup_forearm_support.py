"""Short, contact-gated forearm guidance for the original T800 get-up clip."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply

from .getup_recovery import smoothstep_phase_ramp


def forearm_vertical_score(elbow_pos_w: torch.Tensor, wrist_pos_w: torch.Tensor, std: float) -> torch.Tensor:
    """Score wrist-to-elbow alignment with world +Z, independently for each arm.

    Use link positions, not a link's arbitrarily oriented local axis. Reversed
    vertical arms (elbow below wrist) must not earn the upright support score.
    """
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("forearm angle std must be finite and positive")
    direction = elbow_pos_w - wrist_pos_w
    length = torch.linalg.vector_norm(direction, dim=-1)
    cosine = (direction[..., 2] / length.clamp_min(1.0e-6)).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    return torch.exp(-torch.square(angle / std)) * (length > 1.0e-6)


def support_phase_gate(phase: torch.Tensor, phase_window: tuple[float, float, float, float]) -> torch.Tensor:
    """Smoothly enter and leave support; re-contact after the window earns zero."""
    if len(phase_window) != 4 or not 0 <= phase_window[0] < phase_window[1] <= phase_window[2] < phase_window[3] <= 1:
        raise ValueError("phase_window must satisfy 0 <= start < full <= fade < end <= 1")
    start, full, fade, end = phase_window
    return smoothstep_phase_ramp(phase, start, full) * (1.0 - smoothstep_phase_ramp(phase, fade, end))


def ground_support_gate(force_z: torch.Tensor, force_min: float, force_full: float) -> torch.Tensor:
    """Bound the contribution of normal force so impacts cannot inflate reward."""
    if not (math.isfinite(force_min) and math.isfinite(force_full) and 0 <= force_min < force_full):
        raise ValueError("support forces must satisfy 0 <= force_min < force_full and be finite")
    unit = ((force_z - force_min) / (force_full - force_min)).clamp(0.0, 1.0)
    return unit.square() * (3.0 - 2.0 * unit)


def _ground_force_z(env, sensor_names) -> torch.Tensor:
    forces = []
    for name in sensor_names:
        matrix = env.scene.sensors[name].data.force_matrix_w
        if matrix is None or matrix.shape[1] != 1 or matrix.shape[2] == 0:
            raise RuntimeError(f"{name} must resolve one body and a nonempty ground contact filter")
        forces.append(matrix[..., 2].sum(dim=(1, 2)))
    return torch.stack(forces, dim=-1)


def _wrist_centers(robot, wrist_cfg, wrist_offsets_b):
    positions = robot.data.body_pos_w[:, wrist_cfg.body_ids]
    if positions.shape[1] != 2:
        raise ValueError("wrist_cfg must resolve exactly two bodies, left then right")
    offsets = positions.new_tensor(wrist_offsets_b).expand_as(positions)
    quaternions = robot.data.body_quat_w[:, wrist_cfg.body_ids]
    return positions + quat_apply(quaternions.reshape(-1, 4), offsets.reshape(-1, 3)).reshape_as(positions)


def getup_forearm_vertical_reward(
    env,
    command_name: str,
    elbow_cfg,
    wrist_cfg,
    wrist_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
    ground_sensor_names: tuple[str, str, str, str],
    std: float,
    phase_window: tuple[float, float, float, float],
    force_min: float,
    force_full: float,
) -> torch.Tensor:
    """Average left/right uprightness only while that wrist supports on the floor.

    Sensors are ordered (left forearm, left wrist, right forearm, right wrist),
    with one filtered body per sensor. Only ground-filtered forces count; arm
    contact with the torso or another limb does not qualify as floor support.
    Forearm-body contact alone must not bypass the wrist contact/slip condition.
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene[elbow_cfg.name]
    elbows = robot.data.body_pos_w[:, elbow_cfg.body_ids]
    wrists = _wrist_centers(robot, wrist_cfg, wrist_offsets_b)
    if elbows.shape[1] != 2 or wrists.shape[1] != 2 or len(ground_sensor_names) != 4:
        raise ValueError("forearm support requires two ordered elbows/wrists and four ground sensors")
    force_z = _ground_force_z(env, ground_sensor_names).reshape(-1, 2, 2)[..., 1]
    contact = ground_support_gate(force_z, force_min, force_full)
    phase = command.time_steps.float() / float(max(command.motion.time_step_total - 1, 1))
    phase_gate = support_phase_gate(phase, phase_window)
    score = forearm_vertical_score(elbows, wrists, std)
    # Divide by both arms, not by contact count: losing one support cannot
    # increase the average simply by dropping the weaker side.
    return (score * contact).mean(dim=-1) * phase_gate


def getup_wrist_support_slip_penalty(
    env,
    command_name: str,
    wrist_cfg,
    wrist_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
    wrist_sensor_names: tuple[str, str],
    phase_window: tuple[float, float, float, float],
    force_min: float,
    force_full: float,
    speed_std: float,
    sphere_radius: float,
) -> torch.Tensor:
    """Bounded tangential slip at the wrist sphere's floor contact point.

    body_lin_vel_w in Isaac Lab 2.1 is COM velocity, so include omega cross r
    from the COM to the sphere's lowest point. Only actual wrist-floor support
    enables this penalty; airborne arm repositioning is free of slip penalty.
    """
    if not math.isfinite(speed_std) or speed_std <= 0.0 or not math.isfinite(sphere_radius) or sphere_radius <= 0.0:
        raise ValueError("speed_std and sphere_radius must be finite and positive")
    if len(wrist_sensor_names) != 2:
        raise ValueError("two wrist ground sensors required")
    robot = env.scene[wrist_cfg.name]
    point = _wrist_centers(robot, wrist_cfg, wrist_offsets_b)
    point = point - point.new_tensor((0.0, 0.0, sphere_radius))
    ids = wrist_cfg.body_ids
    lever = point - robot.data.body_com_pos_w[:, ids]
    velocity = robot.data.body_lin_vel_w[:, ids] + torch.cross(robot.data.body_ang_vel_w[:, ids], lever, dim=-1)
    speed_squared = velocity[..., :2].square().sum(dim=-1)
    penalty = 1.0 - torch.exp(-speed_squared / speed_std**2)
    contact = ground_support_gate(_ground_force_z(env, wrist_sensor_names), force_min, force_full)
    command = env.command_manager.get_term(command_name)
    phase = command.time_steps.float() / float(max(command.motion.time_step_total - 1, 1))
    return (penalty * contact).mean(dim=-1) * support_phase_gate(phase, phase_window)
