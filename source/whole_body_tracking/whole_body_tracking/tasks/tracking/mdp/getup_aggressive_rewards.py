"""Aggressive A/B/C/D T800 get-up reward candidates."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.math import quat_apply

from . import rewards as tracking_rewards
from .getup_recovery import (
    _pose_confidence,
    getup_pose_gated_body_angular_velocity_error_exp,
    getup_pose_gated_body_linear_velocity_error_exp,
    smoothstep_phase_ramp,
)


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def _motion_phase(command) -> torch.Tensor:
    denominator = max(command.motion.time_step_total - 1, 1)
    return command.time_steps.to(dtype=torch.float32) / float(denominator)


def _validate_gate(floor: float, height_std: float, orientation_std: float) -> None:
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError(f"gate floor must satisfy 0 < floor <= 1, got {floor}")
    _validate_positive_finite(height_std, "height_std")
    _validate_positive_finite(orientation_std, "orientation_std")


def _pose_gate(command, phase_start: float, phase_end: float, floor: float, height_std: float, orientation_std: float):
    _validate_gate(floor, height_std, orientation_std)
    ramp = smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)
    confidence = _pose_confidence(command, height_std=height_std, orientation_std=orientation_std)
    return 1.0 - ramp * (1.0 - (floor + (1.0 - floor) * confidence))


def aggressive_anchor_orientation_error_exp(
    env, command_name: str, std: float, phase_start: float, phase_end: float
) -> torch.Tensor:
    """Reference anchor orientation reward with a smooth transition ramp."""
    _validate_positive_finite(std, "std")
    command = env.command_manager.get_term(command_name)
    reward = tracking_rewards.motion_global_anchor_orientation_error_exp(
        env=env, command_name=command_name, std=std
    )
    return reward * smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)


def aggressive_pose_gated_anchor_angular_velocity_error_exp(
    env,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    gate_phase_start: float,
    gate_phase_end: float,
    height_std: float = 0.25,
    orientation_std: float = 0.60,
    floor: float = 0.10,
) -> torch.Tensor:
    """Track world anchor angular velocity while suppressing fallen-pose scores."""
    _validate_positive_finite(std, "std")
    command = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    reward = torch.exp(-error / std**2)
    return reward * smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end) * _pose_gate(
        command,
        phase_start=gate_phase_start,
        phase_end=gate_phase_end,
        floor=floor,
        height_std=height_std,
        orientation_std=orientation_std,
    )


def aggressive_anchor_position_error_exp(
    env, command_name: str, std: float, phase_start: float, phase_end: float
) -> torch.Tensor:
    """Reference anchor position reward with a smooth transition ramp."""
    _validate_positive_finite(std, "std")
    command = env.command_manager.get_term(command_name)
    reward = tracking_rewards.motion_global_anchor_position_error_exp(
        env=env, command_name=command_name, std=std
    )
    return reward * smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)


def aggressive_anchor_height_error_exp(
    env, command_name: str, std: float, phase_start: float, phase_end: float
) -> torch.Tensor:
    """Reference base-height reward with a smooth transition ramp."""
    _validate_positive_finite(std, "std")
    command = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    reward = torch.exp(-error / std**2)
    return reward * smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)


def aggressive_standing_quality_reward(
    env, command_name: str, phase_start: float, phase_end: float
) -> torch.Tensor:
    """Late standing quality product using absolute uprightness and actual base speeds."""
    command = env.command_manager.get_term(command_name)
    up_body = torch.zeros_like(command.robot_anchor_quat_w[:, 1:])
    up_body[:, 2] = 1.0
    up_world = quat_apply(command.robot_anchor_quat_w, up_body)
    tilt = torch.acos(torch.clamp(up_world[:, 2], -1.0, 1.0))
    height_error = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
    angular_speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1)
    horizontal_speed = torch.linalg.vector_norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    exponent = (
        torch.square(height_error / 0.18)
        + torch.square(tilt / 0.25)
        + torch.square(angular_speed / 0.50)
        + torch.square(horizontal_speed / 0.25)
    )
    return torch.exp(-exponent) * smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)


def _gated_body_velocity_params(reward) -> None:
    reward.params.update(
        {
            "phase_start": 0.65,
            "phase_end": 0.80,
            "height_std": 0.25,
            "orientation_std": 0.60,
            "floor": 0.10,
        }
    )


def apply_aggressive_rewards(env_cfg, variant: str):
    """Apply aggressive reward candidate ``a``, ``b``, ``c`` or ``d``.

    The caller must first reconstruct the original j9519 reward baseline. This
    function only mutates reward terms and leaves command/termination settings
    untouched.
    """
    if variant not in {"a", "b", "c", "d"}:
        raise ValueError(f"unknown aggressive reward variant: {variant!r}")

    rewards = env_cfg.rewards
    rewards.getup_aggressive_anchor_ori = RewTerm(
        func=aggressive_anchor_orientation_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 0.60, "phase_start": 0.35, "phase_end": 0.45},
    )
    rewards.getup_aggressive_anchor_ang_vel = RewTerm(
        func=aggressive_pose_gated_anchor_angular_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 1.0,
            "phase_start": 0.35,
            "phase_end": 0.45,
            "gate_phase_start": 0.65,
            "gate_phase_end": 0.80,
            "height_std": 0.25,
            "orientation_std": 0.60,
            "floor": 0.10,
        },
    )
    rewards.motion_body_lin_vel.func = getup_pose_gated_body_linear_velocity_error_exp
    rewards.motion_body_ang_vel.func = getup_pose_gated_body_angular_velocity_error_exp
    _gated_body_velocity_params(rewards.motion_body_lin_vel)
    _gated_body_velocity_params(rewards.motion_body_ang_vel)

    if variant in {"b", "c", "d"}:
        rewards.getup_aggressive_anchor_pos = RewTerm(
            func=aggressive_anchor_position_error_exp,
            weight=1.5,
            params={"command_name": "motion", "std": 0.45, "phase_start": 0.45, "phase_end": 0.55},
        )
        rewards.getup_aggressive_anchor_height = RewTerm(
            func=aggressive_anchor_height_error_exp,
            weight=1.75,
            params={"command_name": "motion", "std": 0.25, "phase_start": 0.45, "phase_end": 0.55},
        )

    if variant == "d":
        rewards.getup_aggressive_standing_quality = RewTerm(
            func=aggressive_standing_quality_reward,
            weight=4.0,
            params={"command_name": "motion", "phase_start": 0.825, "phase_end": 0.86},
        )
    return env_cfg
