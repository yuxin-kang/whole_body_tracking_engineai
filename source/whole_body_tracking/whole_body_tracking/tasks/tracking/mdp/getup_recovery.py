"""Reward shaping used by the bounded T800 get-up recovery experiments."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.math import quat_error_magnitude

from . import rewards as tracking_rewards


def _validate_phase_window(phase_start: float, phase_end: float) -> None:
    if not 0.0 <= phase_start < phase_end <= 1.0:
        raise ValueError(f"phase window must satisfy 0 <= start < end <= 1, got [{phase_start}, {phase_end}]")


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def smoothstep_phase_ramp(phase: torch.Tensor, phase_start: float, phase_end: float) -> torch.Tensor:
    """Return a zero-to-one smoothstep ramp over a normalized motion phase."""
    _validate_phase_window(phase_start, phase_end)
    unit_phase = ((phase.to(dtype=torch.float32) - phase_start) / (phase_end - phase_start)).clamp(0.0, 1.0)
    return unit_phase * unit_phase * (3.0 - 2.0 * unit_phase)


def _motion_phase(command) -> torch.Tensor:
    denominator = max(command.motion.time_step_total - 1, 1)
    return command.time_steps.to(dtype=torch.float32) / float(denominator)


def _pose_confidence(command, height_std: float, orientation_std: float) -> torch.Tensor:
    _validate_positive_finite(height_std, "height_std")
    _validate_positive_finite(orientation_std, "orientation_std")
    height_error = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
    orientation_error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-torch.square(height_error / height_std) - torch.square(orientation_error / orientation_std))


def getup_transition_anchor_orientation_error_exp(
    env,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
) -> torch.Tensor:
    """Track reference anchor orientation with a smooth transition ramp."""
    _validate_positive_finite(std, "std")
    command = env.command_manager.get_term(command_name)
    reward = tracking_rewards.motion_global_anchor_orientation_error_exp(
        env=env, command_name=command_name, std=std
    )
    return reward * smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)


def _pose_gated_body_velocity_reward(
    base_reward,
    env,
    command_name: str,
    std: float,
    body_names: list[str] | None,
    phase_start: float,
    phase_end: float,
    height_std: float,
    orientation_std: float,
    floor: float,
) -> torch.Tensor:
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError(f"gate floor must satisfy 0 < floor <= 1, got {floor}")
    _validate_positive_finite(std, "std")
    _validate_positive_finite(height_std, "height_std")
    _validate_positive_finite(orientation_std, "orientation_std")
    command = env.command_manager.get_term(command_name)
    base = base_reward(env=env, command_name=command_name, std=std, body_names=body_names)
    ramp = smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)
    confidence = _pose_confidence(command, height_std=height_std, orientation_std=orientation_std)
    factor = 1.0 - ramp * (1.0 - (floor + (1.0 - floor) * confidence))
    return base * factor


def getup_pose_gated_body_linear_velocity_error_exp(
    env,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str] | None = None,
    height_std: float = 0.25,
    orientation_std: float = 0.60,
    floor: float = 0.10,
) -> torch.Tensor:
    """Wrap the original body-linear-velocity reward with late pose gating."""
    return _pose_gated_body_velocity_reward(
        tracking_rewards.motion_global_body_linear_velocity_error_exp,
        env,
        command_name,
        std,
        body_names,
        phase_start,
        phase_end,
        height_std,
        orientation_std,
        floor,
    )


def getup_pose_gated_body_angular_velocity_error_exp(
    env,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str] | None = None,
    height_std: float = 0.25,
    orientation_std: float = 0.60,
    floor: float = 0.10,
) -> torch.Tensor:
    """Wrap the original body-angular-velocity reward with late pose gating."""
    return _pose_gated_body_velocity_reward(
        tracking_rewards.motion_global_body_angular_velocity_error_exp,
        env,
        command_name,
        std,
        body_names,
        phase_start,
        phase_end,
        height_std,
        orientation_std,
        floor,
    )


def apply_getup_recovery_experiment(env_cfg, variant: str):
    """Apply one bounded get-up recovery variant to an already-built config."""
    if variant not in {"ori", "ori_velgate"}:
        raise ValueError(f"unknown get-up recovery variant: {variant!r}")

    env_cfg.rewards.getup_transition_anchor_ori = RewTerm(
        func=getup_transition_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.4,
            "phase_start": 0.40,
            "phase_end": 0.45,
        },
    )

    if variant == "ori_velgate":
        for reward_name, reward_func in (
            ("motion_body_lin_vel", getup_pose_gated_body_linear_velocity_error_exp),
            ("motion_body_ang_vel", getup_pose_gated_body_angular_velocity_error_exp),
        ):
            reward = getattr(env_cfg.rewards, reward_name)
            reward.func = reward_func
            reward.params.update(
                {
                    "phase_start": 0.70,
                    "phase_end": 0.80,
                    "height_std": 0.25,
                    "orientation_std": 0.60,
                    "floor": 0.10,
                }
            )
    return env_cfg
