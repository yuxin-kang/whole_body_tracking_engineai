"""Bounded command, curriculum, and termination helpers for the T800 candidates.

This module is intentionally separate from :mod:`commands`.  The base motion
command remains the source of truth for loading and applying complete
reference states (including root and joint velocities).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


AggressiveVariant = Literal["a", "b", "c", "d"]
_INITIAL_MIXTURE = (0.40, 0.40, 0.20)
_MID_MIXTURE = (0.80, 0.15, 0.05)
_START_MIXTURE = (1.00, 0.00, 0.00)
_TRANSITION_WINDOW = (0.38, 0.72)
_STANDING_WINDOW = (0.72, 0.86)
_PPO_STEPS_PER_UPDATE = 48.0
_FAILURE_ENABLE_UPDATE = 5_000.0
_FAILURE_TIGHTEN_UPDATE = 10_000.0
_FAILURE_GRACE_SECONDS = 0.30
_FAILURE_SUSTAINED_SECONDS = 0.30


def _checked_variant(variant: str) -> AggressiveVariant:
    normalized = str(variant).lower()
    if normalized not in {"a", "b", "c", "d"}:
        raise ValueError(f"aggressive variant must be one of a/b/c/d, got {variant!r}")
    return normalized  # type: ignore[return-value]


def _finite_positive_triplet(values: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(values) != 3 or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
        raise ValueError(f"mixture probabilities must be finite and non-negative, got {values!r}")
    total = float(sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"mixture probabilities must have a positive finite sum, got {values!r}")
    return tuple(float(value) / total for value in values)  # type: ignore[return-value]


def mixture_probabilities(training_update: float) -> tuple[float, float, float]:
    """Return ``(frame0, transition, near_standing)`` for a PPO update number."""

    update = float(training_update)
    if not math.isfinite(update):
        raise ValueError(f"training update must be finite, got {training_update!r}")
    if update <= _FAILURE_ENABLE_UPDATE:
        return _INITIAL_MIXTURE
    if update < 15_000.0:
        alpha = (update - _FAILURE_ENABLE_UPDATE) / 10_000.0
        return _finite_positive_triplet(tuple(a + alpha * (b - a) for a, b in zip(_INITIAL_MIXTURE, _MID_MIXTURE)))
    if update < 20_000.0:
        alpha = (update - 15_000.0) / 5_000.0
        return _finite_positive_triplet(tuple(a + alpha * (b - a) for a, b in zip(_MID_MIXTURE, _START_MIXTURE)))
    return _START_MIXTURE


def training_iteration(env) -> float:
    """Return real PPO progress, with an optional preflight-only offset.

    ``common_step_counter`` is a control/environment-step counter in this runner.  The
    trainer may set ``command.training_iteration_offset`` before a short test
    to exercise a later curriculum stage without changing iteration zero in a
    production run.
    """

    command = env.command_manager.get_term("motion")
    counter = float(getattr(env, "common_step_counter", 0))
    offset = float(getattr(command, "training_iteration_offset", 0.0))
    if not math.isfinite(counter) or not math.isfinite(offset):
        raise ValueError("common_step_counter and training_iteration_offset must be finite")
    return max(0.0, counter / _PPO_STEPS_PER_UPDATE + offset)


def delayed_failure_thresholds(training_update: float) -> tuple[float, float]:
    """Return the finite ``(orientation_rad, height_deficit_m)`` D thresholds."""

    update = float(training_update)
    if not math.isfinite(update):
        raise ValueError(f"training update must be finite, got {training_update!r}")
    tighten = min(max((update - _FAILURE_ENABLE_UPDATE) / (_FAILURE_TIGHTEN_UPDATE - _FAILURE_ENABLE_UPDATE), 0.0), 1.0)
    orientation = math.radians(80.0 - 20.0 * tighten)
    height = 0.50 - 0.15 * tighten
    if not (math.isfinite(orientation) and orientation > 0.0 and math.isfinite(height) and height > 0.0):
        raise ValueError("D failure thresholds must remain positive and finite")
    return orientation, height


def validate_aggressive_reference_clip(command) -> bool:
    """Validate native reference tensors before a CPU/GPU preflight reset test."""

    motion = command.motion
    if int(motion.time_step_total) <= 0:
        return False
    tensors = (
        motion.joint_pos,
        motion.joint_vel,
        motion.body_pos_w,
        motion.body_quat_w,
        motion.body_lin_vel_w,
        motion.body_ang_vel_w,
    )
    return all(bool(torch.isfinite(value).all()) for value in tensors)


class AggressiveMotionCommand(MotionCommand):
    """Motion command with bounded reference-state mixture resets for C/D."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env):
        super().__init__(cfg, env)
        self.aggressive_variant = _checked_variant(getattr(cfg, "aggressive_variant", "c"))
        self.training_iteration_offset = float(getattr(cfg, "training_iteration_offset", 0.0))
        if not math.isfinite(self.training_iteration_offset):
            raise ValueError("training_iteration_offset must be finite")
        self.failure_streak_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.failure_grace_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_source = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["aggressive_reset_source"] = self.reset_source.float()
        self.metrics["aggressive_reset_frame0"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["aggressive_reset_transition"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["aggressive_reset_standing"] = torch.zeros(self.num_envs, device=self.device)
        self._clear_aggressive_state()

    def _clear_aggressive_state(self, env_ids: torch.Tensor | None = None) -> None:
        self._failure_eval_step = None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        grace = max(1, math.ceil(_FAILURE_GRACE_SECONDS / float(self._env.step_dt)))
        self.failure_streak_steps[env_ids] = 0
        self.failure_grace_steps[env_ids] = grace
        self.reset_source[env_ids] = 0
        self.metrics["aggressive_reset_source"][env_ids] = 0.0
        for name in ("aggressive_reset_frame0", "aggressive_reset_transition", "aggressive_reset_standing"):
            self.metrics[name][env_ids] = 0.0

    def reset(self, env_ids=None):
        """Clear all D termination/curriculum buffers for command-manager resets."""

        # CommandTerm.reset() must first log the previous metrics. Normalize
        # the manager's default slice(None) because its internal _resample
        # helper expects a sized sequence, then let _resample_command clear
        # episode state immediately before drawing the new source.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return super().reset(env_ids)

    def set_training_iteration_offset(self, offset: float) -> None:
        offset = float(offset)
        if not math.isfinite(offset):
            raise ValueError("training_iteration_offset must be finite")
        self.training_iteration_offset = offset

    def _sample_time_steps(self, env_ids: torch.Tensor):
        # The explicit play flag is checked first by design.  Evaluation can
        # therefore force frame zero even when the config is a C/D mixture.
        if self._play_from_start or self.cfg.sampling_mode == "start":
            super()._sample_time_steps(env_ids)
            self.reset_source[env_ids] = 0
            return

        probabilities = torch.tensor(
            mixture_probabilities(training_iteration(self._env)), dtype=torch.float32, device=self.device
        )
        source = torch.multinomial(probabilities, len(env_ids), replacement=True)
        total_frames = int(self.motion.time_step_total)
        max_frame = max(total_frames - 1, 0)
        self.time_steps[env_ids[source == 0]] = 0
        for source_id, window in ((1, _TRANSITION_WINDOW), (2, _STANDING_WINDOW)):
            selected = env_ids[source == source_id]
            if selected.numel() == 0:
                continue
            phase = torch.rand(selected.numel(), device=self.device)
            frames = torch.floor((window[0] + phase * (window[1] - window[0])) * max_frame).long()
            self.time_steps[selected] = frames.clamp(0, max_frame)
        self.reset_source[env_ids] = source
        self.metrics["aggressive_reset_source"][env_ids] = source.float()
        self.metrics["aggressive_reset_frame0"][env_ids] = (source == 0).float()
        self.metrics["aggressive_reset_transition"][env_ids] = (source == 1).float()
        self.metrics["aggressive_reset_standing"][env_ids] = (source == 2).float()

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._clear_aggressive_state(ids)
        super()._resample_command(ids)

    def set_play_from_start_mode(self):
        super().set_play_from_start_mode()
        self._clear_aggressive_state()


def configure_aggressive_commands(env_cfg, variant: str):
    """Configure the existing ``commands.motion`` term in place and return cfg.

    A/B intentionally remain native ``MotionCommand`` configurations: they
    force frame-zero resets and retain the environment's original horizon.
    C/D use :class:`AggressiveMotionCommand`; the trainer must add
    ``getup_motion_timeout`` as a timeout term and, for D, add
    ``getup_delayed_failure`` as a terminated (not timeout) term.
    """

    selected = _checked_variant(variant)
    motion_cfg = env_cfg.commands.motion
    if selected in {"a", "b"}:
        motion_cfg.class_type = MotionCommand
        motion_cfg.play_from_start = True
        motion_cfg.sampling_mode = "start"
        motion_cfg.resample_at_motion_end = False
        return env_cfg

    motion_cfg.class_type = AggressiveMotionCommand
    motion_cfg.play_from_start = False
    if motion_cfg.sampling_mode == "start":
        motion_cfg.sampling_mode = "adaptive"
    motion_cfg.resample_at_motion_end = False
    motion_cfg.aggressive_variant = selected
    motion_cfg.training_iteration_offset = 0.0
    return env_cfg


def _command_for(env, command_name: str):
    return env.command_manager.get_term(command_name)


def getup_motion_timeout(env, command_name: str = "motion") -> torch.Tensor:
    """Timeout exactly at the native final reference frame, without resampling."""

    command = _command_for(env, command_name)
    final_frame = int(command.motion.time_step_total) - 1
    timed_out = command.time_steps >= final_frame
    if getattr(command, "aggressive_variant", "c") == "d" and not command._play_from_start:
        # Failure wins over timeout. Its result is cached per control step,
        # so either manager term order updates debounce exactly once.
        timed_out &= ~getup_delayed_failure(env, command_name)
    return timed_out


def remaining_reference_frames(command) -> torch.Tensor:
    """Number of reference frames still available, including current frame."""

    return (int(command.motion.time_step_total) - command.time_steps).clamp_min(0)


def getup_delayed_failure(env, command_name: str = "motion") -> torch.Tensor:
    """D-only sustained late-phase failure, returned as termination."""

    command = _command_for(env, command_name)
    step = getattr(env, "common_step_counter", None)
    if step is not None and getattr(command, "_failure_eval_step", None) == step:
        return command._failure_eval_result
    result = _delayed_failure_result(env, command_name)
    if step is not None:
        command._failure_eval_step = step
        command._failure_eval_result = result
    return result


def _delayed_failure_result(env, command_name: str) -> torch.Tensor:

    command: AggressiveMotionCommand = _command_for(env, command_name)
    inactive = torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
    if getattr(command, "aggressive_variant", "c") != "d" or command._play_from_start:
        return inactive

    grace_active = command.failure_grace_steps > 0
    update = training_iteration(env)
    if update < _FAILURE_ENABLE_UPDATE:
        command.failure_grace_steps.sub_(grace_active.long())
        command.failure_streak_steps.zero_()
        return inactive

    ori_limit, height_limit = delayed_failure_thresholds(update)
    reference_phase = command.time_steps.float() / max(int(command.motion.time_step_total) - 1, 1)
    reference_z = command.root_pos_w[:, 2] - command._env.scene.env_origins[:, 2]
    from isaaclab.utils.math import quat_error_magnitude

    orientation_error = quat_error_magnitude(command.root_quat_w, command.robot_root_quat_w)
    bad = (reference_phase >= 0.60) & (reference_z >= 0.80)
    robot_z = command.robot_root_pos_w[:, 2] - command._env.scene.env_origins[:, 2]
    bad &= (orientation_error > ori_limit) | ((reference_z - robot_z) > height_limit)

    bad &= ~grace_active
    next_streak = torch.where(bad, command.failure_streak_steps + 1, torch.zeros_like(command.failure_streak_steps))
    command.failure_grace_steps.sub_(grace_active.long())
    command.failure_streak_steps.copy_(next_streak)
    sustained_steps = max(1, math.ceil(_FAILURE_SUSTAINED_SECONDS / float(env.step_dt)))
    return next_streak >= sustained_steps


def reset_aggressive_termination_buffers(env, env_ids=None, command_name: str = "motion") -> None:
    """Public lifecycle hook for runners that reset termination terms directly."""

    command = _command_for(env, command_name)
    if hasattr(command, "reset"):
        command.reset(env_ids)
