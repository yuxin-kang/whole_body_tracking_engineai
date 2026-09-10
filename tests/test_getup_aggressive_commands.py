"""CPU-only contract tests for the isolated aggressive command module."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def aggressive_module():
    """Load the module without importing the Isaac Lab runtime."""

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls
    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    isaaclab_math.quat_error_magnitude = lambda expected, actual: (expected - actual).abs().sum(dim=-1)
    commands = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.commands")

    class MotionCommand:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

        def _sample_time_steps(self, env_ids):
            self.time_steps[env_ids] = 0

        def _resample_command(self, env_ids):
            # The real parent invokes the subclass hook after logging and
            # clearing metrics. Simulate its post-hook sampled-source write.
            self.metrics["aggressive_reset_source"].fill_(2.0)

        def reset(self, env_ids=None):
            self.parent_reset_logged_source = self.metrics["aggressive_reset_source"].mean().item()
            for metric in self.metrics.values():
                metric.zero_()
            ids = torch.arange(self.metrics["aggressive_reset_source"].numel()) if env_ids is None else env_ids
            self._resample_command(ids)
            return {"parent_metric": self.parent_reset_logged_source}

        def set_play_from_start_mode(self):
            self._play_from_start = True
            self.time_steps.zero_()

    class MotionCommandCfg:
        class_type = MotionCommand
        sampling_mode = "adaptive"

    commands.MotionCommand = MotionCommand
    commands.MotionCommandCfg = MotionCommandCfg
    isolated_modules = {
        "isaaclab": isaaclab,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.math": isaaclab_math,
        "whole_body_tracking.tasks.tracking.mdp.commands": commands,
    }
    with patch.dict(sys.modules, isolated_modules):
        module_path = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/getup_aggressive_commands.py")
        spec = importlib.util.spec_from_file_location("aggressive_commands_isolated", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module


def test_mixture_boundaries_are_finite_and_exact(aggressive_module):
    m = aggressive_module
    assert m.mixture_probabilities(0) == pytest.approx((.4, .4, .2))
    assert m.mixture_probabilities(5_000) == pytest.approx((.4, .4, .2))
    assert m.mixture_probabilities(15_000) == pytest.approx((.8, .15, .05))
    assert m.mixture_probabilities(20_000) == pytest.approx((1.0, 0.0, 0.0))
    for update in (0, 5_001, 10_000, 19_999, 20_000, 30_000):
        values = m.mixture_probabilities(update)
        assert all(torch.isfinite(torch.tensor(values)))
        assert sum(values) == pytest.approx(1.0)


def test_failure_thresholds_tighten_without_invalid_values(aggressive_module):
    m = aggressive_module
    early = m.delayed_failure_thresholds(5_000)
    late = m.delayed_failure_thresholds(10_000)
    assert early == pytest.approx((torch.deg2rad(torch.tensor(80.0)).item(), 0.50))
    assert late == pytest.approx((torch.deg2rad(torch.tensor(60.0)).item(), 0.35))
    assert early[0] > late[0] > 0.0
    assert early[1] > late[1] > 0.0


def test_reference_clip_validation_requires_finite_native_data(aggressive_module):
    m = aggressive_module
    motion = SimpleNamespace(
        time_step_total=379,
        joint_pos=torch.zeros(2),
        joint_vel=torch.zeros(2),
        body_pos_w=torch.zeros(2),
        body_quat_w=torch.zeros(2),
        body_lin_vel_w=torch.zeros(2),
        body_ang_vel_w=torch.zeros(2),
    )
    command = SimpleNamespace(motion=motion)
    assert m.validate_aggressive_reference_clip(command)
    motion.body_pos_w[0] = float("nan")
    assert not m.validate_aggressive_reference_clip(command)


def test_sampler_uses_common_step_counter_and_supports_offset(aggressive_module):
    m = aggressive_module
    command = SimpleNamespace(training_iteration_offset=0.0)
    env = SimpleNamespace(common_step_counter=480_000, command_manager=SimpleNamespace(get_term=lambda _: command))
    assert m.training_iteration(env) == pytest.approx(10_000.0)
    command.training_iteration_offset = 10_000
    assert m.training_iteration(env) == pytest.approx(20_000.0)


def test_training_iteration_uses_control_environment_steps(aggressive_module):
    m = aggressive_module
    command = SimpleNamespace(training_iteration_offset=0.0)
    env = SimpleNamespace(common_step_counter=48, command_manager=SimpleNamespace(get_term=lambda _: command))
    assert m.training_iteration(env) == pytest.approx(1.0)


def test_timeout_has_native_379_frame_off_by_one(aggressive_module):
    m = aggressive_module
    command = SimpleNamespace(motion=SimpleNamespace(time_step_total=379), time_steps=torch.tensor([0, 377, 378, 379]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    assert m.getup_motion_timeout(env).tolist() == [False, False, True, True]
    assert m.remaining_reference_frames(command).tolist() == [379, 2, 1, 0]


def test_timeout_lifecycle_allows_379_actions_from_frame_zero(aggressive_module):
    m = aggressive_module
    command = SimpleNamespace(motion=SimpleNamespace(time_step_total=379), time_steps=torch.tensor([0]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    actions = 0
    while True:
        actions += 1
        if bool(m.getup_motion_timeout(env)[0]):
            break
        command.time_steps += 1
    assert actions == 379


def test_delayed_failure_is_disabled_for_explicit_play_eval(aggressive_module):
    m = aggressive_module
    command = SimpleNamespace(
        aggressive_variant="d",
        _play_from_start=True,
        num_envs=1,
        device="cpu",
        failure_streak_steps=torch.tensor([12]),
        failure_grace_steps=torch.tensor([0]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    assert not bool(m.getup_delayed_failure(env)[0])


def test_delayed_failure_is_terminated_only_after_grace_and_sustain(aggressive_module):
    m = aggressive_module
    command = SimpleNamespace(
        aggressive_variant="d",
        _play_from_start=False,
        num_envs=1,
        device="cpu",
        training_iteration_offset=0.0,
        time_steps=torch.tensor([300]),
        motion=SimpleNamespace(time_step_total=379),
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
        robot_root_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        robot_root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        failure_streak_steps=torch.zeros(1, dtype=torch.long),
        failure_grace_steps=torch.zeros(1, dtype=torch.long),
    )
    env = SimpleNamespace(
        step_dt=0.02,
        common_step_counter=480_000,
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    command._env = env
    assert bool(m.getup_delayed_failure(env)[0]) is False
    result = []
    for _ in range(14):
        env.common_step_counter += 1
        result.append(bool(m.getup_delayed_failure(env)[0]))
    assert result[-1] is True

    command.time_steps.fill_(378)
    env.common_step_counter += 1
    assert not bool(m.getup_motion_timeout(env)[0])
    assert bool(m.getup_delayed_failure(env)[0])
    assert command.failure_streak_steps.item() == 16  # Once per control step despite two terms.

    env.common_step_counter = 100_000
    command.failure_streak_steps.zero_()
    command.failure_grace_steps.fill_(15)
    assert not bool(m.getup_delayed_failure(env)[0])
    assert command.failure_grace_steps.item() == 14


def test_reset_returns_parent_metrics_and_preserves_new_sampled_source(aggressive_module):
    m = aggressive_module
    command = m.AggressiveMotionCommand.__new__(m.AggressiveMotionCommand)
    command.num_envs = 2
    command.device = "cpu"
    command._env = SimpleNamespace(step_dt=0.02)
    command.metrics = {
        "aggressive_reset_source": torch.tensor([1.0, 1.0]),
        "aggressive_reset_frame0": torch.tensor([1.0, 1.0]),
        "aggressive_reset_transition": torch.tensor([0.0, 0.0]),
        "aggressive_reset_standing": torch.tensor([0.0, 0.0]),
    }
    command.failure_streak_steps = torch.tensor([4, 5])
    command.failure_grace_steps = torch.tensor([0, 0])
    command.reset_source = torch.tensor([1, 1])

    result = command.reset()
    assert result == {"parent_metric": 1.0}
    assert command.parent_reset_logged_source == pytest.approx(1.0)
    assert command.metrics["aggressive_reset_source"].tolist() == [2.0, 2.0]

    assert command.failure_streak_steps.tolist() == [0, 0]


def test_configure_a_b_and_c_d_contract(aggressive_module):
    m = aggressive_module
    for variant in ("a", "b"):
        cfg = SimpleNamespace(commands=SimpleNamespace(motion=SimpleNamespace(sampling_mode="adaptive")))
        m.configure_aggressive_commands(cfg, variant)
        assert cfg.commands.motion.play_from_start is True
        assert cfg.commands.motion.sampling_mode == "start"
        assert cfg.commands.motion.resample_at_motion_end is False

    cfg = SimpleNamespace(commands=SimpleNamespace(motion=SimpleNamespace(sampling_mode="start")))
    m.configure_aggressive_commands(cfg, "d")
    assert cfg.commands.motion.class_type is m.AggressiveMotionCommand
    assert cfg.commands.motion.play_from_start is False
    assert cfg.commands.motion.resample_at_motion_end is False
    assert cfg.commands.motion.aggressive_variant == "d"
