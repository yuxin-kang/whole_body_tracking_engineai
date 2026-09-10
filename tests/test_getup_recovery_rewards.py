import importlib.util
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def _load_reward_module():
    """Load the pure reward code without requiring the Isaac Sim application."""
    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")
    managers.RewardTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    utils = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")

    def quat_error_magnitude(first, second):
        dot = torch.sum(first * second, dim=-1).abs().clamp(-1.0, 1.0)
        return 2.0 * torch.acos(dot)

    math_module.quat_error_magnitude = quat_error_magnitude
    package_names = [
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
    ]
    injected_modules = {
        "isaaclab": isaaclab,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_module,
    }
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        injected_modules[package_name] = package
    base_rewards = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.rewards")

    def base_linear_velocity_reward(env, command_name, std, body_names=None):
        command = env.command_manager.get_term(command_name)
        return torch.exp(-torch.square(command.body_lin_vel_w - command.robot_body_lin_vel_w).mean())

    def base_angular_velocity_reward(env, command_name, std, body_names=None):
        command = env.command_manager.get_term(command_name)
        return torch.exp(-torch.square(command.body_ang_vel_w - command.robot_body_ang_vel_w).mean())

    base_rewards.motion_global_body_linear_velocity_error_exp = base_linear_velocity_reward
    base_rewards.motion_global_body_angular_velocity_error_exp = base_angular_velocity_reward
    base_rewards.motion_global_anchor_orientation_error_exp = lambda env, command_name, std: torch.ones(1)
    injected_modules[base_rewards.__name__] = base_rewards

    module_name = "whole_body_tracking.tasks.tracking.mdp.getup_recovery"
    module_path = Path(__file__).parents[1] / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/getup_recovery.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    injected_modules[module_name] = module
    with patch.dict(sys.modules, injected_modules, clear=False):
        spec.loader.exec_module(module)
    return module


reward_module = _load_reward_module()
getup_pose_gated_body_linear_velocity_error_exp = reward_module.getup_pose_gated_body_linear_velocity_error_exp
smoothstep_phase_ramp = reward_module.smoothstep_phase_ramp
apply_getup_recovery_experiment = reward_module.apply_getup_recovery_experiment


def test_loader_restores_injected_modules_after_import():
    names = (
        "isaaclab",
        "isaaclab.managers",
        "isaaclab.utils",
        "isaaclab.utils.math",
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
        "whole_body_tracking.tasks.tracking.mdp.rewards",
        "whole_body_tracking.tasks.tracking.mdp.getup_recovery",
    )
    before = {name: sys.modules.get(name) for name in names}
    _load_reward_module()
    assert all(sys.modules.get(name) is before[name] for name in names)


def test_smoothstep_ramp_boundaries_and_interior():
    phase = torch.tensor([0.0, 0.40, 0.425, 0.45, 0.70, 0.75, 0.80, 1.0])
    ramp = smoothstep_phase_ramp(phase, 0.40, 0.45)
    assert torch.equal(ramp[:2], torch.zeros(2))
    assert torch.isclose(ramp[2], torch.tensor(0.5))
    assert torch.equal(ramp[3:], torch.ones(5))


def test_pose_gate_uses_reference_pose_not_absolute_upright_target():
    class Command:
        time_steps = torch.tensor([9])
        motion = SimpleNamespace(time_step_total=11)
        anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        anchor_quat_w = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        robot_anchor_quat_w = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        body_lin_vel_w = torch.zeros((1, 1, 3))
        robot_body_lin_vel_w = torch.zeros((1, 1, 3))
        cfg = SimpleNamespace(body_names=["body"])

    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: Command()))
    reward = getup_pose_gated_body_linear_velocity_error_exp(
        env, "motion", 1.0, 0.70, 0.80, body_names=["body"]
    )
    assert reward.item() == pytest.approx(1.0, abs=1e-6)


def test_bad_late_pose_reduces_velocity_reward_to_floor():
    class Command:
        time_steps = torch.tensor([10])
        motion = SimpleNamespace(time_step_total=11)
        anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 1.10]])
        anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        body_lin_vel_w = torch.zeros((1, 1, 3))
        robot_body_lin_vel_w = torch.zeros((1, 1, 3))
        cfg = SimpleNamespace(body_names=["body"])

    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: Command()))
    reward = getup_pose_gated_body_linear_velocity_error_exp(
        env, "motion", 1.0, 0.70, 0.80, body_names=["body"]
    )
    assert reward.item() == pytest.approx(0.1, abs=1e-5)


@pytest.mark.parametrize("phase, expected_factor", [(0.70, 1.0), (0.75, 0.55), (0.80, 0.10)])
def test_both_velocity_gates_have_numeric_intermediate_and_boundary_values(phase, expected_factor):
    class Command:
        time_steps = torch.tensor([round(phase * 100)])
        motion = SimpleNamespace(time_step_total=101)
        anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 1.10]])
        anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        body_lin_vel_w = torch.zeros((1, 1, 3))
        robot_body_lin_vel_w = torch.zeros((1, 1, 3))
        body_ang_vel_w = torch.zeros((1, 1, 3))
        robot_body_ang_vel_w = torch.zeros((1, 1, 3))
        cfg = SimpleNamespace(body_names=["body"])

    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: Command()))
    linear = reward_module.getup_pose_gated_body_linear_velocity_error_exp(
        env, "motion", 1.0, 0.70, 0.80, body_names=["body"]
    )
    angular = reward_module.getup_pose_gated_body_angular_velocity_error_exp(
        env, "motion", 3.14, 0.70, 0.80, body_names=["body"]
    )
    assert linear.item() == pytest.approx(expected_factor, abs=1e-5)
    assert angular.item() == pytest.approx(expected_factor, abs=1e-5)


def test_gate_uses_reference_relative_angular_error():
    angle = torch.tensor(0.60)

    class Command:
        time_steps = torch.tensor([80])
        motion = SimpleNamespace(time_step_total=101)
        anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        robot_anchor_quat_w = torch.stack(
            (torch.cos(angle / 2), torch.tensor(0.0), torch.tensor(0.0), torch.sin(angle / 2))
        ).reshape(1, 4)
        body_lin_vel_w = torch.zeros((1, 1, 3))
        robot_body_lin_vel_w = torch.zeros((1, 1, 3))
        cfg = SimpleNamespace(body_names=["body"])

    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: Command()))
    reward = reward_module.getup_pose_gated_body_linear_velocity_error_exp(
        env, "motion", 1.0, 0.70, 0.80, body_names=["body"]
    )
    confidence = torch.exp(-torch.square(angle / 0.60))
    assert reward.item() == pytest.approx((0.10 + 0.90 * confidence).item(), abs=1e-5)


def test_early_phase_preserves_original_velocity_reward_exactly():
    class Command:
        time_steps = torch.tensor([5])
        motion = SimpleNamespace(time_step_total=11)
        anchor_pos_w = torch.tensor([[0.0, 0.0, 0.10]])
        robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 1.10]])
        anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        body_lin_vel_w = torch.zeros((1, 1, 3))
        robot_body_lin_vel_w = torch.zeros((1, 1, 3))
        cfg = SimpleNamespace(body_names=["body"])

    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: Command()))
    reward = getup_pose_gated_body_linear_velocity_error_exp(
        env, "motion", 1.0, 0.70, 0.80, body_names=["body"]
    )
    assert torch.equal(reward, torch.ones(1))


@pytest.mark.parametrize(
    "start,end", [(0.0, 0.0), (-0.1, 0.4), (0.8, 1.1), (0.5, 0.4)]
)
def test_bad_ramp_parameters_are_rejected(start, end):
    with pytest.raises(ValueError):
        smoothstep_phase_ramp(torch.tensor([0.5]), start, end)


def test_apply_rejects_unknown_variant_and_preserves_original_term_settings():
    original_linear = SimpleNamespace(
        func="linear", weight=1.0, params={"command_name": "motion", "std": 1.0, "body_names": ["body"]}
    )
    original_angular = SimpleNamespace(
        func="angular", weight=1.0, params={"command_name": "motion", "std": 3.14, "body_names": ["body"]}
    )
    cfg = SimpleNamespace(rewards=SimpleNamespace(motion_body_lin_vel=original_linear, motion_body_ang_vel=original_angular))
    with pytest.raises(ValueError):
        apply_getup_recovery_experiment(cfg, "bad")
    apply_getup_recovery_experiment(cfg, "ori_velgate")
    for reward, original_func, original_std in (
        (cfg.rewards.motion_body_lin_vel, reward_module.getup_pose_gated_body_linear_velocity_error_exp, 1.0),
        (cfg.rewards.motion_body_ang_vel, reward_module.getup_pose_gated_body_angular_velocity_error_exp, 3.14),
    ):
        assert reward.func is original_func
        assert reward.weight == 1.0
        assert reward.params["std"] == original_std
        assert reward.params["body_names"] == ["body"]


@pytest.mark.parametrize("floor", [0.0, -0.1, float("nan"), float("inf"), 1.1])
def test_invalid_gate_floor_is_rejected(floor):
    with pytest.raises(ValueError):
        reward_module.getup_pose_gated_body_linear_velocity_error_exp(
            SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: None)),
            "motion", 1.0, 0.70, 0.80, floor=floor
        )


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_nonfinite_or_nonpositive_reward_stds_are_rejected(std):
    with pytest.raises(ValueError):
        reward_module.getup_transition_anchor_orientation_error_exp(
            SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: None)),
            "motion", std, 0.40, 0.45
        )
    with pytest.raises(ValueError):
        reward_module.getup_pose_gated_body_linear_velocity_error_exp(
            SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: None)),
            "motion", std, 0.70, 0.80
        )


@pytest.mark.parametrize("height_std, orientation_std", [(0.0, 0.6), (float("nan"), 0.6), (0.25, float("inf"))])
def test_nonfinite_or_nonpositive_pose_confidence_stds_are_rejected(height_std, orientation_std):
    with pytest.raises(ValueError):
        reward_module.getup_pose_gated_body_linear_velocity_error_exp(
            SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: None)),
            "motion", 1.0, 0.70, 0.80, height_std=height_std, orientation_std=orientation_std
        )


def test_recovery_config_restores_baseline_and_registers_both_variants():
    config_path = Path(__file__).parents[1] / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/getup_recovery_env_cfg.py"
    config = config_path.read_text()
    assert "self.rewards.motion_global_anchor_pos.weight = 0.5" in config
    assert "self.rewards.motion_global_anchor_ori.weight = 0.5" in config
    assert "self.rewards.getup_anchor_height.weight = 1.25" in config
    assert 'self.rewards.getup_anchor_height.params["phase_end"] = 0.90' in config
    assert "self.rewards.getup_transition_joint_pos.weight = 1.5" in config
    assert config.count('"getup_final_anchor_') == 3
    assert 'apply_getup_recovery_experiment(self, "ori")' in config
    assert 'apply_getup_recovery_experiment(self, "ori_velgate")' in config

    registry_path = config_path.with_name("__init__.py")
    registry = registry_path.read_text()
    assert 'id="Tracking-Flat-T800-GetUp-Recovery-Ori-v0"' in registry
    assert 'id="Tracking-Flat-T800-GetUp-Recovery-OriVelGate-v0"' in registry
    assert registry.count("T800GetUpPPORunnerCfg") >= 3
