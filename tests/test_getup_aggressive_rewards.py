import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _load_module():
    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")
    managers.RewardTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    utils = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")

    def quat_apply(quat, vec):
        q_xyz = quat[:, 1:]
        q_w = quat[:, :1]
        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)

    math_module.quat_apply = quat_apply
    injected = {
        "isaaclab": isaaclab,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_module,
    }
    for name in (
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        injected[name] = package

    rewards = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.rewards")

    def anchor_orientation(env, command_name, std):
        return torch.ones(1)

    def anchor_position(env, command_name, std):
        return torch.ones(1)

    rewards.motion_global_anchor_orientation_error_exp = anchor_orientation
    rewards.motion_global_anchor_position_error_exp = anchor_position
    injected[rewards.__name__] = rewards

    recovery = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.getup_recovery")

    def smoothstep(phase, start, end):
        unit = ((phase - start) / (end - start)).clamp(0.0, 1.0)
        return unit * unit * (3.0 - 2.0 * unit)

    def confidence(command, height_std, orientation_std):
        height_error = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
        return torch.exp(-torch.square(height_error / height_std))

    recovery.smoothstep_phase_ramp = smoothstep
    recovery._pose_confidence = confidence
    recovery.getup_pose_gated_body_linear_velocity_error_exp = lambda *args, **kwargs: torch.ones(1)
    recovery.getup_pose_gated_body_angular_velocity_error_exp = lambda *args, **kwargs: torch.ones(1)
    injected[recovery.__name__] = recovery

    module_name = "whole_body_tracking.tasks.tracking.mdp.getup_aggressive_rewards"
    path = Path(__file__).parents[1] / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/getup_aggressive_rewards.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    injected[module_name] = module
    before = {name: sys.modules.get(name) for name in injected}
    with patch.dict(sys.modules, injected, clear=False):
        spec.loader.exec_module(module)
    assert all(sys.modules.get(name) is before[name] for name in injected)
    return module


aggressive = _load_module()


def _command(phase, *, wrong_pose=True, wrong_angle=False):
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    robot_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]]) if wrong_angle else quat.clone()
    return SimpleNamespace(
        time_steps=torch.tensor([round(phase * 100)]),
        motion=SimpleNamespace(time_step_total=101),
        anchor_pos_w=torch.tensor([[0.0, 0.0, 0.10]]),
        robot_anchor_pos_w=torch.tensor([[1.0, 0.0, 1.10 if wrong_pose else 0.10]]),
        anchor_quat_w=quat,
        robot_anchor_quat_w=robot_quat,
        anchor_ang_vel_w=torch.zeros((1, 3)),
        robot_anchor_ang_vel_w=torch.zeros((1, 3)),
        robot_anchor_lin_vel_w=torch.zeros((1, 3)),
    )


def _env(command):
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))


@pytest.mark.parametrize("phase, expected", [(0.35, 0.0), (0.40, 0.5), (0.45, 1.0), (0.60, 1.0)])
def test_orientation_ramp_edges_and_interior(phase, expected):
    reward = aggressive.aggressive_anchor_orientation_error_exp(_env(_command(phase, wrong_pose=False)), "motion", 0.6, 0.35, 0.45)
    assert reward.item() == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("phase, expected", [(0.65, 1.0), (0.725, 0.55), (0.80, 0.10)])
def test_new_anchor_angular_velocity_pose_gate_edges_and_wrong_pose(phase, expected):
    command = _command(phase)
    if phase == 0.725:
        command.time_steps = torch.tensor([145])
        command.motion = SimpleNamespace(time_step_total=201)
    reward = aggressive.aggressive_pose_gated_anchor_angular_velocity_error_exp(
        _env(command), "motion", 1.0, 0.35, 0.45, 0.65, 0.80
    )
    assert reward.item() == pytest.approx(expected, abs=1e-5)


def test_standing_quality_matches_product_and_late_ramp():
    command = _command(0.86, wrong_pose=False)
    command.robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command.robot_anchor_ang_vel_w = torch.tensor([[0.5, 0.0, 0.0]])
    command.robot_anchor_lin_vel_w = torch.tensor([[0.25, 0.0, 0.0]])
    reward = aggressive.aggressive_standing_quality_reward(_env(command), "motion", 0.825, 0.86)
    assert reward.item() == pytest.approx(torch.exp(torch.tensor(-1.0 - 1.0)).item(), abs=1e-5)


def test_standing_quality_uses_batched_up_vector_for_quat_apply():
    command = _command(0.86, wrong_pose=False)
    command.time_steps = torch.tensor([86, 86])
    command.robot_anchor_quat_w = command.robot_anchor_quat_w.repeat(2, 1)
    command.anchor_pos_w = command.anchor_pos_w.repeat(2, 1)
    command.robot_anchor_pos_w = command.robot_anchor_pos_w.repeat(2, 1)
    command.robot_anchor_ang_vel_w = command.robot_anchor_ang_vel_w.repeat(2, 1)
    command.robot_anchor_lin_vel_w = command.robot_anchor_lin_vel_w.repeat(2, 1)
    reward = aggressive.aggressive_standing_quality_reward(_env(command), "motion", 0.825, 0.86)
    assert reward.shape == (2,)


def test_apply_variants_keep_baseline_velocity_weight_and_std():
    def term(func, std):
        return SimpleNamespace(func=func, weight=1.0, params={"command_name": "motion", "std": std, "body_names": ["body"]})

    for variant in "abcd":
        rewards = SimpleNamespace(motion_body_lin_vel=term("lin", 1.0), motion_body_ang_vel=term("ang", 3.14))
        cfg = SimpleNamespace(rewards=rewards)
        aggressive.apply_aggressive_rewards(cfg, variant)
        assert rewards.motion_body_lin_vel.weight == 1.0
        assert rewards.motion_body_lin_vel.params["std"] == 1.0
        assert rewards.motion_body_ang_vel.weight == 1.0
        assert rewards.motion_body_ang_vel.params["std"] == 3.14
        assert hasattr(rewards, "getup_aggressive_anchor_ang_vel")
        assert (hasattr(rewards, "getup_aggressive_anchor_pos")) is (variant in "bcd")
        assert (hasattr(rewards, "getup_aggressive_standing_quality")) is (variant == "d")


def test_invalid_variant_is_rejected():
    with pytest.raises(ValueError):
        aggressive.apply_aggressive_rewards(SimpleNamespace(rewards=SimpleNamespace()), "A")
