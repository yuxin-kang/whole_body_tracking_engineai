"""CPU contracts for brief, ground-only forearm shaping."""

import importlib.util
import ast
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _load():
    math_module = types.ModuleType("isaaclab.utils.math")

    def quat_apply(q, v):
        t = 2 * torch.cross(q[:, 1:], v, dim=-1)
        return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)

    math_module.quat_apply = quat_apply
    recovery = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.getup_recovery")

    def ramp(p, start, end):
        t = ((p - start) / (end - start)).clamp(0, 1)
        return t.square() * (3 - 2 * t)

    recovery.smoothstep_phase_ramp = ramp
    name = "whole_body_tracking.tasks.tracking.mdp.getup_forearm_support"
    path = Path(__file__).parents[1] / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/getup_forearm_support.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {math_module.__name__: math_module, recovery.__name__: recovery}):
        spec.loader.exec_module(module)
    return module


support = _load()
WINDOW = (.08, .12, .30, .38)
SENSORS = ("forearm_l", "wrist_l", "forearm_r", "wrist_r")


class Scene(dict):
    @property
    def sensors(self):
        return self


def fixture_env(batch=1):
    # Deliberately not left-right or kinematic body ordering.
    positions = torch.tensor([[[0., 0., .1], [1., 0., .4], [1., 0., .1], [0., 0., .4]]]).repeat(batch, 1, 1)
    data = SimpleNamespace(
        body_pos_w=positions,
        body_quat_w=torch.tensor([1., 0., 0., 0.]).expand(batch, 4, 4).clone(),
        body_com_pos_w=positions.clone(),
        body_lin_vel_w=torch.zeros(batch, 4, 3),
        body_ang_vel_w=torch.zeros(batch, 4, 3),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    for name in SENSORS:
        matrix = torch.zeros(batch, 1, 1, 3)
        matrix[..., 2] = 30 if name.startswith("wrist") else 0
        scene[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=matrix, net_forces_w=torch.full((batch, 1, 3), 1000.)))
    command = SimpleNamespace(time_steps=torch.full((batch,), 20), motion=SimpleNamespace(time_step_total=101))
    return SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda _: command))


def params():
    return dict(
        command_name="motion", wrist_cfg=SimpleNamespace(name="robot", body_ids=[0, 2]),
        wrist_offsets_b=((0., 0., 0.), (0., 0., 0.)),
        phase_window=WINDOW, force_min=5., force_full=30.,
    )


def vertical(env):
    return support.getup_forearm_vertical_reward(
        env, **params(), elbow_cfg=SimpleNamespace(name="robot", body_ids=[3, 1]),
        ground_sensor_names=SENSORS, std=.35,
    )


def slip(env):
    return support.getup_wrist_support_slip_penalty(
        env, **params(), wrist_sensor_names=("wrist_l", "wrist_r"), speed_std=1., sphere_radius=.05,
    )


def test_world_vertical_prefers_elbow_above_wrist_and_is_batched():
    wrists = torch.zeros(1, 4, 3)
    elbows = torch.tensor([[[0., 0., 1.], [1., 0., 0.], [0., 0., -1.], [0., 0., 0.]]])
    scores = support.forearm_vertical_score(elbows, wrists, .35)
    assert scores.shape == (1, 4)
    assert scores[0, 0] == 1
    assert scores[0, 1] < 1e-8
    assert scores[0, 2] < 1e-20
    assert scores[0, 3] == 0
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("phase,expected", [(0., 0.), (.08, 0.), (.10, .5), (.12, 1.), (.30, 1.), (.34, .5), (.38, 0.), (1., 0.)])
def test_phase_window_enters_and_exits_smoothly(phase, expected):
    assert support.support_phase_gate(torch.tensor([phase]), WINDOW).item() == pytest.approx(expected, abs=1e-6)


def test_force_is_upward_and_saturates_instead_of_rewarding_impacts():
    forces = torch.tensor([-100., 0., 5., 17.5, 30., 10000.])
    assert torch.allclose(support.ground_support_gate(forces, 5, 30), torch.tensor([0., 0., 0., .5, 1., 1.]))


def test_names_resolve_scrambled_body_order_and_arms_score_independently():
    env = fixture_env(batch=2)
    env.scene["wrist_l"].data.force_matrix_w[1] = 0
    assert torch.allclose(vertical(env), torch.tensor([1., .5]))
    env.scene["wrist_r"].data.force_matrix_w[1] = 0
    assert torch.allclose(vertical(env), torch.tensor([1., 0.]))


def test_self_contact_or_airborne_pose_is_not_floor_support():
    env = fixture_env()
    for name in SENSORS:
        env.scene[name].data.force_matrix_w.zero_()
    assert vertical(env).item() == 0
    assert slip(env).item() == 0


def test_only_touching_side_counts_and_loss_of_support_cannot_improve_score():
    env = fixture_env()
    env.scene["robot"].data.body_pos_w[:, 1, 0] += 1.
    full = vertical(env)
    env.scene["wrist_r"].data.force_matrix_w.zero_()
    assert vertical(env) <= full
    assert vertical(env).item() == pytest.approx(.5)


@pytest.mark.parametrize("phase", [0, 8, 38, 100])
def test_late_recontact_and_initial_lying_do_not_trigger(phase):
    env = fixture_env()
    env.command_manager.get_term("motion").time_steps.fill_(phase)
    env.scene["robot"].data.body_lin_vel_w[..., 0] = 1.
    assert vertical(env).item() == 0
    assert slip(env).item() == 0


def test_slip_is_tangential_bounded_and_wrist_contact_only():
    env = fixture_env()
    env.scene["robot"].data.body_lin_vel_w[..., 2] = 20.
    assert slip(env).item() == 0
    env.scene["robot"].data.body_lin_vel_w[..., 0] = 1.
    assert slip(env).item() == pytest.approx(1 - math.exp(-1))
    env.scene["wrist_l"].data.force_matrix_w.zero_()
    env.scene["wrist_r"].data.force_matrix_w.zero_()
    for name in ("forearm_l", "forearm_r"):
        env.scene[name].data.force_matrix_w[..., 2] = 100.
    assert slip(env).item() == 0
    assert vertical(env).item() == 0


def test_slip_uses_contact_point_velocity_not_com_velocity():
    env = fixture_env()
    data = env.scene["robot"].data
    data.body_ang_vel_w[..., 1] = 2.
    # Sphere contact is 5cm below COM: omega cross r gives vx=-0.1.
    assert slip(env).item() == pytest.approx(1 - math.exp(-.1**2), abs=1e-6)
    data.body_lin_vel_w[..., 0] = .1
    assert slip(env).item() == pytest.approx(0., abs=1e-6)


def test_wrist_offsets_follow_world_rotation():
    env = fixture_env()
    data = env.scene["robot"].data
    data.body_quat_w[:] = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
    centers = support._wrist_centers(env.scene["robot"], params()["wrist_cfg"], ((0., 0., -.1), (0., 0., -.1)))
    expected = data.body_pos_w[:, [0, 2]] + torch.tensor([-.1, 0., 0.])
    assert torch.allclose(centers, expected, atol=1e-6)


def test_missing_ground_filter_fails_instead_of_using_unfiltered_contacts():
    env = fixture_env()
    env.scene["wrist_l"].data.force_matrix_w = None
    with pytest.raises(RuntimeError, match="ground contact filter"):
        vertical(env)


@pytest.mark.parametrize("window", [(0, 0, .2, .3), (0, .2, .1, .3), (.1, .2, .3, 1.1)])
def test_invalid_phase_windows_fail(window):
    with pytest.raises(ValueError):
        support.support_phase_gate(torch.tensor([.2]), window)


def test_four_experiments_preserve_c_terms_and_only_vary_two_weights():
    managers = types.ModuleType("isaaclab.managers")
    managers.RewardTermCfg = lambda **kw: SimpleNamespace(**kw)
    managers.SceneEntityCfg = lambda name, **kw: SimpleNamespace(name=name, **kw)
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = lambda **kw: SimpleNamespace(**kw)
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    base = types.ModuleType("whole_body_tracking.tasks.tracking.config.t800.getup_aggressive_env_cfg")
    base.T800GetUpAggressiveCEnvCfg = type("C", (), {})
    name = "whole_body_tracking.tasks.tracking.config.t800.getup_forearm_env_cfg"
    path = Path(__file__).parents[1] / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/getup_forearm_env_cfg.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    injected = {m.__name__: m for m in (managers, sensors, utils, base, support)}
    with patch.dict(sys.modules, injected):
        spec.loader.exec_module(module)
    for variant, (vertical_weight, slip_weight) in {
        "f1": (1, 0), "f2": (3, 0), "f3": (1, -.5), "f4": (3, -.5)
    }.items():
        existing = object()
        robot = object()
        motion = SimpleNamespace(aggressive_variant="c", play_from_start=False, training_iteration_offset=0.)
        cfg = SimpleNamespace(
            scene=SimpleNamespace(robot=robot, terrain=SimpleNamespace(prim_path="/World/ground")),
            rewards=SimpleNamespace(existing=existing), commands=SimpleNamespace(motion=motion),
        )
        module.apply_getup_forearm_experiment(cfg, variant)
        assert cfg.scene.robot is robot and cfg.rewards.existing is existing
        assert cfg.commands.motion is motion and not motion.play_from_start and motion.training_iteration_offset == 0.
        assert cfg.rewards.getup_forearm_vertical.weight == vertical_weight
        assert cfg.rewards.getup_wrist_support_slip.weight == slip_weight
        assert len(vars(cfg.rewards)) == 3
        sensors = [v for k, v in vars(cfg.scene).items() if k.startswith("getup_ground_")]
        assert len(sensors) == 4
        assert len({s.prim_path for s in sensors}) == 4
        assert all(s.filter_prim_paths_expr == ["/World/ground/terrain/GroundPlane/CollisionPlane"] for s in sensors)


def test_finetune_log_checks_losses_and_updates_status_without_isaac(tmp_path):
    # Exercise the real override, including its actual imports. An absent
    # runtime math import previously stopped all four preflight processes.
    path = Path(__file__).parents[1] / "scripts/rsl_rl/finetune_getup_forearm.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if
            (isinstance(node, ast.Import) and all(alias.name in {"math", "json"} for alias in node.names))
            or (isinstance(node, ast.ClassDef) and node.name == "ForearmRunner")]

    class Base:
        def log(self, locs):
            self.parent_called = True

    namespace = {"MotionOnPolicyRunner": Base, "args": SimpleNamespace(preflight=False), "RESUME_ITERATION": 18501}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    runner = namespace["ForearmRunner"]()
    runner.forearm_manifest = {}
    runner.forearm_manifest_path = tmp_path / "manifest.json"
    runner.log({"it": 18501, "loss_dict": {"surrogate": .5, "value": 1.}})
    assert runner.parent_called
    assert runner.forearm_manifest == {"status": "training", "last_iteration": 18501}
    with pytest.raises(RuntimeError, match="Non-finite"):
        runner.log({"it": 18502, "loss_dict": {"value": float("nan")}})
