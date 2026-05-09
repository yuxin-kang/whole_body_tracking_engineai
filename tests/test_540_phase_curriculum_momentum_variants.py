import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types


class _FakeRewTerm:
    def __init__(self, func, weight, params):
        self.func = func
        self.weight = weight
        self.params = params


class _FakeSceneEntityCfg:
    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeRobotCfg:
    def replace(self, **kwargs):
        return {"replace_kwargs": kwargs}


_FLAT_ENV_CFG_MODULE = None


def _package_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_flat_env_cfg_module():
    global _FLAT_ENV_CFG_MODULE

    if _FLAT_ENV_CFG_MODULE is not None:
        return _FLAT_ENV_CFG_MODULE

    isaaclab_mod = _package_module("isaaclab")
    isaaclab_managers = types.ModuleType("isaaclab.managers")
    isaaclab_managers.RewardTermCfg = _FakeRewTerm
    isaaclab_managers.SceneEntityCfg = _FakeSceneEntityCfg
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls
    isaaclab_mod.managers = isaaclab_managers
    isaaclab_mod.utils = isaaclab_utils

    whole_body_tracking_mod = _package_module("whole_body_tracking")
    whole_body_tracking_tasks = _package_module("whole_body_tracking.tasks")
    whole_body_tracking_tracking = _package_module("whole_body_tracking.tasks.tracking")
    whole_body_tracking_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    whole_body_tracking_mdp.motion_relative_body_position_error_exp = object()
    whole_body_tracking_mdp.motion_global_body_linear_velocity_error_exp = object()
    whole_body_tracking_mdp.phase_motion_relative_body_position_error_exp = object()
    whole_body_tracking_mdp.phase_motion_global_body_linear_velocity_error_exp = object()
    whole_body_tracking_mdp.phase_motion_global_anchor_xy_velocity_error_exp = object()
    whole_body_tracking_mdp.support_foot_com_distance_reward = object()
    whole_body_tracking_config = _package_module("whole_body_tracking.tasks.tracking.config")
    whole_body_tracking_config_t800 = _package_module("whole_body_tracking.tasks.tracking.config.t800")
    whole_body_tracking_config_t800_t800_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.config.t800.t800_mdp")
    whole_body_tracking_config_t800_t800_mdp.T800_DFS_JOINT_NAMES = []
    whole_body_tracking_config_t800_t800_mdp.T800_MOTION_BODY_NAMES = []

    class _FakeResidualRefJointPositionActionCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.scale = None

    whole_body_tracking_config_t800_t800_mdp.ResidualRefJointPositionActionCfg = _FakeResidualRefJointPositionActionCfg
    whole_body_tracking_agents = _package_module("whole_body_tracking.tasks.tracking.config.t800.agents")
    whole_body_tracking_agents_rsl = types.ModuleType(
        "whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg"
    )
    whole_body_tracking_agents_rsl.LOW_FREQ_SCALE = 2
    whole_body_tracking_tracking_env_cfg = types.ModuleType("whole_body_tracking.tasks.tracking.tracking_env_cfg")

    class _FakeTrackingEnvCfg:
        pass

    whole_body_tracking_tracking_env_cfg.TrackingEnvCfg = _FakeTrackingEnvCfg
    whole_body_tracking_robots = _package_module("whole_body_tracking.robots")
    whole_body_tracking_robots_t800 = types.ModuleType("whole_body_tracking.robots.t800")
    whole_body_tracking_robots_t800.T800_ACTION_SCALE = 1.0
    whole_body_tracking_robots_t800.T800_CFG = _FakeRobotCfg()

    stub_modules = {
        "isaaclab": isaaclab_mod,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.utils": isaaclab_utils,
        "whole_body_tracking": whole_body_tracking_mod,
        "whole_body_tracking.tasks": whole_body_tracking_tasks,
        "whole_body_tracking.tasks.tracking": whole_body_tracking_tracking,
        "whole_body_tracking.tasks.tracking.mdp": whole_body_tracking_mdp,
        "whole_body_tracking.tasks.tracking.config": whole_body_tracking_config,
        "whole_body_tracking.tasks.tracking.config.t800": whole_body_tracking_config_t800,
        "whole_body_tracking.tasks.tracking.config.t800.t800_mdp": whole_body_tracking_config_t800_t800_mdp,
        "whole_body_tracking.tasks.tracking.config.t800.agents": whole_body_tracking_agents,
        "whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg": whole_body_tracking_agents_rsl,
        "whole_body_tracking.tasks.tracking.tracking_env_cfg": whole_body_tracking_tracking_env_cfg,
        "whole_body_tracking.robots": whole_body_tracking_robots,
        "whole_body_tracking.robots.t800": whole_body_tracking_robots_t800,
    }
    original_modules = {name: sys.modules.get(name) for name in stub_modules}

    try:
        sys.modules.update(stub_modules)
        flat_env_cfg_path = Path(
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py"
        )
        spec = importlib.util.spec_from_file_location(
            "whole_body_tracking.tasks.tracking.config.t800.flat_env_cfg",
            flat_env_cfg_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original_module

    _FLAT_ENV_CFG_MODULE = module
    return module


def test_540_momentum_reward_helpers_exist():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()
    rewards = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py").read_text()

    assert "T800_540_PRE_KICK_PHASE = (0.0000, 0.5363)" in config
    assert "def _make_phase_anchor_xy_velocity_reward(" in config
    assert "def _add_540_phase_anchor_xy_velocity_reward(" in config
    assert "phase_motion_global_anchor_xy_velocity_error_exp" in config
    assert "def phase_motion_global_anchor_xy_velocity_error_exp(" in rewards


def test_540_phase_anchor_xy_velocity_reward_helper_attaches_expected_reward():
    flat_env_cfg = _load_flat_env_cfg_module()
    env_cfg = SimpleNamespace(commands=SimpleNamespace(motion=SimpleNamespace()), rewards=SimpleNamespace())

    flat_env_cfg._add_540_phase_anchor_xy_velocity_reward(
        env_cfg,
        weight=1.1,
        std=0.35,
        phase=flat_env_cfg.T800_540_PRE_KICK_PHASE,
    )

    reward = env_cfg.rewards.kick_anchor_xy_vel

    assert reward.weight == 1.1
    assert reward.params["std"] == 0.35
    assert reward.params["phase_start"] == flat_env_cfg.T800_540_PRE_KICK_PHASE[0]
    assert reward.params["phase_end"] == flat_env_cfg.T800_540_PRE_KICK_PHASE[1]


def test_540_phase_curriculum_momentum_env_cfgs_are_defined():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()

    phase_curriculum_block = config.split(
        "class T800Flat540Huixuanti1KickPosPhaseCurriculumEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):",
        1,
    )[1].split("@configclass", 1)[0]

    assert "_set_540_phase_sampling_focus(self, kick_weight=5.5, retract_weight=3.0)" in phase_curriculum_block
    assert "self.commands.motion.reset_preroll_frames = 10" in phase_curriculum_block


def test_540_phase_curriculum_momentum_tasks_are_registered():
    registry = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py").read_text()

    assert 'id="Tracking-Flat-T800-540Huixuanti1-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1EnvCfg' in registry
    assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPos-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosEnvCfg' in registry
    assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPosPhaseCurriculum-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosPhaseCurriculumEnvCfg' in registry
