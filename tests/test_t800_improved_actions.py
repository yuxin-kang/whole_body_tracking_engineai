import ast
from pathlib import Path

import numpy as np


CONFIG_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py")
REGISTRY_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py")
COMMANDS_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py")
REWARDS_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py")
CANONICAL_DIR = Path("data/npz/traj_eng_50hz_canonical")
IMPROVED_DIR = Path("data/npz/traj_eng_50hz_improved")

IMPROVED_MOTIONS = {
    "straight_punch_L_terminal_hold_0p5s.npz": ("straight_punch_L.npz", 25),
    "straight_punch_R_terminal_hold_0p5s.npz": ("straight_punch_R.npz", 25),
    "left_hook_001_terminal_hold_0p5s.npz": ("left_hook_001.npz", 25),
    "roundhouse_kick_001_retimed_terminal_hold_1s.npz": ("roundhouse_kick_001.npz", 50),
}


def _class_block(source: str, class_name: str) -> str:
    return source.split(f"class {class_name}", 1)[1].split("@configclass", 1)[0]


def test_improved_tasks_are_separate_from_existing_baselines():
    registry = REGISTRY_PATH.read_text()
    expected = {
        "StraightPunchL": "T800FlatImprovedStraightPunchLEnvCfg",
        "StraightPunchR": "T800FlatImprovedStraightPunchREnvCfg",
        "LeftHook": "T800FlatImprovedLeftHookEnvCfg",
        "RoundhouseKick": "T800FlatImprovedRoundhouseKickEnvCfg",
    }

    for task_name, config_name in expected.items():
        assert f'id="Tracking-Flat-T800-Improved-{task_name}-v0"' in registry
        assert f'"env_cfg_entry_point": flat_env_cfg.{config_name}' in registry

    config = CONFIG_PATH.read_text()
    baseline_blocks = (
        _class_block(config, "T800FlatBaselineStraightPunchLEnvCfg"),
        _class_block(config, "T800FlatBaselineStraightPunchREnvCfg"),
        _class_block(config, "T800FlatBaselineLeftHookEnvCfg"),
    )
    assert all("_apply_" not in block for block in baseline_blocks)


def test_improved_tasks_use_only_improved_canonical_motion_paths():
    source = CONFIG_PATH.read_text()
    assert 'T800_IMPROVED_MOTION_DIR = "data/npz/traj_eng_50hz_improved"' in source

    expected = {
        "T800FlatImprovedStraightPunchLEnvCfg": ("straight_punch_L_terminal_hold_0p5s", 1.70),
        "T800FlatImprovedStraightPunchREnvCfg": ("straight_punch_R_terminal_hold_0p5s", 1.50),
        "T800FlatImprovedLeftHookEnvCfg": ("left_hook_001_terminal_hold_0p5s", 3.10),
        "T800FlatImprovedRoundhouseKickEnvCfg": ("roundhouse_kick_001_retimed_terminal_hold_1s", 4.36),
    }
    classes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    for class_name, expected_args in expected.items():
        calls = [node for node in ast.walk(classes[class_name]) if isinstance(node, ast.Call)]
        configure_call = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "_configure_short_episode_baseline"
        )
        assert tuple(ast.literal_eval(arg) for arg in configure_call.args[1:]) == expected_args
        assert any(
            keyword.arg == "motion_dir"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "T800_IMPROVED_MOTION_DIR"
            for keyword in configure_call.keywords
        )


def test_improved_motion_prefix_is_canonical_and_hold_is_stationary():
    for output_name, (source_name, hold_frames) in IMPROVED_MOTIONS.items():
        with np.load(CANONICAL_DIR / source_name, allow_pickle=False) as source:
            source_arrays = {key: np.asarray(source[key]) for key in source.files}
        with np.load(IMPROVED_DIR / output_name, allow_pickle=False) as improved:
            assert int(np.asarray(improved["body_schema_version"]).reshape(-1)[0]) == 1
            assert bool(np.asarray(improved["derived_from_canonical"]).reshape(-1)[0])
            is_retimed = "derived_source_frame_coordinates" in improved
            is_drift_corrected = "derived_removed_foot_midpoint_drift_xy" in improved
            motion_frames = improved["joint_pos"].shape[0] - hold_frames
            if not is_retimed and not is_drift_corrected:
                assert motion_frames == source_arrays["joint_pos"].shape[0]

            for key in ("joint_pos", "body_pos_w", "body_quat_w"):
                if not is_retimed and not is_drift_corrected:
                    np.testing.assert_array_equal(improved[key][:motion_frames], source_arrays[key])
                elif is_drift_corrected and key in ("joint_pos", "body_quat_w"):
                    np.testing.assert_array_equal(improved[key][:motion_frames], source_arrays[key])
                expected_tail = np.repeat(improved[key][motion_frames - 1 : motion_frames], hold_frames, axis=0)
                np.testing.assert_array_equal(improved[key][-hold_frames:], expected_tail)
            for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                if not is_retimed and not is_drift_corrected:
                    np.testing.assert_array_equal(improved[key][:motion_frames], source_arrays[key])
                elif is_drift_corrected and key in ("joint_vel", "body_ang_vel_w"):
                    np.testing.assert_array_equal(improved[key][:motion_frames], source_arrays[key])
                np.testing.assert_array_equal(improved[key][-hold_frames:], 0.0)


def test_left_hook_removes_only_global_planar_drift():
    source_path = CANONICAL_DIR / "left_hook_001.npz"
    output_path = IMPROVED_DIR / "left_hook_001_terminal_hold_0p5s.npz"
    with np.load(source_path, allow_pickle=False) as source, np.load(output_path, allow_pickle=False) as improved:
        motion_frames = source["joint_pos"].shape[0]
        drift = np.asarray(improved["derived_removed_foot_midpoint_drift_xy"])
        np.testing.assert_allclose(improved["body_pos_w"][:motion_frames, :, :2], source["body_pos_w"][:, :, :2] - drift[:, None, :])
        np.testing.assert_array_equal(improved["body_pos_w"][:motion_frames, :, 2], source["body_pos_w"][:, :, 2])
        np.testing.assert_array_equal(improved["joint_pos"][:motion_frames], source["joint_pos"])

        body_names = tuple(str(name) for name in improved["body_names"].tolist())
        feet = improved["body_pos_w"][:motion_frames, [body_names.index("LINK_ANKLE_ROLL_L"), body_names.index("LINK_ANKLE_ROLL_R")], :2]
        foot_midpoint = feet.mean(axis=1)
        assert float(np.max(np.linalg.norm(foot_midpoint - foot_midpoint[0], axis=1))) < 1.0e-5


def test_roundhouse_local_retime_reduces_peak_joint_velocity_without_changing_endpoints():
    source_path = CANONICAL_DIR / "roundhouse_kick_001.npz"
    output_path = IMPROVED_DIR / "roundhouse_kick_001_retimed_terminal_hold_1s.npz"
    with np.load(source_path, allow_pickle=False) as source, np.load(output_path, allow_pickle=False) as improved:
        coordinates = np.asarray(improved["derived_source_frame_coordinates"])
        motion_frames = len(coordinates)
        np.testing.assert_allclose(improved["joint_pos"][0], source["joint_pos"][0], atol=1.0e-6)
        np.testing.assert_allclose(improved["joint_pos"][motion_frames - 1], source["joint_pos"][-1], atol=1.0e-6)
        assert float(np.max(np.abs(improved["joint_vel"][:motion_frames, 6]))) < 25.96
        assert float(np.max(np.abs(improved["joint_vel"][:motion_frames, 8]))) < 23.19


def test_prepare_script_rejects_legacy_motion_as_input(tmp_path):
    import importlib.util

    script_path = Path("scripts/prepare_t800_improved_motions.py")
    spec = importlib.util.spec_from_file_location("prepare_t800_improved_motions", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)

    legacy = Path("data/npz/traj_eng_50hz/straight_punch_L.npz")
    try:
        module.prepare_motion(legacy, tmp_path / "should_not_exist.npz", 0.5)
    except ValueError as error:
        assert "missing canonical arrays" in str(error)
    else:
        raise AssertionError("Legacy motion must not be accepted by the improved-motion generator")


def test_punch_and_hook_rewards_target_the_active_fist_only():
    source = CONFIG_PATH.read_text()
    punch_helper = source.split("def _apply_punch_improvements", 1)[1].split("def _apply_hook_improvements", 1)[0]
    hook_helper = source.split("def _apply_hook_improvements", 1)[1].split("def _apply_roundhouse_improvements", 1)[0]

    assert "wrist_body_name=wrist_body_name" in punch_helper
    assert "body_names=[hand_body_name]" in punch_helper
    assert "T800_TRACKING_END_EFFECTOR_BODY_NAMES" not in punch_helper
    assert 'wrist_body_name="LINK_WRIST_END_L"' in hook_helper
    assert "T800_LEFT_HOOK_STRIKE_PHASE" in hook_helper
    assert "T800_LEFT_HOOK_JOINT_NAMES" in hook_helper

    rewards = REWARDS_PATH.read_text()
    assert "def motion_fixed_offset_body_position_error_exp(" in rewards
    assert "def phase_motion_fixed_offset_body_position_error_exp(" in rewards


def test_feet_slip_is_wired_for_plain_motion_command():
    commands = COMMANDS_PATH.read_text()
    config = CONFIG_PATH.read_text()

    assert "self.feet_indexes = [self.cfg.body_names.index(name) for name in self.cfg.feet_body_names]" in commands
    assert "class MotionCommandCfg" in commands
    assert "feet_body_names: list[str] = []" in commands
    assert "self.commands.motion.feet_body_names = T800_SUPPORT_FOOT_BODY_NAMES" in config
    assert "env_cfg.rewards.foot_slip = _make_feet_slip_reward" in config


def test_improved_tasks_keep_only_height_fall_reset():
    source = CONFIG_PATH.read_text()
    helper = source.split("def _apply_roundhouse_improvements", 1)[1].split("@configclass", 1)[0]

    assert "T800_ROUNDHOUSE_BODY_NAMES" in helper
    assert "T800_ROUNDHOUSE_KICK_PHASE" in helper
    assert "T800_ROUNDHOUSE_JOINT_NAMES" in helper
    assert "T800_ROUNDHOUSE_EXTENSION_PHASE" in helper
    assert "J06_HIP_PITCH_R" in source
    assert "J09_HIP_PITCH_R" not in source
    assert "env_cfg.terminations.fall_orientation = None" in helper

    for class_name in (
        "T800FlatImprovedStraightPunchLEnvCfg",
        "T800FlatImprovedStraightPunchREnvCfg",
        "T800FlatImprovedLeftHookEnvCfg",
        "T800FlatImprovedRoundhouseKickEnvCfg",
    ):
        assert "_keep_height_fall_only(self)" in _class_block(source, class_name)


def test_punches_and_hook_use_narrower_ground_randomization():
    source = CONFIG_PATH.read_text()
    helper = source.split("def _apply_stable_ground_randomization", 1)[1].split("@configclass", 1)[0]
    assert 'material_params["static_friction_range"] = (0.7, 1.2)' in helper
    assert 'material_params["dynamic_friction_range"] = (0.6, 1.0)' in helper
    assert 'material_params["restitution_range"] = (0.0, 0.1)' in helper

    for class_name in (
        "T800FlatImprovedStraightPunchLEnvCfg",
        "T800FlatImprovedStraightPunchREnvCfg",
        "T800FlatImprovedLeftHookEnvCfg",
    ):
        assert "_apply_stable_ground_randomization(self)" in _class_block(source, class_name)

    roundhouse = _class_block(source, "T800FlatImprovedRoundhouseKickEnvCfg")
    assert "_apply_stable_ground_randomization(self)" not in roundhouse
