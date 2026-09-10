from pathlib import Path

import numpy as np


CONFIG_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py")
REWARDS_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py")
REGISTRY_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py")
DATA_DIR = Path("data/npz/t800_get_up")

EXPECTED_BODY_NAMES = (
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_TOE",
    "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_TOE",
    "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_L",
    "LINK_WRIST_ROLL_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_PITCH_R",
    "LINK_WRIST_ROLL_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
)


def test_getup_task_is_registered_with_the_t800_ppo_runner():
    registry = REGISTRY_PATH.read_text()

    assert 'id="Tracking-Flat-T800-GetUp-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800FlatGetUpEnvCfg' in registry
    assert '"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg"' in registry
    assert 'id="Tracking-Flat-T800-GetUp-Smooth-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800FlatGetUpSmoothEnvCfg' in registry
    assert '"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg"' in registry


def test_getup_config_uses_start_from_zero_and_no_fall_termination():
    config = CONFIG_PATH.read_text()
    block = config.split("def _configure_get_up_tracking", 1)[1].split("def _apply_punch_improvements", 1)[0]

    expected_lines = (
        'motion_cfg.motion_file = f"{T800_GET_UP_MOTION_DIR}/{motion_name}.npz"',
        'motion_cfg.sampling_mode = "start"',
        "motion_cfg.play_from_start = True",
        "motion_cfg.resample_at_motion_end = False",
        "env_cfg.terminations.anchor_pos = None",
        "env_cfg.terminations.anchor_ori = None",
        "env_cfg.terminations.ee_body_pos = None",
        "motion_cfg.joint_position_range = (0.0, 0.0)",
        "env_cfg.events.push_robot = None",
        "env_cfg.rewards.undesired_contacts = None",
        "class T800FlatGetUpEnvCfg(T800FlatEnvCfg):",
    )
    for line in expected_lines:
        assert line in block


def test_smooth_getup_variant_uses_continuous_clip_and_transition_shaping():
    config = CONFIG_PATH.read_text()

    assert 'T800_GET_UP_SMOOTH_MOTION = "faint_prone_getup_03_t800"' in config
    assert "T800_GET_UP_TRANSITION_PHASE = (0.24, 0.76)" in config
    smooth_block = config.split("class T800FlatGetUpSmoothEnvCfg", 1)[1].split("def _apply_punch_improvements", 1)[0]
    assert "_configure_get_up_tracking(self, T800_GET_UP_SMOOTH_MOTION)" in smooth_block
    assert "self.rewards.getup_transition_joint_pos = _make_phase_joint_position_reward(" in smooth_block
    assert "self.rewards.getup_transition_body_pos = _make_phase_body_position_reward(" in smooth_block
    assert "weight=1.5" in smooth_block


def test_smooth_getup_variant_has_explicit_height_and_vertical_velocity_shaping():
    config = CONFIG_PATH.read_text()
    rewards = REWARDS_PATH.read_text()
    smooth_block = config.split("class T800FlatGetUpSmoothEnvCfg", 1)[1].split("def _apply_punch_improvements", 1)[0]

    assert "self.rewards.getup_anchor_height = _make_phase_anchor_height_reward(" in smooth_block
    assert "self.rewards.getup_anchor_vertical_velocity = _make_phase_anchor_vertical_velocity_reward(" in smooth_block
    assert "weight=3.0" in smooth_block
    assert "phase_end=1.0" in smooth_block
    assert "std=0.18" in smooth_block
    assert "weight=0.35" in smooth_block
    assert "std=0.75" in smooth_block
    assert "def phase_motion_anchor_height_error_exp(" in rewards
    assert "def phase_motion_anchor_vertical_velocity_error_exp(" in rewards
    assert "command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]" in rewards
    assert "command.anchor_lin_vel_w[:, 2] - command.robot_anchor_lin_vel_w[:, 2]" in rewards


def test_smooth_getup_variant_doubles_anchor_tracking_during_final_standing():
    config = CONFIG_PATH.read_text()
    rewards = REWARDS_PATH.read_text()
    smooth_block = config.split("class T800FlatGetUpSmoothEnvCfg", 1)[1].split("def _apply_punch_improvements", 1)[0]

    assert "T800_GET_UP_FINAL_STANDING_PHASE = (0.90, 1.0)" in config
    assert "self.rewards.motion_global_anchor_pos.weight = 3.0" in smooth_block
    assert "self.rewards.motion_global_anchor_ori.weight = 3.0" in smooth_block
    assert "self.rewards.getup_final_anchor_pos = _make_phase_anchor_position_reward(" in smooth_block
    assert "self.rewards.getup_final_anchor_ori = _make_phase_anchor_orientation_reward(" in smooth_block
    assert "self.rewards.getup_final_anchor_height = _make_phase_anchor_height_reward(" in smooth_block
    assert "def phase_motion_global_anchor_position_error_exp(" in rewards
    assert "def phase_motion_global_anchor_orientation_error_exp(" in rewards


def test_getup_smooth_training_script_and_sync_mapping_are_defaulted_together():
    train_script = Path("scripts/slurm/train_t800_getup_rtx4000.sbatch").read_text()
    sync_script = Path("scripts/sync_latest_ckpt_to_client").read_text()

    assert 'TASK="${TASK:-Tracking-Flat-T800-GetUp-Smooth-v0}"' in train_script
    assert 'MOTION_FILE="${MOTION_FILE:-data/npz/t800_get_up/faint_prone_getup_03_t800.npz}"' in train_script
    assert "*getup_smooth*" in sync_script
    assert "Tracking-Flat-T800-GetUp-Smooth-v0" in sync_script


def test_generated_getup_npz_files_match_tracking_contract():
    motions = sorted(DATA_DIR.glob("*.npz"))
    assert {motion.name for motion in motions} == {
        "faint_prone_getup_03_t800.npz",
        "getup1_t800.npz",
        "getup4_t800.npz",
        "prone_getup_02_t800.npz",
    }

    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "body_schema_version",
    }
    for motion_path in motions:
        with np.load(motion_path, allow_pickle=False) as motion:
            assert required <= set(motion.files)
            frame_count = motion["joint_pos"].shape[0]
            assert motion["joint_pos"].shape == (frame_count, 25)
            assert motion["joint_vel"].shape == (frame_count, 25)
            assert motion["body_pos_w"].shape == (frame_count, 34, 3)
            assert motion["body_quat_w"].shape == (frame_count, 34, 4)
            assert motion["body_lin_vel_w"].shape == (frame_count, 34, 3)
            assert motion["body_ang_vel_w"].shape == (frame_count, 34, 3)
            assert float(motion["fps"][0]) == 50.0
            assert int(motion["body_schema_version"][0]) == 1
            assert tuple(str(name) for name in motion["body_names"].tolist()) == EXPECTED_BODY_NAMES
            assert float(motion["source_key_body_fk_max_error_m"][0]) < 1.0e-4
            assert np.isfinite(motion["body_pos_w"]).all()
            assert float(motion["body_pos_w"][:, 0, 2].min()) < 0.4
