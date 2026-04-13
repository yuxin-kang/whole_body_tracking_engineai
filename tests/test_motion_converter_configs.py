from pathlib import Path


def test_csv_to_npz_uses_25_t800_joints():
    script = Path("scripts/csv_to_npz.py").read_text()

    assert "from whole_body_tracking.tasks.tracking.config.t800.t800_mdp import T800_DFS_JOINT_NAMES, T800_MOTION_BODY_NAMES" in script
    assert "*T800_DFS_JOINT_NAMES" in script
    assert 'self.motion_dof_poss_input = motion[:, 7 : 7 + len(joint_names)]' in script
    assert 'log["joint_pos"].append(robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy())' in script


def test_npy_to_npz_accepts_32_and_40_columns():
    script = Path("scripts/npy_to_npz.py").read_text()

    assert "if motion.ndim != 2 or motion.shape[1] not in (32, 40):" in script
    assert "motion.shape[1] == 32" in script
    assert "self.motion_base_rots_input = motion[:, 3:7]" in script
    assert "self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]" in script
