from __future__ import annotations


G1_MOTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_MOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

G1_TRACKING_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_END_EFFECTOR_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_ROOT_BODY_NAME = "pelvis"
G1_ANCHOR_BODY_NAME = "torso_link"
G1_SHOULDER_BODY_NAMES = ["left_shoulder_roll_link", "right_shoulder_roll_link"]
G1_FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def root_state_quat_xyzw_to_wxyz(root_state: list[float] | tuple[float, ...]) -> list[float]:
    """Convert an upstream root state quaternion slice from xyzw to Isaac wxyz."""
    if len(root_state) < 7:
        raise ValueError("Root state must contain at least position and quaternion values.")
    converted = list(root_state)
    converted[3:7] = [root_state[6], root_state[3], root_state[4], root_state[5]]
    return converted


def apply_standing_reset_branch(
    motion_root_state: list[float],
    sampled_root_state_xyzw: list[float],
    motion_joint_pos: list[float],
    sampled_joint_pos: list[float],
    use_standing: bool,
) -> tuple[list[float], list[float]]:
    """Pure helper mirroring the upstream standing-vs-motion reset branch."""
    if not use_standing:
        return list(motion_root_state), list(motion_joint_pos)

    sampled_root_state = root_state_quat_xyzw_to_wxyz(sampled_root_state_xyzw)
    root_state = list(motion_root_state)
    root_state[:2] = motion_root_state[:2]
    root_state[2:13] = sampled_root_state[2:13]
    return root_state, list(sampled_joint_pos)
