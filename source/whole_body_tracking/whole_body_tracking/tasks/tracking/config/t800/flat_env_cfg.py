from math import cos, sin

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg

import whole_body_tracking.tasks.tracking.mdp as mdp

from . import t800_mdp
from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, TrackingEnvCfg


T800_TRACKING_END_EFFECTOR_BODY_NAMES = [
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
]
T800_SUPPORT_FOOT_BODY_NAMES = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
T800_540_KICK_BODY_NAMES = ["LINK_ANKLE_ROLL_R"]
T800_540_KICK_JOINT_NAMES = ["J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R"]
T800_540_KICK_PHASE = (0.3841, 0.5363)
T800_540_KICK_PHASE_LATE = (0.4637, 0.5363)
T800_SHORT_EPISODE_MOTION_DIR = "data/npz/traj_eng_50hz"
T800_COMPLETE_EPISODE_MOTION_DIR = "data/npz/traj_eng_50hz_episode_complete"
T800_IMPROVED_MOTION_DIR = "data/npz/traj_eng_50hz_improved"
T800_GET_UP_MOTION_DIR = "data/npz/t800_get_up"
T800_GET_UP_DEFAULT_MOTION = "getup1_t800"
T800_GET_UP_SMOOTH_MOTION = "faint_prone_getup_03_t800"
T800_GET_UP_TRANSITION_PHASE = (0.24, 0.76)
T800_GET_UP_FINAL_STANDING_PHASE = (0.90, 1.0)

T800_LEFT_PUNCH_BODY_NAMES = ["LINK_ELBOW_YAW_L"]
T800_RIGHT_PUNCH_BODY_NAMES = ["LINK_ELBOW_YAW_R"]
T800_LEFT_WRIST_END_OFFSET_B = (0.029436, 0.0124855, -0.13222151)
T800_RIGHT_WRIST_END_OFFSET_B = (0.029436, -0.0124855, -0.13222151)
T800_LEFT_HOOK_JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]
T800_ROUNDHOUSE_BODY_NAMES = ["LINK_ANKLE_ROLL_R"]
T800_ROUNDHOUSE_JOINT_NAMES = [
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
]
# Phase windows are normalized over the derived motion including terminal hold.
# Left hook source frames 9-50 map to 155 total frames; the locally retimed
# roundhouse source frames 31-56/48-66 map to 218 total frames.
T800_LEFT_HOOK_STRIKE_PHASE = (0.0584, 0.3247)
T800_ROUNDHOUSE_KICK_PHASE = (0.1429, 0.2765)
T800_ROUNDHOUSE_EXTENSION_PHASE = (0.2212, 0.3456)
# The retimed reference touches down around frame 85 and is upright again by
# frame 90. Keep the dynamic kick rewards unchanged, then supervise leg
# retraction and standing stability only after that transient.
T800_ROUNDHOUSE_RETRACTION_PHASE = (0.34, 0.46)
T800_ROUNDHOUSE_RECOVERY_PHASE = (0.41, 1.0)
T800_ROUNDHOUSE_FINAL_STANDING_PHASE = (0.90, 1.0)

# Target centers are measured from the same canonical/derived motion files
# used by the best tracking baselines.  The center is 6 cm behind the first
# dominant endpoint peak, so the endpoint reaches the compliant pad without
# forcing the robot to over-extend.
T800_TARGET_STRIKE_SPECS = {
    "straight_punch_L": {
        "motion_name": "straight_punch_L_terminal_hold_0p5s",
        "motion_dir": T800_IMPROVED_MOTION_DIR,
        "episode_length_s": 1.70,
        # Calibrated from the wrist-end collision sphere center rather than
        # the LINK_ELBOW_YAW_L frame origin.
        "target_position": (0.680, 0.105, 1.295),
        "strike_body_name": "LINK_WRIST_END_L",
        "contact_frames": (9, 13),
        "recovery_frame": 14,
        "frame_count": 85,
        "target_yaw": 0.0,
    },
    "straight_punch_R": {
        "motion_name": "straight_punch_R_terminal_hold_0p5s",
        "motion_dir": T800_IMPROVED_MOTION_DIR,
        "episode_length_s": 1.50,
        # Right wrist collision-sphere center at the extension plateau is
        # approximately (0.869, -0.196, 1.297); move the wall inward by 6 cm.
        "target_position": (0.810, -0.182, 1.297),
        "strike_body_name": "LINK_WRIST_END_R",
        "contact_frames": (15, 19),
        "recovery_frame": 20,
        "frame_count": 75,
        "target_yaw": 0.0,
    },
    "left_hook": {
        "motion_name": "left_hook_001_terminal_hold_0p5s",
        "motion_dir": T800_IMPROVED_MOTION_DIR,
        "episode_length_s": 3.10,
        # Calibrated from the wrist-end collision sphere at the hook peak.
        "target_position": (0.535, -0.372, 1.205),
        "strike_body_name": "LINK_WRIST_END_L",
        "contact_frames": (34, 38),
        "recovery_frame": 39,
        "frame_count": 155,
        "target_yaw": -2.52,
    },
    "left_front_kick": {
        "motion_name": "left_front_kick_002_terminal_hold_1s",
        "motion_dir": T800_COMPLETE_EPISODE_MOTION_DIR,
        "episode_length_s": 3.58,
        "target_position": (1.821, 0.034, 1.135),
        "strike_body_name": "LINK_ANKLE_ROLL_L",
        "contact_frames": (44, 48),
        "recovery_frame": 49,
        "frame_count": 179,
        "target_yaw": 0.0,
    },
    "roundhouse_kick": {
        "motion_name": "roundhouse_kick_001_retimed_terminal_hold_1s",
        "motion_dir": T800_IMPROVED_MOTION_DIR,
        "episode_length_s": 4.36,
        # Calibrated from the ankle collision box center at the kick peak.
        "target_position": (1.330, -0.200, 1.570),
        "strike_body_name": "LINK_ANKLE_ROLL_R",
        "contact_frames": (57, 61),
        "recovery_frame": 62,
        "frame_count": 218,
        "target_yaw": -1.33,
    },
}


def _make_body_position_reward(weight: float, std: float, body_names: list[str]) -> RewTerm:
    return RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "body_names": body_names,
        },
    )


def _make_body_linear_velocity_reward(weight: float, std: float, body_names: list[str]) -> RewTerm:
    return RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "body_names": body_names,
        },
    )


def _make_phase_body_linear_velocity_reward(
    weight: float,
    std: float,
    body_names: list[str],
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_global_body_linear_velocity_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "body_names": body_names,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_body_angular_velocity_reward(
    weight: float,
    std: float,
    body_names: list[str],
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_global_body_angular_velocity_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "body_names": body_names,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_joint_position_reward(
    weight: float,
    std: float,
    joint_names: list[str],
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_joint_position_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "joint_names": joint_names,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_body_position_reward(
    weight: float,
    std: float,
    body_names: list[str],
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_relative_body_position_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "body_names": body_names,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_anchor_height_reward(
    weight: float,
    std: float,
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_anchor_height_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_anchor_position_reward(
    weight: float,
    std: float,
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_global_anchor_position_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_anchor_orientation_reward(
    weight: float,
    std: float,
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_global_anchor_orientation_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_anchor_vertical_velocity_reward(
    weight: float,
    std: float,
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_motion_anchor_vertical_velocity_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_wrist_end_position_reward(
    weight: float,
    std: float,
    reference_body_name: str,
    wrist_body_name: str,
    target_offset_b: tuple[float, float, float],
) -> RewTerm:
    return RewTerm(
        func=mdp.motion_fixed_offset_body_position_error_exp,
        weight=weight,
        params={
            "command_name": "motion",
            "std": std,
            "reference_body_name": reference_body_name,
            "target_offset_b": target_offset_b,
            "asset_cfg": SceneEntityCfg("robot", body_names=[wrist_body_name], preserve_order=True),
        },
    )


def _make_phase_wrist_end_position_reward(
    weight: float,
    std: float,
    reference_body_name: str,
    wrist_body_name: str,
    target_offset_b: tuple[float, float, float],
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    reward = _make_wrist_end_position_reward(
        weight=weight,
        std=std,
        reference_body_name=reference_body_name,
        wrist_body_name=wrist_body_name,
        target_offset_b=target_offset_b,
    )
    reward.func = mdp.phase_motion_fixed_offset_body_position_error_exp
    reward.params["phase_start"] = phase_start
    reward.params["phase_end"] = phase_end
    return reward


def _make_end_effector_position_reward(weight: float, std: float) -> RewTerm:
    return _make_body_position_reward(
        weight=weight,
        std=std,
        body_names=T800_TRACKING_END_EFFECTOR_BODY_NAMES,
    )


def _make_end_effector_linear_velocity_reward(weight: float, std: float) -> RewTerm:
    return _make_body_linear_velocity_reward(
        weight=weight,
        std=std,
        body_names=T800_TRACKING_END_EFFECTOR_BODY_NAMES,
    )


def _add_540_kick_position_reward(env_cfg, weight: float, std: float, reward_name: str = "kick_right_pos"):
    setattr(
        env_cfg.rewards,
        reward_name,
        _make_body_position_reward(weight=weight, std=std, body_names=T800_540_KICK_BODY_NAMES),
    )


def _add_540_kick_phase_velocity_reward(
    env_cfg,
    weight: float,
    std: float,
    phase: tuple[float, float],
    reward_name: str = "kick_right_lin_vel",
):
    setattr(
        env_cfg.rewards,
        reward_name,
        _make_phase_body_linear_velocity_reward(
            weight=weight,
            std=std,
            body_names=T800_540_KICK_BODY_NAMES,
            phase_start=phase[0],
            phase_end=phase[1],
        ),
    )


def _add_540_kick_joint_phase_position_reward(
    env_cfg,
    weight: float,
    std: float,
    phase: tuple[float, float],
    reward_name: str = "kick_right_joint_phase_pos",
):
    setattr(
        env_cfg.rewards,
        reward_name,
        _make_phase_joint_position_reward(
            weight=weight,
            std=std,
            joint_names=T800_540_KICK_JOINT_NAMES,
            phase_start=phase[0],
            phase_end=phase[1],
        ),
    )


def _make_support_foot_com_reward(weight: float = 0.75, std: float = 0.18, force_threshold: float = 10.0) -> RewTerm:
    return RewTerm(
        func=mdp.support_foot_com_distance_reward,
        weight=weight,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=T800_SUPPORT_FOOT_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_SUPPORT_FOOT_BODY_NAMES, preserve_order=True),
            "force_threshold": force_threshold,
            "std": std,
        },
    )


def _make_feet_slip_reward(weight: float = -0.02, force_threshold: float = 10.0) -> RewTerm:
    return RewTerm(
        func=mdp.feet_slip_penalty,
        weight=weight,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_SUPPORT_FOOT_BODY_NAMES, preserve_order=True),
            "force_threshold": force_threshold,
        },
    )


def _make_phase_feet_slip_reward(
    weight: float,
    force_threshold: float,
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_feet_slip_penalty,
        weight=weight,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=T800_SUPPORT_FOOT_BODY_NAMES, preserve_order=True
            ),
            "force_threshold": force_threshold,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _make_phase_support_foot_com_reward(
    weight: float,
    std: float,
    force_threshold: float,
    phase_start: float,
    phase_end: float,
) -> RewTerm:
    return RewTerm(
        func=mdp.phase_support_foot_com_distance_reward,
        weight=weight,
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", body_names=T800_SUPPORT_FOOT_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=T800_SUPPORT_FOOT_BODY_NAMES, preserve_order=True
            ),
            "force_threshold": force_threshold,
            "std": std,
            "phase_start": phase_start,
            "phase_end": phase_end,
        },
    )


def _disable_early_terminations(env_cfg):
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    for term_name in ("fall_height", "fall_orientation"):
        if hasattr(env_cfg.terminations, term_name):
            setattr(env_cfg.terminations, term_name, None)


@configclass
class T800FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # The training entry point replaces this fallback with the selected NPZ's
        # native duration before creating the simulator.
        self.episode_length_s = 10.0
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = t800_mdp.ResidualRefJointPositionActionCfg(
            asset_name="robot",
            joint_names=t800_mdp.T800_DFS_JOINT_NAMES,
            command_name="motion",
            preserve_order=True,
        )
        self.actions.joint_pos.scale = T800_ACTION_SCALE
        dfs_joint_asset_cfg = SceneEntityCfg("robot", joint_names=t800_mdp.T800_DFS_JOINT_NAMES, preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.motion_joint_names = t800_mdp.T800_DFS_JOINT_NAMES
        self.commands.motion.motion_body_names = t800_mdp.T800_MOTION_BODY_NAMES
        # Keep the source clip native-length. During training, reset from
        # adaptively sampled phases and resample at the motion end instead of
        # manufacturing a cyclic bridge for non-cyclic strikes.
        self.commands.motion.min_traj_duration = None
        self.commands.motion.bridge_frames = 0
        self.commands.motion.sampling_mode = "adaptive"
        self.commands.motion.play_from_start = False
        self.commands.motion.resample_at_motion_end = True
        self.commands.motion.pd_stand_reset_ratio = 0.0
        self.commands.motion.feet_body_names = T800_SUPPORT_FOOT_BODY_NAMES
        self.commands.motion.body_names = [
            "LINK_BASE",
            "LINK_HIP_ROLL_L",
            "LINK_KNEE_PITCH_L",
            "LINK_ANKLE_ROLL_L",
            "LINK_HIP_ROLL_R",
            "LINK_KNEE_PITCH_R",
            "LINK_ANKLE_ROLL_R",
            "LINK_TORSO_YAW",
            "LINK_SHOULDER_PITCH_L",
            "LINK_ELBOW_PITCH_L",
            "LINK_ELBOW_YAW_L",
            "LINK_SHOULDER_PITCH_R",
            "LINK_ELBOW_PITCH_R",
            "LINK_ELBOW_YAW_R",
            "LINK_HEAD_PITCH",
            "LINK_HEAD_YAW",
        ]
        self.events.base_com.params["asset_cfg"].body_names = "LINK_BASE"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_ELBOW_YAW_L$)(?!LINK_ELBOW_YAW_R$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
        ]
        # BeyondMimic-style tracking early reset. Restrict position checks to
        # height so benign planar drift does not terminate strikes and kicks.
        self.terminations.anchor_pos.func = mdp.bad_anchor_pos_z_only
        self.terminations.anchor_pos.params["threshold"] = 0.25
        self.terminations.anchor_ori.params["threshold"] = 0.8
        self.terminations.ee_body_pos.func = mdp.bad_motion_body_pos_z_only
        self.terminations.ee_body_pos.params["threshold"] = 0.25

        self.viewer.eye = (-1.5, -1.5, 1.5)


def _configure_short_episode_baseline(
    env_cfg,
    motion_name: str,
    episode_length_s: float,
    motion_dir: str = T800_SHORT_EPISODE_MOTION_DIR,
) -> None:
    """Reproduce the successful j7236 native-clip training distribution."""
    motion_cfg = env_cfg.commands.motion
    motion_cfg.motion_file = f"{motion_dir}/{motion_name}.npz"
    motion_cfg.min_traj_duration = None
    motion_cfg.bridge_frames = 0
    motion_cfg.sampling_mode = "adaptive"
    motion_cfg.phase_sampling_windows = []
    motion_cfg.pd_stand_reset_ratio = 0.0
    motion_cfg.reset_preroll_frames = 0
    motion_cfg.play_from_start = True
    motion_cfg.resample_at_motion_end = False
    env_cfg.episode_length_s = episode_length_s

    # j7236 did not terminate on tracking error. It reset only at the native
    # clip horizon or when the base height/orientation indicated a fall.
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.fall_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
    env_cfg.terminations.fall_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    env_cfg.events.push_robot.interval_range_s = (5.0, 10.0)


def _configure_get_up_tracking(env_cfg, motion_name: str = T800_GET_UP_DEFAULT_MOTION) -> None:
    """Configure deterministic full-clip tracking for a lying-to-standing motion."""
    motion_cfg = env_cfg.commands.motion
    motion_cfg.motion_file = f"{T800_GET_UP_MOTION_DIR}/{motion_name}.npz"
    motion_cfg.min_traj_duration = None
    motion_cfg.bridge_frames = 0
    motion_cfg.sampling_mode = "start"
    motion_cfg.phase_sampling_windows = []
    motion_cfg.pd_stand_reset_ratio = 0.0
    motion_cfg.reset_preroll_frames = 0
    motion_cfg.play_from_start = True
    motion_cfg.resample_at_motion_end = False

    # The clips intentionally begin on the floor (root z is roughly 0.09 m in
    # getup1), so ordinary standing/fall thresholds would terminate the valid
    # reference before the robot has a chance to get up.
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    for term_name in ("fall_height", "fall_orientation"):
        if hasattr(env_cfg.terminations, term_name):
            setattr(env_cfg.terminations, term_name, None)

    # Start each reset from the demonstrated state.  This is deliberately a
    # clean tracking baseline; robustness randomization can be added after the
    # reference get-up is learned.
    zero_pose_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    motion_cfg.pose_range = zero_pose_range
    motion_cfg.velocity_range = dict(zero_pose_range)
    motion_cfg.joint_position_range = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.rewards.undesired_contacts = None


@configclass
class T800FlatGetUpEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_get_up_tracking(self)


@configclass
class T800FlatGetUpSmoothEnvCfg(T800FlatEnvCfg):
    """Track the continuous T800 get-up clip with transition-focused shaping."""

    def __post_init__(self):
        super().__post_init__()
        _configure_get_up_tracking(self, T800_GET_UP_SMOOTH_MOTION)

        # The smooth clip contains two linked lift phases.  Extra dense
        # joint/body terms make the policy pay attention to the transition,
        # while the ordinary tracking stack still controls the full motion.
        transition_start, transition_end = T800_GET_UP_TRANSITION_PHASE
        final_start, final_end = T800_GET_UP_FINAL_STANDING_PHASE
        # Keep global anchor tracking active across the clip, while putting
        # most of the height shaping pressure on the actual lift trajectory.
        # The final-standing terms still add another 3.0 from 90% to the end.
        self.rewards.motion_global_anchor_pos.weight = 2.0
        self.rewards.motion_global_anchor_ori.weight = 2.0
        self.rewards.getup_anchor_height = _make_phase_anchor_height_reward(
            weight=10.0,
            std=0.18,
            phase_start=0.20,
            phase_end=1.0,
        )
        self.rewards.getup_final_anchor_pos = _make_phase_anchor_position_reward(
            weight=3.0,
            std=0.30,
            phase_start=final_start,
            phase_end=final_end,
        )
        self.rewards.getup_final_anchor_ori = _make_phase_anchor_orientation_reward(
            weight=3.0,
            std=0.40,
            phase_start=final_start,
            phase_end=final_end,
        )
        self.rewards.getup_final_anchor_height = _make_phase_anchor_height_reward(
            weight=3.0,
            std=0.18,
            phase_start=final_start,
            phase_end=final_end,
        )
        self.rewards.getup_anchor_vertical_velocity = _make_phase_anchor_vertical_velocity_reward(
            weight=0.35,
            std=0.75,
            phase_start=transition_start,
            phase_end=transition_end,
        )
        self.rewards.getup_transition_joint_pos = _make_phase_joint_position_reward(
            weight=3.0,
            std=0.30,
            joint_names=t800_mdp.T800_DFS_JOINT_NAMES,
            phase_start=transition_start,
            phase_end=transition_end,
        )
        self.rewards.getup_transition_body_pos = _make_phase_body_position_reward(
            weight=1.5,
            std=0.26,
            body_names=[
                "LINK_BASE",
                "LINK_HIP_PITCH_L",
                "LINK_KNEE_PITCH_L",
                "LINK_ANKLE_ROLL_L",
                "LINK_HIP_PITCH_R",
                "LINK_KNEE_PITCH_R",
                "LINK_ANKLE_ROLL_R",
                "LINK_SHOULDER_ROLL_L",
                "LINK_SHOULDER_ROLL_R",
            ],
            phase_start=transition_start,
            phase_end=transition_end,
        )


def _apply_punch_improvements(
    env_cfg,
    hand_body_name: str,
    wrist_body_name: str,
    wrist_offset_b: tuple[float, float, float],
) -> None:
    env_cfg.rewards.punch_end_pos = _make_wrist_end_position_reward(
        weight=1.2,
        std=0.18,
        reference_body_name=hand_body_name,
        wrist_body_name=wrist_body_name,
        target_offset_b=wrist_offset_b,
    )
    env_cfg.rewards.punch_end_lin_vel = _make_body_linear_velocity_reward(
        weight=0.7,
        std=0.35,
        body_names=[hand_body_name],
    )
    env_cfg.rewards.foot_slip = _make_feet_slip_reward(weight=-0.02, force_threshold=10.0)
    env_cfg.rewards.support_foot_com = _make_support_foot_com_reward(weight=0.4, std=0.18, force_threshold=10.0)


def _apply_hook_improvements(env_cfg) -> None:
    env_cfg.rewards.hook_end_pos = _make_phase_wrist_end_position_reward(
        weight=1.5,
        std=0.18,
        reference_body_name=T800_LEFT_PUNCH_BODY_NAMES[0],
        wrist_body_name="LINK_WRIST_END_L",
        target_offset_b=T800_LEFT_WRIST_END_OFFSET_B,
        phase_start=T800_LEFT_HOOK_STRIKE_PHASE[0],
        phase_end=T800_LEFT_HOOK_STRIKE_PHASE[1],
    )
    env_cfg.rewards.hook_end_lin_vel = _make_phase_body_linear_velocity_reward(
        weight=0.8,
        std=0.40,
        body_names=T800_LEFT_PUNCH_BODY_NAMES,
        phase_start=T800_LEFT_HOOK_STRIKE_PHASE[0],
        phase_end=T800_LEFT_HOOK_STRIKE_PHASE[1],
    )
    env_cfg.rewards.hook_joints_peak = _make_phase_joint_position_reward(
        weight=0.8,
        std=0.20,
        joint_names=T800_LEFT_HOOK_JOINT_NAMES,
        phase_start=T800_LEFT_HOOK_STRIKE_PHASE[0],
        phase_end=T800_LEFT_HOOK_STRIKE_PHASE[1],
    )
    env_cfg.rewards.foot_slip = _make_feet_slip_reward(weight=-0.02, force_threshold=10.0)
    env_cfg.rewards.support_foot_com = _make_support_foot_com_reward(weight=0.4, std=0.18, force_threshold=10.0)


def _apply_roundhouse_improvements(env_cfg) -> None:
    env_cfg.rewards.roundhouse_kick_pos = _make_phase_body_position_reward(
        weight=1.5,
        std=0.18,
        body_names=T800_ROUNDHOUSE_BODY_NAMES,
        phase_start=T800_ROUNDHOUSE_KICK_PHASE[0],
        phase_end=T800_ROUNDHOUSE_KICK_PHASE[1],
    )
    env_cfg.rewards.roundhouse_kick_vel = _make_phase_body_linear_velocity_reward(
        weight=0.8,
        std=0.55,
        body_names=T800_ROUNDHOUSE_BODY_NAMES,
        phase_start=T800_ROUNDHOUSE_KICK_PHASE[0],
        phase_end=T800_ROUNDHOUSE_KICK_PHASE[1],
    )
    env_cfg.rewards.roundhouse_kick_joints = _make_phase_joint_position_reward(
        weight=1.0,
        std=0.16,
        joint_names=T800_ROUNDHOUSE_JOINT_NAMES,
        phase_start=T800_ROUNDHOUSE_EXTENSION_PHASE[0],
        phase_end=T800_ROUNDHOUSE_EXTENSION_PHASE[1],
    )


def _apply_roundhouse_stable_recovery_improvements(env_cfg) -> None:
    """Add landing/recovery shaping without changing the kick transient."""
    retract_start, retract_end = T800_ROUNDHOUSE_RETRACTION_PHASE
    recovery_start, recovery_end = T800_ROUNDHOUSE_RECOVERY_PHASE
    final_start, final_end = T800_ROUNDHOUSE_FINAL_STANDING_PHASE

    # Keep the whole clip anchored to the reference, then add an identical
    # phase-gated copy during the terminal standing hold.  This gives the
    # policy a stronger incentive to finish at the demonstrated pose without
    # changing the transient kick objective.
    env_cfg.rewards.motion_global_anchor_pos.weight = 1.0
    env_cfg.rewards.motion_global_anchor_ori.weight = 1.0
    env_cfg.rewards.roundhouse_final_anchor_pos = _make_phase_anchor_position_reward(
        weight=1.0,
        std=0.3,
        phase_start=final_start,
        phase_end=final_end,
    )
    env_cfg.rewards.roundhouse_final_anchor_ori = _make_phase_anchor_orientation_reward(
        weight=1.0,
        std=0.4,
        phase_start=final_start,
        phase_end=final_end,
    )

    # The right foot is back on the floor shortly after the strike. This late
    # joint term teaches the policy to complete the retraction instead of
    # stopping at the high extension pose.
    env_cfg.rewards.roundhouse_recovery_joints = _make_phase_joint_position_reward(
        weight=0.6,
        std=0.24,
        joint_names=T800_ROUNDHOUSE_JOINT_NAMES,
        phase_start=retract_start,
        phase_end=retract_end,
    )
    # The reference is upright again at this phase; make the height and base
    # spin objective explicit while leaving the large kick tilt unpenalized.
    env_cfg.rewards.roundhouse_recovery_height = _make_phase_anchor_height_reward(
        weight=0.45,
        std=0.12,
        phase_start=recovery_start,
        phase_end=recovery_end,
    )
    env_cfg.rewards.roundhouse_recovery_ang_vel = _make_phase_body_angular_velocity_reward(
        weight=0.35,
        std=0.80,
        body_names=["LINK_BASE"],
        phase_start=recovery_start,
        phase_end=recovery_end,
    )
    env_cfg.rewards.roundhouse_recovery_stability = RewTerm(
        func=mdp.phase_recovery_stability_reward,
        weight=0.35,
        params={
            "command_name": "motion",
            "phase_start": recovery_start,
            "phase_end": recovery_end,
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.65,
            "height_margin": 0.10,
            "angular_velocity_scale": 2.5,
        },
    )

    # Gate both support terms to the landing/recovery window: penalizing the
    # pivot foot during the kick would fight the demonstrated rotation.
    env_cfg.rewards.roundhouse_recovery_foot_slip = _make_phase_feet_slip_reward(
        weight=-2.0,
        force_threshold=10.0,
        phase_start=recovery_start,
        phase_end=recovery_end,
    )
    env_cfg.rewards.roundhouse_recovery_support_com = _make_phase_support_foot_com_reward(
        weight=1.0,
        std=0.18,
        force_threshold=10.0,
        phase_start=recovery_start,
        phase_end=recovery_end,
    )

    # The terminal hold is the actual standing outcome.  Add one more copy of
    # each requested recovery term from 90% to the end so their effective
    # weights are doubled only after the kick has finished.
    env_cfg.rewards.roundhouse_final_foot_slip = _make_phase_feet_slip_reward(
        weight=-2.0,
        force_threshold=10.0,
        phase_start=final_start,
        phase_end=final_end,
    )
    env_cfg.rewards.roundhouse_final_support_com = _make_phase_support_foot_com_reward(
        weight=1.0,
        std=0.18,
        force_threshold=10.0,
        phase_start=final_start,
        phase_end=final_end,
    )


def _keep_height_fall_only(env_cfg) -> None:
    """Avoid ending valid strike transients before the policy can learn recovery."""
    env_cfg.terminations.fall_orientation = None


def _apply_stable_ground_randomization(env_cfg) -> None:
    """Use a modest friction range for the first tracking-quality baseline."""
    material_params = env_cfg.events.physics_material.params
    material_params["static_friction_range"] = (0.7, 1.2)
    material_params["dynamic_friction_range"] = (0.6, 1.0)
    material_params["restitution_range"] = (0.0, 0.1)


def _apply_target_high_friction_randomization(env_cfg) -> None:
    """Randomize target-task ground contact over a high-friction range.

    The terrain material remains at static/dynamic friction 1.0.  The existing
    startup event randomizes the robot material, which yields the effective
    ground-contact range used by these target tasks under the terrain's
    multiply combine mode.
    """
    material_params = env_cfg.events.physics_material.params
    material_params["static_friction_range"] = (1.0, 1.7)
    material_params["dynamic_friction_range"] = (1.0, 1.7)
    material_params["restitution_range"] = (0.0, 0.1)


def _phase_from_frame(frame: int, frame_count: int) -> float:
    return frame / float(max(frame_count - 1, 1))


def _make_padded_target_cfg(position: tuple[float, float, float], yaw: float) -> RigidObjectCfg:
    """Create a fixed, low-restitution compliant pad proxy.

    Isaac Lab 2.1's rigid-body compliant-contact material is used here rather
    than a hard ground-like collider.  A true FEM sponge would need a mesh and
    deformable-body asset; this proxy keeps the contact impulse bounded while
    remaining inexpensive enough for PPO vectorized training.  The pad is
    kinematic so it cannot be knocked away, while the dynamic robot still
    receives the PhysX contact impulse measured by ``target_contact``.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            # A thin, wide, tall wall gives the endpoint a generous hit area
            # without placing a thick obstacle in front of the robot.
            size=(0.18, 1.40, 1.60),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.55, 0.88),
                roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.30,
                dynamic_friction=0.22,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
                compliant_contact_stiffness=3500.0,
                compliant_contact_damping=160.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Keep the target fixed in world space.  Kinematic bodies still
                # participate in collision resolution and apply contact forces
                # to the dynamic robot; only the target's own motion is locked.
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=8.0,
                angular_damping=12.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=12.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position,
            rot=(cos(yaw / 2.0), 0.0, 0.0, sin(yaw / 2.0)),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


def _configure_target_strike(env_cfg, spec: dict) -> None:
    """Attach one fixed padded target to a native-length tracking task."""
    motion_cfg = env_cfg.commands.motion
    # The target coordinates are measured in the zero-phase motion frame.  A
    # deterministic start is required until target pose randomization is made
    # phase-aware; otherwise a random root yaw would move the target away from
    # the endpoint while the motion command still starts from frame zero.
    motion_cfg.pose_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    motion_cfg.velocity_range = {key: (0.0, 0.0) for key in VELOCITY_RANGE}

    env_cfg.scene.target = _make_padded_target_cfg(spec["target_position"], spec["target_yaw"])
    # Sense the hit on the robot's already-contact-enabled strike body.  This
    # avoids requiring a PhysX contact reporter on the padded proxy itself;
    # the filter still restricts the force to the target pad.
    env_cfg.scene.target_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{spec['strike_body_name']}",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=2.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Target"],
        debug_vis=False,
    )
    env_cfg.events.reset_target = EventTerm(
        func=mdp.reset_rigid_object_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("target")},
    )
    env_cfg.observations.critic.target_pos_b = ObsTerm(
        func=mdp.target_position_b,
        params={"asset_name": "target", "command_name": "motion"},
    )

    contact_start, contact_end = spec["contact_frames"]
    contact_phase = (_phase_from_frame(contact_start, spec["frame_count"]), _phase_from_frame(contact_end, spec["frame_count"]))
    recovery_phase = _phase_from_frame(spec["recovery_frame"], spec["frame_count"])
    env_cfg.rewards.target_contact = RewTerm(
        func=mdp.target_contact_force_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("target_contact"),
            "phase_start": contact_phase[0],
            "phase_end": contact_phase[1],
            "force_scale": 25.0,
            "force_threshold": 2.0,
        },
    )
    env_cfg.rewards.target_recovery = RewTerm(
        func=mdp.target_recovery_stability_reward,
        weight=0.20,
        params={
            "command_name": "motion",
            "phase_start": recovery_phase,
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.4,
            "height_margin": 0.08,
            "angular_velocity_scale": 3.0,
        },
    )


@configclass
class T800FlatBaselineStraightPunchLEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(self, "straight_punch_L", 1.20)


@configclass
class T800FlatBaselineStraightPunchREnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(self, "straight_punch_R", 1.00)


@configclass
class T800FlatBaselineLeftHookEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(self, "left_hook_001", 2.60)


@configclass
class T800FlatBaselineRearHookEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(self, "rear_hook_001", 3.54)


@configclass
class T800FlatBaselineLeftFrontKickEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(self, "left_front_kick_002", 2.58)


@configclass
class T800FlatBaselineRoundhouseKickEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(self, "roundhouse_kick_001", 3.18)


@configclass
class T800FlatBaselineLeftFrontKickCompleteEpisodeEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(
            self,
            "left_front_kick_002_terminal_hold_1s",
            3.58,
            motion_dir=T800_COMPLETE_EPISODE_MOTION_DIR,
        )


@configclass
class T800FlatBaselineLeftFrontKickCompleteEpisodeHeightFallOnlyEnvCfg(
    T800FlatBaselineLeftFrontKickCompleteEpisodeEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        # The previous start-from-zero runs collapsed at frame 44-45 because
        # orientation termination fired before PPO could observe the recovery.
        # Keep the physical height-based fall reset, but let the policy learn
        # through the kick's large transient torso tilt.
        self.terminations.fall_orientation = None


@configclass
class T800FlatBaselineRoundhouseKickCompleteEpisodeEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(
            self,
            "roundhouse_kick_001_terminal_hold_1s",
            4.18,
            motion_dir=T800_COMPLETE_EPISODE_MOTION_DIR,
        )


@configclass
class T800FlatImprovedStraightPunchLEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(
            self,
            "straight_punch_L_terminal_hold_0p5s",
            1.70,
            motion_dir=T800_IMPROVED_MOTION_DIR,
        )
        _apply_punch_improvements(
            self,
            hand_body_name="LINK_ELBOW_YAW_L",
            wrist_body_name="LINK_WRIST_END_L",
            wrist_offset_b=T800_LEFT_WRIST_END_OFFSET_B,
        )
        _apply_stable_ground_randomization(self)
        _keep_height_fall_only(self)


@configclass
class T800FlatImprovedStraightPunchREnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(
            self,
            "straight_punch_R_terminal_hold_0p5s",
            1.50,
            motion_dir=T800_IMPROVED_MOTION_DIR,
        )
        _apply_punch_improvements(
            self,
            hand_body_name="LINK_ELBOW_YAW_R",
            wrist_body_name="LINK_WRIST_END_R",
            wrist_offset_b=T800_RIGHT_WRIST_END_OFFSET_B,
        )
        _apply_stable_ground_randomization(self)
        _keep_height_fall_only(self)


@configclass
class T800FlatImprovedLeftHookEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(
            self,
            "left_hook_001_terminal_hold_0p5s",
            3.10,
            motion_dir=T800_IMPROVED_MOTION_DIR,
        )
        _apply_hook_improvements(self)
        _apply_stable_ground_randomization(self)
        _keep_height_fall_only(self)


@configclass
class T800FlatImprovedRoundhouseKickEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_short_episode_baseline(
            self,
            "roundhouse_kick_001_retimed_terminal_hold_1s",
            4.36,
            motion_dir=T800_IMPROVED_MOTION_DIR,
        )
        _apply_roundhouse_improvements(self)
        _keep_height_fall_only(self)


@configclass
class T800FlatImprovedRoundhouseKickStableRecoveryEnvCfg(T800FlatImprovedRoundhouseKickEnvCfg):
    """Fine-tune the improved roundhouse policy for post-landing stability."""

    def __post_init__(self):
        super().__post_init__()
        _apply_roundhouse_stable_recovery_improvements(self)


@configclass
class T800FlatTargetStraightPunchLEnvCfg(T800FlatEnvCfg):
    """Left straight-punch tracking with a compliant external target."""

    def __post_init__(self):
        super().__post_init__()
        spec = T800_TARGET_STRIKE_SPECS["straight_punch_L"]
        _configure_short_episode_baseline(
            self,
            spec["motion_name"],
            spec["episode_length_s"],
            motion_dir=spec["motion_dir"],
        )
        _apply_punch_improvements(
            self,
            hand_body_name="LINK_ELBOW_YAW_L",
            wrist_body_name="LINK_WRIST_END_L",
            wrist_offset_b=T800_LEFT_WRIST_END_OFFSET_B,
        )
        _apply_stable_ground_randomization(self)
        _keep_height_fall_only(self)
        _configure_target_strike(self, spec)


@configclass
class T800FlatTargetStraightPunchREnvCfg(T800FlatEnvCfg):
    """Right straight-punch tracking with a compliant external target."""

    def __post_init__(self):
        super().__post_init__()
        spec = T800_TARGET_STRIKE_SPECS["straight_punch_R"]
        _configure_short_episode_baseline(
            self,
            spec["motion_name"],
            spec["episode_length_s"],
            motion_dir=spec["motion_dir"],
        )
        _apply_punch_improvements(
            self,
            hand_body_name="LINK_ELBOW_YAW_R",
            wrist_body_name="LINK_WRIST_END_R",
            wrist_offset_b=T800_RIGHT_WRIST_END_OFFSET_B,
        )
        _apply_stable_ground_randomization(self)
        _keep_height_fall_only(self)
        _configure_target_strike(self, spec)


@configclass
class T800FlatTargetLeftHookEnvCfg(T800FlatEnvCfg):
    """Left hook tracking with a compliant external target."""

    def __post_init__(self):
        super().__post_init__()
        spec = T800_TARGET_STRIKE_SPECS["left_hook"]
        _configure_short_episode_baseline(
            self,
            spec["motion_name"],
            spec["episode_length_s"],
            motion_dir=spec["motion_dir"],
        )
        _apply_hook_improvements(self)
        _apply_stable_ground_randomization(self)
        _keep_height_fall_only(self)
        _configure_target_strike(self, spec)


@configclass
class T800FlatTargetLeftFrontKickEnvCfg(T800FlatEnvCfg):
    """Left front-kick tracking with a compliant external target."""

    def __post_init__(self):
        super().__post_init__()
        spec = T800_TARGET_STRIKE_SPECS["left_front_kick"]
        _configure_short_episode_baseline(
            self,
            spec["motion_name"],
            spec["episode_length_s"],
            motion_dir=spec["motion_dir"],
        )
        # The complete front-kick clip contains a large but valid transient
        # tilt.  Keep only the physical height fall reset, as in the best
        # tracking-only checkpoint.
        self.terminations.fall_orientation = None
        _apply_target_high_friction_randomization(self)
        _configure_target_strike(self, spec)


@configclass
class T800FlatTargetRoundhouseKickEnvCfg(T800FlatEnvCfg):
    """Retimed roundhouse tracking with a compliant external target."""

    def __post_init__(self):
        super().__post_init__()
        spec = T800_TARGET_STRIKE_SPECS["roundhouse_kick"]
        _configure_short_episode_baseline(
            self,
            spec["motion_name"],
            spec["episode_length_s"],
            motion_dir=spec["motion_dir"],
        )
        _apply_roundhouse_improvements(self)
        _apply_target_high_friction_randomization(self)
        _keep_height_fall_only(self)
        _configure_target_strike(self, spec)


@configclass
class T800FlatWoStateEstimationEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class T800Flat540Huixuanti1EnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 290 frames consumed at the 50 Hz policy rate: keep one 540 cut per episode.
        self.episode_length_s = 5.8

        self.commands.motion.min_traj_duration = None
        self.commands.motion.bridge_frames = 0
        self.commands.motion.pd_stand_reset_ratio = 0.0

        self.rewards.huixuanti_end_effector_pos = _make_end_effector_position_reward(weight=1.5, std=0.25)
        self.rewards.huixuanti_end_effector_lin_vel = _make_end_effector_linear_velocity_reward(weight=1.0, std=0.7)

        self.viewer.eye = (-2.5, -2.5, 1.5)


@configclass
class T800Flat540Huixuanti1Start0EnvCfg(T800Flat540Huixuanti1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play_from_start = True
        self.commands.motion.resample_at_motion_end = False
        self.commands.motion.pd_stand_reset_ratio = 0.0


@configclass
class T800Flat540Huixuanti1NoEarlyTerminationsEnvCfg(T800Flat540Huixuanti1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _disable_early_terminations(self)


@configclass
class T800Flat540Huixuanti1OrigEpisodeEnvCfg(T800Flat540Huixuanti1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0


@configclass
class T800Flat540Huixuanti1KickVelKickPosEnvCfg(T800Flat540Huixuanti1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_540_kick_phase_velocity_reward(self, weight=1.0, std=0.7, phase=T800_540_KICK_PHASE)
        _add_540_kick_position_reward(self, weight=0.6, std=0.22)


@configclass
class T800Flat540Huixuanti1KickJointLateEnvCfg(T800Flat540Huixuanti1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_540_kick_joint_phase_position_reward(self, weight=0.55, std=0.22, phase=T800_540_KICK_PHASE_LATE)


@configclass
class T800FlatZhiquanEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 666 frames consumed at the 50 Hz policy rate: keep one zhiquan cut per episode.
        self.episode_length_s = 13.32

        self.commands.motion.min_traj_duration = None
        self.commands.motion.bridge_frames = 0
        self.commands.motion.pd_stand_reset_ratio = 0.0

        self.rewards.zhiquan_end_effector_pos = _make_end_effector_position_reward(weight=2.5, std=0.15)
        self.rewards.zhiquan_end_effector_lin_vel = _make_end_effector_linear_velocity_reward(weight=1.5, std=0.4)
        self.rewards.support_foot_com = _make_support_foot_com_reward()


@configclass
class T800FlatZhiquanOrigEpisodeEnvCfg(T800FlatZhiquanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 23.0


@configclass
class T800FlatZhiquanBridgeEnvCfg(T800FlatZhiquanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 24.0
        self.commands.motion.min_traj_duration = self.episode_length_s
        self.commands.motion.bridge_frames = 20


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
