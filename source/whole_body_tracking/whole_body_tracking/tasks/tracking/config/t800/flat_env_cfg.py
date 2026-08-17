from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from . import t800_mdp
from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


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


def _keep_height_fall_only(env_cfg) -> None:
    """Avoid ending valid strike transients before the policy can learn recovery."""
    env_cfg.terminations.fall_orientation = None


def _apply_stable_ground_randomization(env_cfg) -> None:
    """Use a modest friction range for the first tracking-quality baseline."""
    material_params = env_cfg.events.physics_material.params
    material_params["static_friction_range"] = (0.7, 1.2)
    material_params["dynamic_friction_range"] = (0.6, 1.0)
    material_params["restitution_range"] = (0.0, 0.1)


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
