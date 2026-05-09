from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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


@configclass
class T800FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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
        self.commands.motion.min_traj_duration = self.episode_length_s
        self.commands.motion.bridge_frames = 20
        self.commands.motion.pd_stand_reset_ratio = 0.2
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

        self.viewer.eye = (-1.5, -1.5, 1.5)

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
        self.commands.motion.pd_stand_reset_ratio = 0.2

        self.rewards.huixuanti_end_effector_pos = _make_end_effector_position_reward(weight=1.5, std=0.25)
        self.rewards.huixuanti_end_effector_lin_vel = _make_end_effector_linear_velocity_reward(weight=1.0, std=0.7)

        self.terminations.anchor_pos.params["threshold"] = 0.6
        self.terminations.anchor_ori.params["threshold"] = 1.6
        self.terminations.ee_body_pos = None

        self.viewer.eye = (-2.5, -2.5, 1.5)


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
        self.commands.motion.pd_stand_reset_ratio = 0.2

        self.rewards.zhiquan_end_effector_pos = _make_end_effector_position_reward(weight=2.5, std=0.15)
        self.rewards.zhiquan_end_effector_lin_vel = _make_end_effector_linear_velocity_reward(weight=1.5, std=0.4)
        self.rewards.support_foot_com = _make_support_foot_com_reward()

        self.terminations.anchor_pos.params["threshold"] = 0.4
        self.terminations.anchor_ori.params["threshold"] = 1.0
        self.terminations.ee_body_pos.params["threshold"] = 0.4


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
