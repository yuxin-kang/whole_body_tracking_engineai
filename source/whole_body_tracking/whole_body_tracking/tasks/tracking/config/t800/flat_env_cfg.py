from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from . import t800_mdp
from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


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
        self.episode_length_s = 10

        self.commands.motion.min_traj_duration = self.episode_length_s
        self.commands.motion.bridge_frames = 20
        self.commands.motion.pd_stand_reset_ratio = 0.2

        self.rewards.huixuanti_end_effector_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": 0.25,
                "body_names": [
                    "LINK_ANKLE_ROLL_L",
                    "LINK_ANKLE_ROLL_R",
                    "LINK_ELBOW_YAW_L",
                    "LINK_ELBOW_YAW_R",
                ],
            },
        )
        self.rewards.huixuanti_end_effector_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.7,
                "body_names": [
                    "LINK_ANKLE_ROLL_L",
                    "LINK_ANKLE_ROLL_R",
                    "LINK_ELBOW_YAW_L",
                    "LINK_ELBOW_YAW_R",
                ],
            },
        )
        self.rewards.support_foot_com = RewTerm(
            func=mdp.support_foot_com_distance_reward,
            weight=0.75,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
                    preserve_order=True,
                ),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
                    preserve_order=True,
                ),
                "force_threshold": 10.0,
                "std": 0.18,
            },
        )

        self.terminations.anchor_pos.params["threshold"] = 0.6
        self.terminations.anchor_ori.params["threshold"] = 1.6
        self.terminations.ee_body_pos = None


@configclass
class T800FlatZhiquanEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 23.0

        self.commands.motion.min_traj_duration = None
        self.commands.motion.bridge_frames = 0
        self.commands.motion.pd_stand_reset_ratio = 0.2

        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.zhiquan_end_effector_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=2.5,
            params={
                "command_name": "motion",
                "std": 0.15,
                "body_names": [
                    "LINK_ANKLE_ROLL_L",
                    "LINK_ANKLE_ROLL_R",
                    "LINK_ELBOW_YAW_L",
                    "LINK_ELBOW_YAW_R",
                ],
            },
        )
        self.rewards.zhiquan_end_effector_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": 0.4,
                "body_names": [
                    "LINK_ANKLE_ROLL_L",
                    "LINK_ANKLE_ROLL_R",
                    "LINK_ELBOW_YAW_L",
                    "LINK_ELBOW_YAW_R",
                ],
            },
        )

        self.terminations.anchor_pos.params["threshold"] = 0.4
        self.terminations.anchor_ori.params["threshold"] = 1.0
        self.terminations.ee_body_pos.params["threshold"] = 0.4


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
