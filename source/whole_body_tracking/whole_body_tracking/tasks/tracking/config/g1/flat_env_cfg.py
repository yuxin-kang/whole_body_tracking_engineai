from __future__ import annotations

import math
from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from . import g1_mdp
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CFG
from whole_body_tracking.tasks.tracking.config.t800 import t800_mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, TrackingEnvCfg


G1_1307_MOTION_FILE = str(Path("data/g1/1307.npz"))
G1_STANDING_INIT_FILE = str(Path("data/g1/robot_init_states_8192.pth"))

VELOCITY_RANGE_ADD = {
    "x": (-0.75, 0.75),
    "y": (-0.75, 0.75),
    "z": (-0.3, 0.3),
    "roll": (-0.78, 0.78),
    "pitch": (-0.78, 0.78),
    "yaw": (-1.17, 1.17),
}

G1_STAGE_III_ISAAC_EQUIVALENCE_NOTES = {
    "terrain": "Upstream Stage III uses mjlab terrain; this Isaac task keeps plane terrain until an equivalent terrain importer is added.",
    "standing_init": "Standing root states and dof positions are copied from upstream robot_init_states_8192.pth.",
    "reset_events": "Stage III adds Isaac reset_base/reset_robot_joints events to match the upstream reset curriculum.",
    "motion": "The 1307.npz arrays, G1 joint order, motion body order, and 14 tracked bodies are kept byte/order compatible.",
}


def _set_g1_common(env_cfg: TrackingEnvCfg):
    env_cfg.episode_length_s = 10.0
    env_cfg.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos = t800_mdp.ResidualRefJointPositionActionCfg(
        asset_name="robot",
        joint_names=g1_mdp.G1_MOTION_JOINT_NAMES,
        command_name="motion",
        preserve_order=True,
    )
    env_cfg.actions.joint_pos.scale = G1_ACTION_SCALE

    joint_asset_cfg = SceneEntityCfg("robot", joint_names=g1_mdp.G1_MOTION_JOINT_NAMES, preserve_order=True)
    env_cfg.observations.policy.joint_pos.params = {"asset_cfg": joint_asset_cfg}
    env_cfg.observations.policy.joint_vel.params = {"asset_cfg": joint_asset_cfg}
    env_cfg.observations.critic.joint_pos.params = {"asset_cfg": joint_asset_cfg}
    env_cfg.observations.critic.joint_vel.params = {"asset_cfg": joint_asset_cfg}

    env_cfg.commands.motion.motion_file = G1_1307_MOTION_FILE
    env_cfg.commands.motion.anchor_body_name = g1_mdp.G1_ANCHOR_BODY_NAME
    env_cfg.commands.motion.root_body_name = g1_mdp.G1_ROOT_BODY_NAME
    env_cfg.commands.motion.body_names = g1_mdp.G1_TRACKING_BODY_NAMES
    env_cfg.commands.motion.motion_body_names = g1_mdp.G1_MOTION_BODY_NAMES
    env_cfg.commands.motion.motion_joint_names = g1_mdp.G1_MOTION_JOINT_NAMES
    env_cfg.commands.motion.min_traj_duration = None
    env_cfg.commands.motion.bridge_frames = 0
    env_cfg.commands.motion.pd_stand_reset_ratio = 0.0

    env_cfg.events.base_com.params["asset_cfg"].body_names = g1_mdp.G1_ANCHOR_BODY_NAME
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    ]
    env_cfg.terminations.ee_body_pos.params["body_names"] = g1_mdp.G1_END_EFFECTOR_BODY_NAMES
    env_cfg.viewer.eye = (-2.5, -2.5, 1.6)


def _make_tolerant_tracking_failure(anchor_z_threshold: float, ee_z_threshold: float):
    return DoneTerm(
        func=mdp.TolerantTermination,
        params={
            "bad_tracking_time_threshold_s": 3.0,
            "command_name": "motion",
            "recovering_shoulder_threshold": 1.0,  # I_recovering gate: shoulder-height deviation (matches before_stand)
            "terms": [
                {
                    "name": "anchor_pos_z",  # name kept for stage helpers; now full 3D L2 (paper-aligned)
                    "func": "bad_anchor_pos",
                    "params": {"command_name": "motion", "threshold": anchor_z_threshold},
                },
                {
                    "name": "anchor_ori",
                    "func": "bad_anchor_ori",
                    "params": {"asset_name": "robot", "command_name": "motion", "threshold": 0.8},
                },
                {
                    "name": "ee_body_pos_z",  # name kept for stage helpers; now full 3D L2 (paper-aligned)
                    "func": "bad_motion_body_pos",
                    "params": {
                        "command_name": "motion",
                        "threshold": ee_z_threshold,
                        "body_names": g1_mdp.G1_END_EFFECTOR_BODY_NAMES,
                    },
                },
            ],
        },
    )


def _set_tolerant_term(cfg, term_name: str, threshold: float):
    for term in cfg.terminations.tracking_failure.params["terms"]:
        if term["name"] == term_name:
            term["params"]["threshold"] = threshold


def _drop_tolerant_terms(cfg, names: set[str]):
    cfg.terminations.tracking_failure.params["terms"] = [
        term for term in cfg.terminations.tracking_failure.params["terms"] if term["name"] not in names
    ]


def _add_stage_ii_tolerant_terms(cfg):
    cfg.terminations.tracking_failure.params["terms"].append(
        {"name": "anchor_pos", "func": "bad_anchor_pos", "params": {"command_name": "motion", "threshold": 1.0}}
    )
    cfg.terminations.tracking_failure.params["terms"].append(
        {"name": "hip_dof", "func": "bad_hip_dof", "params": {"command_name": "motion", "threshold": math.pi / 6.0}}
    )


def _set_g1_standing(env_cfg: TrackingEnvCfg):
    env_cfg.commands.motion = mdp.MotionStandingCommandCfg(
        asset_name="robot",
        motion_file=G1_1307_MOTION_FILE,
        init_pos_file=G1_STANDING_INIT_FILE,
        anchor_body_name=g1_mdp.G1_ANCHOR_BODY_NAME,
        root_body_name=g1_mdp.G1_ROOT_BODY_NAME,
        root_body_names=[g1_mdp.G1_ROOT_BODY_NAME],
        shoulders_body_names=g1_mdp.G1_SHOULDER_BODY_NAMES,
        feet_body_names=g1_mdp.G1_FEET_BODY_NAMES,
        body_names=g1_mdp.G1_TRACKING_BODY_NAMES,
        motion_body_names=g1_mdp.G1_MOTION_BODY_NAMES,
        motion_joint_names=g1_mdp.G1_MOTION_JOINT_NAMES,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        tracking_standing_weight=(1.0, 1.0),
        min_traj_duration=None,
        bridge_frames=0,
        sampling_mode="adaptive",
    )
    _set_g1_common(env_cfg)
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.tracking_failure = _make_tolerant_tracking_failure(
        anchor_z_threshold=0.25,
        ee_z_threshold=0.25,
    )

    env_cfg.rewards.self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "force_threshold": 10.0,
        },
    )
    env_cfg.rewards.electrical_power_cost = RewTerm(
        func=mdp.penalty_electrical_power_cost,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )
    env_cfg.rewards.penalty_relative_shoulder_high = RewTerm(
        func=mdp.penalty_relative_shoulder_high,
        weight=-2.0,
        params={"command_name": "motion"},
    )
    env_cfg.rewards.penalty_relative_root_orientation = RewTerm(
        func=mdp.penalty_relative_root_orientation,
        weight=-0.5,
        params={"command_name": "motion"},
    )
    env_cfg.rewards.penalty_xy_rate_before_stand = RewTerm(
        func=mdp.penalty_xy_rate_before_stand,
        weight=-1.0,
        params={"command_name": "motion", "stand_threshold": 0.1},
    )


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_g1_common(self)


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatStandingEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_g1_standing(self)


@configclass
class G1Flat1307StageIEnvCfg(G1FlatStandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_tolerant_term(self, "anchor_pos_z", 0.5)
        _set_tolerant_term(self, "ee_body_pos_z", 0.4)


@configclass
class G1Flat1307StageIIEnvCfg(G1FlatStandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.reward_center_of_mass = RewTerm(
            func=mdp.reward_center_of_mass,
            weight=1.0,
            params={"command_name": "motion", "sigma_com": 0.1},
        )
        _drop_tolerant_terms(self, {"anchor_pos_z", "ee_body_pos_z"})
        _set_tolerant_term(self, "anchor_ori", 0.6)
        _add_stage_ii_tolerant_terms(self)


@configclass
class G1Flat1307StageIIIEnvCfg(G1Flat1307StageIIEnvCfg):
    isaac_equivalence_notes = G1_STAGE_III_ISAAC_EQUIVALENCE_NOTES

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.velocity_range = VELOCITY_RANGE_ADD
        self.events.push_robot.params["velocity_range"] = VELOCITY_RANGE_ADD
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.01, 0.01),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
                },
                "velocity_range": VELOCITY_RANGE_ADD,
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=g1_mdp.G1_MOTION_JOINT_NAMES, preserve_order=True),
            },
        )


@configclass
class G1Flat1307CheckpointEnvCfg(G1Flat1307StageIEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
