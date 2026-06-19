from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.actuator import DelayedImplicitActuatorCfg

from . import g1_mdp
from .flat_env_cfg import (
    G1Flat1307StageIEnvCfg,
    G1Flat1307StageIIEnvCfg,
    G1Flat1307StageIIIEnvCfg,
    _drop_tolerant_terms,
    _set_tolerant_term,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE
from .paper_contract import get_paper_equivalence_contract


G1_PAPER_ACTION_DELAY_RANGE_STEPS = (0, 3)


G1_PAPER_SOURCE_HIERARCHY = {
    "paper_primary": "arXiv:2602.13656",
    "fastsac_reference": "arXiv:2512.01996",
    "public_repo": "out_of_scope",
}


G1_PAPER_RANDOMIZATION_EQUIVALENCE = {
    "mass": {
        "paper_requirement": "Randomize robot masses/inertial properties.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "IsaacLab rigid body mass randomization event for G1 bodies.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/events.py"],
    },
    "friction": {
        "paper_requirement": "Randomize contact/foot friction.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "randomize_rigid_body_material startup event.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"],
    },
    "base_com": {
        "paper_requirement": "Randomize base center of mass.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "randomize_rigid_body_com startup event.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/events.py"],
    },
    "pd_gain": {
        "paper_requirement": "Randomize PD gains/stiffness/damping.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "Actuator gain scale randomization for G1 actuator groups.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/robots/g1.py"],
    },
    "action_delay": {
        "paper_requirement": "Randomize action delay.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "PaperFastSAC G1 robot actuator groups use delayed implicit actuators.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/robots/actuator.py"],
        "delay_range_steps": G1_PAPER_ACTION_DELAY_RANGE_STEPS,
    },
    "torque_disturbance": {
        "paper_requirement": "Apply torque disturbances/noise.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "Add IsaacLab torque disturbance event or actuator noise hook.",
        "evidence_refs": ["isaaclab.envs.mdp.apply_external_force_torque"],
    },
    "joint_bias": {
        "paper_requirement": "Randomize joint encoder or position bias.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "randomize_joint_default_pos startup event.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/events.py"],
    },
    "push": {
        "paper_requirement": "Apply randomized pushes.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "push_by_setting_velocity interval event.",
        "evidence_refs": ["source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"],
    },
    "reset_pose_velocity": {
        "paper_requirement": "Randomize reset pose and velocities.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "Stage III reset_base and reset_robot_joints events.",
        "evidence_refs": [
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"
        ],
    },
    "terrain": {
        "paper_requirement": "Train with randomized terrain/contact surface variation.",
        "status": "paper_category_required",
        "range_source": "local",
        "implementation_target": "PaperFastSAC Stage III uses IsaacLab rough terrain generator.",
        "evidence_refs": [
            "isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG",
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/paper_full_env_cfg.py",
        ],
        "stage": "PaperFastSAC-Stage-III",
    },
}


G1_PAPER_REWARD_PROFILE = {
    "relative_body_pos": {"weight": 4.0, "status": "implemented"},
    "relative_body_ori": {"weight": 2.0, "status": "implemented"},
    "global_body_ang_vel": {"weight": 1.0, "status": "implemented"},
    "center_of_mass": {"weight": 2.0, "status": "implemented"},
    "close_feet": {"weight": -1000.0, "status": "implemented"},
    "feet_slip": {"weight": -2.0, "status": "implemented"},
    "root_orientation": {"weight": -1.0, "status": "implemented"},
    "knee_action_rate": {"weight": -3.0, "status": "implemented"},
    "ankle_action_rate": {"weight": -20.0, "status": "implemented"},
    "soft_dof_limit": {"weight": -100.0, "status": "implemented"},
    "undesired_contacts": {"weight": -0.5, "status": "implemented"},
    "shoulder_height_before_stand": {"weight": -2.0, "status": "implemented"},
    "xy_root_movement_before_stand": {"weight": -1.0, "status": "implemented"},
    "action_rate_before_stand": {"weight": -2.0, "status": "implemented"},
}


def _apply_paper_full_reward_weights(cfg) -> None:
    rewards = cfg.rewards
    rewards.motion_body_pos.weight = G1_PAPER_REWARD_PROFILE["relative_body_pos"]["weight"]
    rewards.motion_body_ori.weight = G1_PAPER_REWARD_PROFILE["relative_body_ori"]["weight"]
    rewards.motion_body_ang_vel.weight = G1_PAPER_REWARD_PROFILE["global_body_ang_vel"]["weight"]
    rewards.joint_limit.weight = G1_PAPER_REWARD_PROFILE["soft_dof_limit"]["weight"]
    rewards.penalty_relative_root_orientation.weight = G1_PAPER_REWARD_PROFILE["root_orientation"]["weight"]
    rewards.penalty_relative_shoulder_high.weight = G1_PAPER_REWARD_PROFILE["shoulder_height_before_stand"]["weight"]
    rewards.penalty_xy_rate_before_stand.weight = G1_PAPER_REWARD_PROFILE["xy_root_movement_before_stand"]["weight"]
    rewards.penalty_xy_rate_before_stand.params["stand_threshold"] = 1.0
    rewards.action_rate_l2.weight = 0.0
    rewards.undesired_contacts.weight = G1_PAPER_REWARD_PROFILE["undesired_contacts"]["weight"]
    # Paper tracking reward set is only body pos/ori/ang-vel + CoM. Zero the non-paper tracking terms.
    rewards.motion_global_anchor_pos.weight = 0.0
    rewards.motion_global_anchor_ori.weight = 0.0
    rewards.motion_body_lin_vel.weight = 0.0  # paper has no linear-velocity tracking term
    # Non-paper penalties carried in from the standing config (paper only keeps undesired_contacts).
    rewards.self_collisions.weight = 0.0
    rewards.electrical_power_cost.weight = 0.0

    rewards.reward_center_of_mass = RewTerm(
        func=mdp.reward_center_of_mass,
        weight=G1_PAPER_REWARD_PROFILE["center_of_mass"]["weight"],
        params={"command_name": "motion", "sigma_com": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    rewards.close_feet = RewTerm(
        func=mdp.close_feet_penalty,
        weight=G1_PAPER_REWARD_PROFILE["close_feet"]["weight"],
        params={"command_name": "motion", "min_distance": 0.16},  # paper-aligned close_feet threshold (was 0.18)
    )
    rewards.feet_slip = RewTerm(
        func=mdp.feet_slip_penalty,
        weight=G1_PAPER_REWARD_PROFILE["feet_slip"]["weight"],
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=g1_mdp.G1_FEET_BODY_NAMES),
            "force_threshold": 8.0,
        },
    )
    rewards.knee_action_rate = RewTerm(
        func=mdp.action_rate_l2_by_joint_names,
        weight=G1_PAPER_REWARD_PROFILE["knee_action_rate"]["weight"],
        params={
            "command_name": "motion",
            "joint_names": ["left_knee_joint", "right_knee_joint"],
        },
    )
    rewards.ankle_action_rate = RewTerm(
        func=mdp.action_rate_l2_by_joint_names,
        weight=G1_PAPER_REWARD_PROFILE["ankle_action_rate"]["weight"],
        params={
            "command_name": "motion",
            "joint_names": [
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
        },
    )
    rewards.penalty_action_rate_before_stand = RewTerm(
        func=mdp.penalty_action_rate_before_stand,
        weight=G1_PAPER_REWARD_PROFILE["action_rate_before_stand"]["weight"],
        params={"command_name": "motion", "stand_threshold": 1.0},
    )


def _apply_paper_full_randomization(cfg) -> None:
    cfg.events.robot_mass_scale = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    cfg.events.robot_pd_gain_scale = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # L4-b: torque disturbance removed (not in the paper). push_robot is kept (paper-aligned perturbation).
    cfg.events.torque_disturbance = None


def _make_delayed_actuator_cfg(actuator) -> DelayedImplicitActuatorCfg:
    return DelayedImplicitActuatorCfg(
        joint_names_expr=list(actuator.joint_names_expr),
        effort_limit=actuator.effort_limit,
        velocity_limit=actuator.velocity_limit,
        effort_limit_sim=actuator.effort_limit_sim,
        velocity_limit_sim=actuator.velocity_limit_sim,
        stiffness=actuator.stiffness,
        damping=actuator.damping,
        armature=actuator.armature,
        friction=actuator.friction,
        dynamic_friction=actuator.dynamic_friction,
        viscous_friction=actuator.viscous_friction,
        min_delay=G1_PAPER_ACTION_DELAY_RANGE_STEPS[0],
        max_delay=G1_PAPER_ACTION_DELAY_RANGE_STEPS[1],
    )


def _apply_paper_full_action_delay(cfg) -> None:
    cfg.scene.robot.actuators = {
        name: _make_delayed_actuator_cfg(actuator) for name, actuator in cfg.scene.robot.actuators.items()
    }


def _apply_paper_full_rough_terrain(cfg) -> None:
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.replace(
        seed=cfg.seed,
        curriculum=False,
        size=(4.0, 4.0),
        border_width=1.0,
        num_rows=2,
        num_cols=2,
        use_cache=False,
    )
    cfg.scene.terrain.visual_material = None


class G1PaperFullMixin:
    paper_full_algorithm = "FastSAC"
    paper_equivalence_contract = get_paper_equivalence_contract()
    grsi_state_file = "data/g1/grsi_states.pth"
    recovery_tracking_mix = (1.0, 1.0)
    paper_rough_terrain = False

    def _apply_paper_full_common(self):
        self.paper_equivalence_contract = get_paper_equivalence_contract()
        self.commands.motion.sampling_mode = "lke"
        self.commands.motion.tracking_standing_weight = self.recovery_tracking_mix
        self.commands.motion.init_pos_file = self.grsi_state_file
        _apply_paper_full_action_delay(self)
        _apply_paper_full_reward_weights(self)
        _apply_paper_full_randomization(self)
        if self.paper_rough_terrain:
            _apply_paper_full_rough_terrain(self)
        self.paper_source_hierarchy = G1_PAPER_SOURCE_HIERARCHY
        self.paper_randomization_equivalence = G1_PAPER_RANDOMIZATION_EQUIVALENCE
        self.paper_reward_profile = G1_PAPER_REWARD_PROFILE


@configclass
class G1Flat1307PaperFastSACStageIEnvCfg(G1PaperFullMixin, G1Flat1307StageIEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._apply_paper_full_common()


@configclass
class G1Flat1307PaperFastSACStageIIEnvCfg(G1PaperFullMixin, G1Flat1307StageIIEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._apply_paper_full_common()


@configclass
class G1Flat1307PaperFastSACStageIIIEnvCfg(G1PaperFullMixin, G1Flat1307StageIIIEnvCfg):
    paper_rough_terrain = False  # L4-a: rough terrain off (not in the paper); was True

    def __post_init__(self):
        super().__post_init__()
        self._apply_paper_full_common()


@configclass
class G1Flat1307PaperFastSACUnifiedEnvCfg(G1PaperFullMixin, G1Flat1307StageIIEnvCfg):
    """A1: single-stage paper-aligned config (no Stage I/II/III curriculum).

    Built on the Stage-II env (CoM reward + 3D anchor_pos/hip_dof tolerant terminations), plus:
    - paper-full common (full reward profile, DR, GRSI init, LKES sampling, action delay);
    - Stage-III-style reset events so episodes reset across the whole motion from the GRSI pool;
    - base VELOCITY_RANGE and plane terrain (no curriculum escalation, no rough terrain);
    - an added full-3D-L2 body-position termination over the tracked body set B (paper L2-7).
    torque disturbance and rough terrain are off via the L4 edits above.
    """

    paper_rough_terrain = False

    def __post_init__(self):
        super().__post_init__()
        self._apply_paper_full_common()
        # Single-stage reset events (from Stage III), but with the base (non-escalated) velocity range.
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
                "velocity_range": VELOCITY_RANGE,
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
        # Paper success metrics are exactly 3: anchor position (tau_pos=0.5), body position over B
        # (tau_body=0.5), orientation (tau_ori=0.8). Align thresholds and drop the non-paper hip_dof term.
        _drop_tolerant_terms(self, {"hip_dof"})
        _set_tolerant_term(self, "anchor_pos", 0.5)  # paper tau_pos = 0.5 (Stage-II had 1.0)
        _set_tolerant_term(self, "anchor_ori", 0.8)  # paper tau_ori = 0.8 (Stage-II tightened to 0.6)
        # L2-7: full 3D L2 body-position termination over the tracked body set B (tau_body=0.5).
        self.terminations.tracking_failure.params["terms"].append(
            {
                "name": "body_pos",
                "func": "bad_motion_body_pos",
                "params": {
                    "command_name": "motion",
                    "threshold": 0.5,
                    "body_names": g1_mdp.G1_TRACKING_BODY_NAMES,
                },
            }
        )
