import os
import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR


G1_JOINT_NAMES = [
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


def _reflected_inertia(rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]) -> float:
    reflected = 0.0
    gear_product = 1.0
    for inertia, gear in zip(rotor_inertias, gears):
        gear_product *= gear
        reflected += inertia * gear_product**2
    return reflected


ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1.0, 1.0 + (46.0 / 18.0), 1.0 + (56.0 / 16.0))
ARMATURE_5020 = _reflected_inertia(ROTOR_INERTIAS_5020, GEARS_5020)

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1.0, 4.5, 1.0 + (48.0 / 22.0))
ARMATURE_7520_14 = _reflected_inertia(ROTOR_INERTIAS_7520_14, GEARS_7520_14)

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1.0, 4.5, 5.0)
ARMATURE_7520_22 = _reflected_inertia(ROTOR_INERTIAS_7520_22, GEARS_7520_22)

# B1 fix: the two downstream gear-stage rotor inertias were 0.0 (clearly a typo), which made
# ARMATURE_4010~6.8e-6, STIFFNESS_4010~0.027 and the wrist action scale ~46.6 rad (vs ~0.36 elsewhere).
# Filled with values that put the wrist on the same order as peer joints (stiffness~5, action scale~0.25).
ROTOR_INERTIAS_4010 = (0.068e-4, 0.010e-4, 0.020e-4)
GEARS_4010 = (1.0, 5.0, 5.0)
ARMATURE_4010 = _reflected_inertia(ROTOR_INERTIAS_4010, GEARS_4010)

EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0
VELOCITY_LIMIT_5020 = 37.0
VELOCITY_LIMIT_7520_14 = 32.0
VELOCITY_LIMIT_7520_22 = 20.0
VELOCITY_LIMIT_4010 = 22.0

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ASSET_DIR = os.path.join(ASSET_DIR, "g1")
G1_XML_PATH = os.path.join(G1_ASSET_DIR, "xmls", "g1.xml")
G1_URDF_PATH = os.path.join(ASSET_DIR, "g1_description", "urdf", "g1_29dof_rev_1_0.urdf")

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=G1_URDF_PATH,
        fix_base=False,
        root_link_name="base",
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms_5020": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_elbow_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=EFFORT_LIMIT_5020,
            velocity_limit_sim=VELOCITY_LIMIT_5020,
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
        ),
        "hip_yaw_pitch_7520_14": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=EFFORT_LIMIT_7520_14,
            velocity_limit_sim=VELOCITY_LIMIT_7520_14,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "hip_roll_knee_7520_22": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=EFFORT_LIMIT_7520_22,
            velocity_limit_sim=VELOCITY_LIMIT_7520_22,
            stiffness=STIFFNESS_7520_22,
            damping=DAMPING_7520_22,
            armature=ARMATURE_7520_22,
        ),
        "wrists_4010": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=EFFORT_LIMIT_4010,
            velocity_limit_sim=VELOCITY_LIMIT_4010,
            stiffness=STIFFNESS_4010,
            damping=DAMPING_4010,
            armature=ARMATURE_4010,
        ),
        "waist_ankles_5020_pair": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=EFFORT_LIMIT_5020 * 2.0,
            velocity_limit_sim=VELOCITY_LIMIT_5020,
            stiffness=STIFFNESS_5020 * 2.0,
            damping=DAMPING_5020 * 2.0,
            armature=ARMATURE_5020 * 2.0,
        ),
    },
)

G1_ACTION_SCALE = {}
for actuator in G1_CFG.actuators.values():
    effort_limit = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in actuator.joint_names_expr}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in actuator.joint_names_expr}
    for pattern, effort in effort_limit.items():
        if pattern not in stiffness or not stiffness[pattern]:
            continue
        for joint_name in G1_JOINT_NAMES:
            if re.fullmatch(pattern, joint_name):
                G1_ACTION_SCALE[joint_name] = 0.25 * effort / stiffness[pattern]
