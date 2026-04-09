import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Physical Parameters (based on pm.py motor specs)
# High-torque joints: Q90 motor (HIP_PITCH, HIP_ROLL, KNEE_PITCH)
ARMATURE_Q90 = 0.0453
EFFORT_LIMIT_Q90 = 164.0
VELOCITY_LIMIT_Q90 = 26.3

# Low-torque joints: Q25 motor (HIP_YAW, ANKLE, WAIST, SHOULDER, ELBOW, HEAD)
ARMATURE_Q25 = 0.0067
EFFORT_LIMIT_Q25 = 52.0
VELOCITY_LIMIT_Q25 = 35.2

# Control parameters: 10Hz natural frequency with critical damping
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# Calculate stiffness and damping based on natural frequency and damping ratio
STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2  # ≈ 178.5
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2  # ≈ 26.4
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ  # ≈ 11.4
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ  # ≈ 1.69


PM01_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        asset_path=f"{ASSET_DIR}/pm01/urdf/serial_pm_v2_primitive.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*HIP_PITCH.*": -0.06,
            ".*KNEE_PITCH.*": 0.12,
            ".*ANKLE_PITCH.*": -0.06,
            ".*ELBOW_PITCH.*": -0.25,
            "J14_SHOULDER_ROLL_L": 0.15,
            "J19_SHOULDER_ROLL_R": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*HIP.*",
                ".*KNEE.*",
            ],
            effort_limit_sim={
                ".*HIP_PITCH.*": EFFORT_LIMIT_Q90,
                ".*HIP_ROLL.*": EFFORT_LIMIT_Q90,
                ".*HIP_YAW.*": EFFORT_LIMIT_Q25,
                ".*KNEE_PITCH.*": EFFORT_LIMIT_Q90,
            },
            velocity_limit_sim={
                ".*HIP_PITCH.*": VELOCITY_LIMIT_Q90,
                ".*HIP_ROLL.*": VELOCITY_LIMIT_Q90,
                ".*HIP_YAW.*": VELOCITY_LIMIT_Q25,
                ".*KNEE_PITCH.*": VELOCITY_LIMIT_Q90,
            },
            stiffness={
                ".*HIP_PITCH.*": STIFFNESS_Q90,
                ".*HIP_ROLL.*": STIFFNESS_Q90,
                ".*HIP_YAW.*": STIFFNESS_Q25,
                ".*KNEE_PITCH.*": STIFFNESS_Q90,
            },
            damping={
                ".*HIP_PITCH.*": DAMPING_Q90,
                ".*HIP_ROLL.*": DAMPING_Q90,
                ".*HIP_YAW.*": DAMPING_Q25,
                ".*KNEE_PITCH.*": DAMPING_Q90,
            },
            armature={
                ".*HIP_PITCH.*": ARMATURE_Q90,
                ".*HIP_ROLL.*": ARMATURE_Q90,
                ".*HIP_YAW.*": ARMATURE_Q25,
                ".*KNEE_PITCH.*": ARMATURE_Q90,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=0.5,
            armature=ARMATURE_Q25,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J12_WAIST_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=DAMPING_Q25,
            armature=ARMATURE_Q25,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*SHOULDER.*",
                ".*ELBOW.*",
            ],
            effort_limit_sim={
                ".*SHOULDER.*": EFFORT_LIMIT_Q25,
                ".*ELBOW.*": EFFORT_LIMIT_Q25,
            },
            velocity_limit_sim={
                ".*SHOULDER.*": VELOCITY_LIMIT_Q25,
                ".*ELBOW.*": VELOCITY_LIMIT_Q25,
            },
            stiffness={
                ".*SHOULDER.*": STIFFNESS_Q25,
                ".*ELBOW.*": STIFFNESS_Q25,
            },
            damping={
                ".*SHOULDER.*": DAMPING_Q25,
                ".*ELBOW.*": DAMPING_Q25,
            },
            armature={
                ".*SHOULDER.*": ARMATURE_Q25,
                ".*ELBOW.*": ARMATURE_Q25,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["J23_HEAD_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=DAMPING_Q25,
            armature=ARMATURE_Q25,
        ),
    },
)

PM01_ACTION_SCALE = {}
for a in PM01_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    if not isinstance(e, dict):
        e = {n: e for n in a.joint_names_expr}
    if not isinstance(s, dict):
        s = {n: s for n in a.joint_names_expr}
    for n in e.keys():  # 改用 e.keys() 而不是 joint_names_expr
        # if "ANKLE" in n:
        #     continue
        if n in s and s[n]:
            PM01_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


