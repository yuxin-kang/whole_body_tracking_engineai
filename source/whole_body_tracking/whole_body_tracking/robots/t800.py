import os
import tempfile
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR


def _implicit_actuator_cfg(**kwargs) -> ImplicitActuatorCfg:
    """Build an actuator config across Isaac Lab friction API versions."""
    for field_name in ("dynamic_friction", "viscous_friction"):
        if not hasattr(ImplicitActuatorCfg, field_name):
            kwargs.pop(field_name, None)
    return ImplicitActuatorCfg(**kwargs)

# T800 motor/inertia parameters aligned with the original T800 setup.
ARMATURE_Q300H_L = 0.2427264
ARMATURE_Q300H = 0.14110848
ARMATURE_Q200H = 0.0448737
ARMATURE_Q50H = 0.0354625
ARMATURE_Q25H = 0.00671625

EFFORT_LIMIT_Q300H_L = 415.0
EFFORT_LIMIT_Q300H = 370.0
EFFORT_LIMIT_Q200H = 222.0
EFFORT_LIMIT_Q50H = 160.0
EFFORT_LIMIT_Q25H = 52.0

VELOCITY_LIMIT_Q300H_L = 25.96
VELOCITY_LIMIT_Q300H = 25.31
VELOCITY_LIMIT_Q200H = 23.19
VELOCITY_LIMIT_Q50H = 33.51
VELOCITY_LIMIT_Q25H = 35.2

DEFAULT_Q_HIP_PITCH = -0.06
DEFAULT_Q_HIP_ROLL = 0.0
DEFAULT_Q_HIP_YAW = 0.0
DEFAULT_Q_KNEE_PITCH = 0.12
DEFAULT_Q_ANKLE_PITCH = -0.06
DEFAULT_Q_ANKLE_ROLL = 0.0
DEFAULT_Q_TORSO_YAW = 0.0
DEFAULT_Q_SHOULDER_PITCH = 0.0
DEFAULT_Q_SHOULDER_ROLL_L = 0.15
DEFAULT_Q_SHOULDER_ROLL_R = -0.15
DEFAULT_Q_SHOULDER_YAW = 0.0
DEFAULT_Q_ELBOW_PITCH = -0.25
DEFAULT_Q_ELBOW_YAW = 0.0
DEFAULT_Q_HEAD_PITCH = 0.0
DEFAULT_Q_HEAD_YAW = 0.0

STIFFNESS_Q300H_L = 180.0
STIFFNESS_Q300H = 100.0
STIFFNESS_Q200H = 100.0
STIFFNESS_Q50H = 40.0
STIFFNESS_Q25H = 50.0

DAMPING_Q300H_L = 5.0
DAMPING_Q300H = 3.0
DAMPING_Q200H = 3.0
DAMPING_Q50H = 0.3
DAMPING_Q25H = 0.3

T800_URDF_PATH = os.path.join(ASSET_DIR, "t800", "urdf", "serial_t800.urdf")
T800_USD_DIR = os.path.join(tempfile.gettempdir(), "IsaacLab", "t800")

# Collision primitives from engineai_robotics_native_sdk/assets/resource/t800.xml
# (MuJoCo 3.2.3).  MuJoCo cylinders are Z-axis primitives; ``length`` below is
# their full length.  The SDK's fixed foot bodies are identity children of the
# ankle-roll links, so their three primitives belong on the ankle-roll links in
# the URDF as well (rather than on this URDF's offset fixed FOOT links).
MUJOCO_COLLISION_PROFILE_VERSION = "engineai-sdk-t800-mujoco-3.2.3-v1"
_MUJOCO_COLLISION_PRIMITIVES = {
    "LINK_BASE": (("sphere", "-0.01 0 0.01", "0 0 0", "0.09"),),
    "LINK_HIP_PITCH_L": (("cylinder", "0 0.04 -0.10", "0 1.5707963267948966 0", "0.062 0.16"),),
    "LINK_HIP_PITCH_R": (("cylinder", "0 -0.04 -0.10", "0 1.5707963267948966 0", "0.062 0.16"),),
    "LINK_HIP_YAW_L": (("box", "0 0.002 -0.10", "0 0 0", "0.12 0.12 0.22"),),
    "LINK_HIP_YAW_R": (("box", "0 -0.002 -0.10", "0 0 0", "0.12 0.12 0.22"),),
    "LINK_KNEE_PITCH_L": (
        ("sphere", "0 0 0", "0 0 0", "0.08"),
        ("box", "-0.02 0 -0.25", "0 0 0", "0.10 0.10 0.28"),
    ),
    "LINK_KNEE_PITCH_R": (
        ("sphere", "0 0 0", "0 0 0", "0.08"),
        ("box", "-0.02 0 -0.25", "0 0 0", "0.10 0.10 0.28"),
    ),
    "LINK_ANKLE_ROLL_L": (
        ("box", "0.1 0.00166 -0.054", "0 0 0", "0.114 0.11 0.02"),
        ("box", "-0.053 0.00166 -0.054", "0 0 0", "0.06 0.10 0.02"),
        ("box", "-0.092 0.00166 -0.045", "0 0.592879010508 0", "0.036 0.10 0.02"),
    ),
    "LINK_ANKLE_ROLL_R": (
        ("box", "0.1 -0.00166 -0.054", "0 0 0", "0.114 0.11 0.02"),
        ("box", "-0.053 -0.00166 -0.054", "0 0 0", "0.06 0.10 0.02"),
        ("box", "-0.092 -0.00166 -0.045", "0 0.592879010508 0", "0.036 0.10 0.02"),
    ),
    # MuJoCo calls this LINK_WAIST_YAW; the Isaac training URDF calls it
    # LINK_TORSO_YAW.  It is the same kinematic link.
    "LINK_TORSO_YAW": (
        ("box", "-0.007 -0.001 0.075", "0 0 0", "0.18 0.20 0.14"),
        ("box", "-0.007 -0.001 0.25", "0 0 0", "0.22 0.28 0.19"),
    ),
    "LINK_SHOULDER_ROLL_L": (("cylinder", "0 0 0", "0 1.5707963267948966 0", "0.042 0.12"),),
    "LINK_SHOULDER_ROLL_R": (("cylinder", "0 0 0", "0 1.5707963267948966 0", "0.042 0.12"),),
    "LINK_SHOULDER_YAW_L": (("cylinder", "0 0 -0.04", "0 0 0", "0.042 0.12"),),
    "LINK_SHOULDER_YAW_R": (("cylinder", "0 0 -0.04", "0 0 0", "0.042 0.12"),),
    "LINK_ELBOW_PITCH_L": (("sphere", "0 0 0", "0 0 0", "0.05"),),
    "LINK_ELBOW_PITCH_R": (("sphere", "0 0 0", "0 0 0", "0.05"),),
    "LINK_ELBOW_YAW_L": (("box", "0.014 0.005 -0.05", "-0.000897139264503 -0.175544008532 0.261733738702", "0.07 0.07 0.16"),),
    "LINK_ELBOW_YAW_R": (("box", "0.014 -0.005 -0.05", "0.000897139264503 -0.175544008532 -0.261733738702", "0.07 0.07 0.16"),),
    "LINK_WRIST_END_L": (("sphere", "0.026 -0.009 -0.06", "0 0 0", "0.05"),),
    "LINK_WRIST_END_R": (("sphere", "0.026 0.009 -0.06", "0 0 0", "0.05"),),
    "LINK_HEAD_YAW": (("sphere", "0.018 0 0.08", "0 0 0", "0.09"),),
}


def _make_mujoco_collision_urdf() -> str:
    """Materialize a URDF whose collision primitives match the native SDK.

    Visual meshes, inertials, joint axes/limits and all non-collision fields
    remain the ordinary Isaac training URDF.  Atomic replacement is necessary
    because two Slurm steps can begin importing the asset concurrently.
    """
    output_dir = os.path.join(tempfile.gettempdir(), "IsaacLab", "t800_mujoco_collision")
    output_path = os.path.join(output_dir, f"serial_t800_{MUJOCO_COLLISION_PROFILE_VERSION}.urdf")
    if os.path.isfile(output_path):
        return output_path

    root = ET.parse(T800_URDF_PATH).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    if set(_MUJOCO_COLLISION_PRIMITIVES) - set(links):
        raise RuntimeError("MuJoCo collision profile refers to a missing T800 URDF link")
    for link in links.values():
        for collision in tuple(link.findall("collision")):
            link.remove(collision)
    for link_name, primitives in _MUJOCO_COLLISION_PRIMITIVES.items():
        link = links[link_name]
        for primitive_type, xyz, rpy, dimensions in primitives:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", xyz=xyz, rpy=rpy)
            geometry = ET.SubElement(collision, "geometry")
            if primitive_type == "box":
                ET.SubElement(geometry, "box", size=dimensions)
            elif primitive_type == "sphere":
                ET.SubElement(geometry, "sphere", radius=dimensions)
            elif primitive_type == "cylinder":
                radius, length = dimensions.split()
                ET.SubElement(geometry, "cylinder", radius=radius, length=length)
            else:
                raise RuntimeError(f"Unsupported MuJoCo collision primitive: {primitive_type}")

    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".serial_t800_", suffix=".urdf", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
        os.replace(temporary_path, output_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return output_path


T800_MUJOCO_COLLISION_URDF_PATH = _make_mujoco_collision_urdf()

T800_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        asset_path=T800_URDF_PATH,
        usd_dir=T800_USD_DIR,
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
            "J00_HIP_PITCH_L": DEFAULT_Q_HIP_PITCH,
            "J01_HIP_ROLL_L": DEFAULT_Q_HIP_ROLL,
            "J02_HIP_YAW_L": DEFAULT_Q_HIP_YAW,
            "J03_KNEE_PITCH_L": DEFAULT_Q_KNEE_PITCH,
            "J04_ANKLE_PITCH_L": DEFAULT_Q_ANKLE_PITCH,
            "J05_ANKLE_ROLL_L": DEFAULT_Q_ANKLE_ROLL,
            "J06_HIP_PITCH_R": DEFAULT_Q_HIP_PITCH,
            "J07_HIP_ROLL_R": DEFAULT_Q_HIP_ROLL,
            "J08_HIP_YAW_R": DEFAULT_Q_HIP_YAW,
            "J09_KNEE_PITCH_R": DEFAULT_Q_KNEE_PITCH,
            "J10_ANKLE_PITCH_R": DEFAULT_Q_ANKLE_PITCH,
            "J11_ANKLE_ROLL_R": DEFAULT_Q_ANKLE_ROLL,
            "J12_TORSO_YAW": DEFAULT_Q_TORSO_YAW,
            "J13_SHOULDER_PITCH_L": DEFAULT_Q_SHOULDER_PITCH,
            "J14_SHOULDER_ROLL_L": DEFAULT_Q_SHOULDER_ROLL_L,
            "J15_SHOULDER_YAW_L": DEFAULT_Q_SHOULDER_YAW,
            "J16_ELBOW_PITCH_L": DEFAULT_Q_ELBOW_PITCH,
            "J17_ELBOW_YAW_L": DEFAULT_Q_ELBOW_YAW,
            "J20_SHOULDER_PITCH_R": DEFAULT_Q_SHOULDER_PITCH,
            "J21_SHOULDER_ROLL_R": DEFAULT_Q_SHOULDER_ROLL_R,
            "J22_SHOULDER_YAW_R": DEFAULT_Q_SHOULDER_YAW,
            "J23_ELBOW_PITCH_R": DEFAULT_Q_ELBOW_PITCH,
            "J24_ELBOW_YAW_R": DEFAULT_Q_ELBOW_YAW,
            "J27_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
            "J28_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": _implicit_actuator_cfg(
            joint_names_expr=[
                ".*_HIP_PITCH.*",
                ".*_HIP_ROLL.*",
                ".*_HIP_YAW.*",
                ".*_KNEE_PITCH.*",
            ],
            effort_limit_sim={
                ".*_HIP_PITCH.*": EFFORT_LIMIT_Q300H_L,
                ".*_HIP_ROLL.*": EFFORT_LIMIT_Q300H,
                ".*_HIP_YAW.*": EFFORT_LIMIT_Q200H,
                ".*_KNEE_PITCH.*": EFFORT_LIMIT_Q300H_L,
            },
            velocity_limit_sim={
                ".*_HIP_PITCH.*": VELOCITY_LIMIT_Q300H_L,
                ".*_HIP_ROLL.*": VELOCITY_LIMIT_Q300H,
                ".*_HIP_YAW.*": VELOCITY_LIMIT_Q200H,
                ".*_KNEE_PITCH.*": VELOCITY_LIMIT_Q300H_L,
            },
            stiffness={
                ".*_HIP_PITCH.*": STIFFNESS_Q300H_L,
                ".*_HIP_ROLL.*": STIFFNESS_Q300H,
                ".*_HIP_YAW.*": STIFFNESS_Q200H,
                ".*_KNEE_PITCH.*": STIFFNESS_Q300H_L,
            },
            damping={
                ".*_HIP_PITCH.*": DAMPING_Q300H_L,
                ".*_HIP_ROLL.*": DAMPING_Q300H,
                ".*_HIP_YAW.*": DAMPING_Q200H,
                ".*_KNEE_PITCH.*": DAMPING_Q300H_L,
            },
            armature={
                ".*_HIP_PITCH.*": ARMATURE_Q300H_L,
                ".*_HIP_ROLL.*": ARMATURE_Q300H,
                ".*_HIP_YAW.*": ARMATURE_Q200H,
                ".*_KNEE_PITCH.*": ARMATURE_Q300H_L,
            },
            friction={
                ".*_HIP_PITCH.*": 0.1,
                ".*_HIP_ROLL.*": 0.1,
                ".*_HIP_YAW.*": 0.05,
                ".*_KNEE_PITCH.*": 0.1,
            },
            dynamic_friction={
                ".*_HIP_PITCH.*": 0.08,
                ".*_HIP_ROLL.*": 0.08,
                ".*_HIP_YAW.*": 0.04,
                ".*_KNEE_PITCH.*": 0.08,
            },
            viscous_friction={
                ".*_HIP_PITCH.*": 0.01,
                ".*_HIP_ROLL.*": 0.01,
                ".*_HIP_YAW.*": 0.005,
                ".*_KNEE_PITCH.*": 0.01,
            },
        ),
        "feet": _implicit_actuator_cfg(
            joint_names_expr=[".*_ANKLE_PITCH.*", ".*_ANKLE_ROLL.*"],
            effort_limit_sim={
                ".*_ANKLE_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_ANKLE_ROLL.*": EFFORT_LIMIT_Q50H,
            },
            velocity_limit_sim={
                ".*_ANKLE_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_ANKLE_ROLL.*": VELOCITY_LIMIT_Q50H,
            },
            stiffness={
                ".*_ANKLE_PITCH.*": STIFFNESS_Q50H,
                ".*_ANKLE_ROLL.*": STIFFNESS_Q50H,
            },
            damping={
                ".*_ANKLE_PITCH.*": DAMPING_Q50H,
                ".*_ANKLE_ROLL.*": DAMPING_Q50H,
            },
            armature={
                ".*_ANKLE_ROLL.*": ARMATURE_Q50H,
                ".*_ANKLE_PITCH.*": ARMATURE_Q50H,
            },
            friction={
                ".*_ANKLE_PITCH.*": 0.15,
                ".*_ANKLE_ROLL.*": 0.12,
            },
            dynamic_friction={
                ".*_ANKLE_PITCH.*": 0.12,
                ".*_ANKLE_ROLL.*": 0.1,
            },
            viscous_friction={
                ".*_ANKLE_PITCH.*": 0.015,
                ".*_ANKLE_ROLL.*": 0.012,
            },
        ),
        "torso_yaw": _implicit_actuator_cfg(
            joint_names_expr=["J12_TORSO_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q200H,
            velocity_limit_sim=VELOCITY_LIMIT_Q200H,
            stiffness=STIFFNESS_Q200H,
            damping=DAMPING_Q200H,
            armature=ARMATURE_Q200H,
            friction=0.08,
            dynamic_friction=0.06,
            viscous_friction=0.008,
        ),
        "arms": _implicit_actuator_cfg(
            joint_names_expr=[
                ".*_SHOULDER_PITCH.*",
                ".*_SHOULDER_ROLL.*",
                ".*_SHOULDER_YAW.*",
                ".*_ELBOW_PITCH.*",
                ".*_ELBOW_YAW.*",
            ],
            effort_limit_sim={
                ".*_SHOULDER_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_SHOULDER_ROLL.*": EFFORT_LIMIT_Q50H,
                ".*_SHOULDER_YAW.*": EFFORT_LIMIT_Q50H,
                ".*_ELBOW_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_ELBOW_YAW.*": EFFORT_LIMIT_Q25H,
            },
            velocity_limit_sim={
                ".*_SHOULDER_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_SHOULDER_ROLL.*": VELOCITY_LIMIT_Q50H,
                ".*_SHOULDER_YAW.*": VELOCITY_LIMIT_Q50H,
                ".*_ELBOW_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_ELBOW_YAW.*": VELOCITY_LIMIT_Q25H,
            },
            stiffness={
                ".*_SHOULDER_PITCH.*": STIFFNESS_Q50H,
                ".*_SHOULDER_ROLL.*": STIFFNESS_Q50H,
                ".*_SHOULDER_YAW.*": STIFFNESS_Q50H,
                ".*_ELBOW_PITCH.*": STIFFNESS_Q50H,
                ".*_ELBOW_YAW.*": STIFFNESS_Q25H,
            },
            damping={
                ".*_SHOULDER_PITCH.*": DAMPING_Q50H,
                ".*_SHOULDER_ROLL.*": DAMPING_Q50H,
                ".*_SHOULDER_YAW.*": DAMPING_Q50H,
                ".*_ELBOW_PITCH.*": DAMPING_Q50H,
                ".*_ELBOW_YAW.*": DAMPING_Q25H,
            },
            armature={
                ".*_SHOULDER_PITCH.*": ARMATURE_Q50H,
                ".*_SHOULDER_ROLL.*": ARMATURE_Q50H,
                ".*_SHOULDER_YAW.*": ARMATURE_Q50H,
                ".*_ELBOW_PITCH.*": ARMATURE_Q50H,
                ".*_ELBOW_YAW.*": ARMATURE_Q25H,
            },
            friction={
                ".*_SHOULDER_PITCH.*": 0.12,
                ".*_SHOULDER_ROLL.*": 0.1,
                ".*_SHOULDER_YAW.*": 0.08,
                ".*_ELBOW_PITCH.*": 0.08,
                ".*_ELBOW_YAW.*": 0.06,
            },
            dynamic_friction={
                ".*_SHOULDER_PITCH.*": 0.1,
                ".*_SHOULDER_ROLL.*": 0.08,
                ".*_SHOULDER_YAW.*": 0.06,
                ".*_ELBOW_PITCH.*": 0.06,
                ".*_ELBOW_YAW.*": 0.05,
            },
            viscous_friction={
                ".*_SHOULDER_PITCH.*": 0.012,
                ".*_SHOULDER_ROLL.*": 0.01,
                ".*_SHOULDER_YAW.*": 0.008,
                ".*_ELBOW_PITCH.*": 0.008,
                ".*_ELBOW_YAW.*": 0.006,
            },
        ),
        "head": _implicit_actuator_cfg(
            joint_names_expr=["J27_HEAD_PITCH", "J28_HEAD_YAW"],
            effort_limit_sim={
                "J27_HEAD_PITCH": EFFORT_LIMIT_Q25H,
                "J28_HEAD_YAW": EFFORT_LIMIT_Q25H,
            },
            velocity_limit_sim={
                "J27_HEAD_PITCH": VELOCITY_LIMIT_Q25H,
                "J28_HEAD_YAW": VELOCITY_LIMIT_Q25H,
            },
            stiffness={
                "J27_HEAD_PITCH": STIFFNESS_Q25H,
                "J28_HEAD_YAW": STIFFNESS_Q25H,
            },
            damping={
                "J27_HEAD_PITCH": DAMPING_Q25H,
                "J28_HEAD_YAW": DAMPING_Q25H,
            },
            armature={
                "J27_HEAD_PITCH": ARMATURE_Q25H,
                "J28_HEAD_YAW": ARMATURE_Q25H,
            },
            friction={
                "J27_HEAD_PITCH": 0.05,
                "J28_HEAD_YAW": 0.05,
            },
            dynamic_friction={
                "J27_HEAD_PITCH": 0.04,
                "J28_HEAD_YAW": 0.04,
            },
            viscous_friction={
                "J27_HEAD_PITCH": 0.005,
                "J28_HEAD_YAW": 0.005,
            },
        ),
    },
)

# Keep the regular T800 task on the original training collision profile.  This
# alternate asset is selected only by the dedicated MuJoCo-aligned get-up task.
T800_MUJOCO_COLLISION_CFG = T800_CFG.replace(
    spawn=T800_CFG.spawn.replace(asset_path=T800_MUJOCO_COLLISION_URDF_PATH)
)


T800_ACTION_SCALE = {
    "J00_HIP_PITCH_L": 0.5,
    "J01_HIP_ROLL_L": 0.2,
    "J02_HIP_YAW_L": 0.2,
    "J03_KNEE_PITCH_L": 0.5,
    "J04_ANKLE_PITCH_L": 0.5,
    "J05_ANKLE_ROLL_L": 0.2,
    "J06_HIP_PITCH_R": 0.5,
    "J07_HIP_ROLL_R": 0.2,
    "J08_HIP_YAW_R": 0.2,
    "J09_KNEE_PITCH_R": 0.5,
    "J10_ANKLE_PITCH_R": 0.5,
    "J11_ANKLE_ROLL_R": 0.2,
    "J12_TORSO_YAW": 0.2,
    "J13_SHOULDER_PITCH_L": 0.2,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.2,
    "J17_ELBOW_YAW_L": 0.05,
    "J20_SHOULDER_PITCH_R": 0.2,
    "J21_SHOULDER_ROLL_R": 0.2,
    "J22_SHOULDER_YAW_R": 0.05,
    "J23_ELBOW_PITCH_R": 0.2,
    "J24_ELBOW_YAW_R": 0.05,
    "J27_HEAD_PITCH": 0.2,
    "J28_HEAD_YAW": 0.2,
}

T800_ACTION_OFFSET = {
    "J00_HIP_PITCH_L": DEFAULT_Q_HIP_PITCH,
    "J01_HIP_ROLL_L": DEFAULT_Q_HIP_ROLL,
    "J02_HIP_YAW_L": DEFAULT_Q_HIP_YAW,
    "J03_KNEE_PITCH_L": DEFAULT_Q_KNEE_PITCH,
    "J04_ANKLE_PITCH_L": DEFAULT_Q_ANKLE_PITCH,
    "J05_ANKLE_ROLL_L": DEFAULT_Q_ANKLE_ROLL,
    "J06_HIP_PITCH_R": DEFAULT_Q_HIP_PITCH,
    "J07_HIP_ROLL_R": DEFAULT_Q_HIP_ROLL,
    "J08_HIP_YAW_R": DEFAULT_Q_HIP_YAW,
    "J09_KNEE_PITCH_R": DEFAULT_Q_KNEE_PITCH,
    "J10_ANKLE_PITCH_R": DEFAULT_Q_ANKLE_PITCH,
    "J11_ANKLE_ROLL_R": DEFAULT_Q_ANKLE_ROLL,
    "J12_TORSO_YAW": DEFAULT_Q_TORSO_YAW,
    "J13_SHOULDER_PITCH_L": DEFAULT_Q_SHOULDER_PITCH,
    "J14_SHOULDER_ROLL_L": DEFAULT_Q_SHOULDER_ROLL_L,
    "J15_SHOULDER_YAW_L": DEFAULT_Q_SHOULDER_YAW,
    "J16_ELBOW_PITCH_L": DEFAULT_Q_ELBOW_PITCH,
    "J17_ELBOW_YAW_L": DEFAULT_Q_ELBOW_YAW,
    "J20_SHOULDER_PITCH_R": DEFAULT_Q_SHOULDER_PITCH,
    "J21_SHOULDER_ROLL_R": DEFAULT_Q_SHOULDER_ROLL_R,
    "J22_SHOULDER_YAW_R": DEFAULT_Q_SHOULDER_YAW,
    "J23_ELBOW_PITCH_R": DEFAULT_Q_ELBOW_PITCH,
    "J24_ELBOW_YAW_R": DEFAULT_Q_ELBOW_YAW,
    "J27_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
    "J28_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
}
