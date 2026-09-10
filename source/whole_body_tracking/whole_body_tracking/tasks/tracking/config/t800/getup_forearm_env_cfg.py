"""C get-up candidate with brief forearm support guidance."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.mdp.getup_forearm_support import (
    getup_forearm_vertical_reward,
    getup_wrist_support_slip_penalty,
)

from .getup_aggressive_env_cfg import T800GetUpAggressiveCEnvCfg


FOREARM_EXPERIMENTS = {
    "f1": (1.0, 0.0),
    "f2": (3.0, 0.0),
    "f3": (1.0, -0.5),
    "f4": (3.0, -0.5),
}


def apply_getup_forearm_experiment(env_cfg, variant: str):
    """Apply the 2x2 vertical-strength / slip ablation to a C baseline snapshot."""
    if variant not in FOREARM_EXPERIMENTS:
        raise ValueError(f"Unknown forearm experiment: {variant}")
    if env_cfg.commands.motion.aggressive_variant != "c":
        raise ValueError("Forearm experiments require the aggressive C baseline")
    vertical_weight, slip_weight = FOREARM_EXPERIMENTS[variant]
    sensor_names = []
    for side in ("L", "R"):
        for link in ("ELBOW_YAW", "WRIST_END"):
            name = f"getup_ground_{link.lower()}_{side.lower()}"
            setattr(
                env_cfg.scene,
                name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/LINK_{link}_{side}",
                    filter_prim_paths_expr=[f"{env_cfg.scene.terrain.prim_path}/terrain/GroundPlane/CollisionPlane"],
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                ),
            )
            sensor_names.append(name)
    shared_params = {
        "command_name": "motion",
        "wrist_cfg": SceneEntityCfg(
            "robot", body_names=["LINK_WRIST_END_L", "LINK_WRIST_END_R"], preserve_order=True
        ),
        # Sphere centers relative to WRIST_END, from serial_t800.urdf.
        "wrist_offsets_b": ((0.026, -0.009, -0.060), (0.026, 0.009, -0.060)),
        # 379-frame clip: fade in 0.60--0.91s; fade out 2.27--2.87s.
        "phase_window": (0.08, 0.12, 0.30, 0.38),
        "force_min": 5.0,
        "force_full": 30.0,
    }
    env_cfg.rewards.getup_forearm_vertical = RewTerm(
        func=getup_forearm_vertical_reward,
        weight=vertical_weight,
        params={
            **shared_params,
            "elbow_cfg": SceneEntityCfg(
                "robot", body_names=["LINK_ELBOW_PITCH_L", "LINK_ELBOW_PITCH_R"], preserve_order=True
            ),
            "ground_sensor_names": tuple(sensor_names),
            "std": 0.35,
        },
    )
    env_cfg.rewards.getup_wrist_support_slip = RewTerm(
        func=getup_wrist_support_slip_penalty,
        weight=slip_weight,
        params={
            **shared_params,
            "wrist_sensor_names": (sensor_names[1], sensor_names[3]),
            "speed_std": 0.20,
            "sphere_radius": 0.05,
        },
    )
    return env_cfg


@configclass
class T800GetUpAggressiveCForearmF1EnvCfg(T800GetUpAggressiveCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_forearm_experiment(self, "f1")


@configclass
class T800GetUpAggressiveCForearmF2EnvCfg(T800GetUpAggressiveCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_forearm_experiment(self, "f2")


@configclass
class T800GetUpAggressiveCForearmF3EnvCfg(T800GetUpAggressiveCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_forearm_experiment(self, "f3")


@configclass
class T800GetUpAggressiveCForearmF4EnvCfg(T800GetUpAggressiveCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_forearm_experiment(self, "f4")
