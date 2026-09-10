"""Independent scratch experiments: half wall strikes, half free kicks."""

from isaaclab.utils import configclass

from ...mdp.mixed_wall import (
    mixed_wall_contact_reward,
    mixed_wall_target_position_b,
    reset_mixed_wall_target,
)
from .flat_env_cfg import (
    T800_TARGET_STRIKE_SPECS,
    T800FlatTargetLeftFrontKickEnvCfg,
    T800FlatTargetRoundhouseKickEnvCfg,
    _apply_roundhouse_stable_recovery_improvements,
    _configure_target_strike,
)


# Keep the accepted front pad height/width, but move its front face to the
# full-clip extension envelope. The old face at x=1.731 m cuts 0.354 m into
# the demonstrated kick; x=2.0600544 m allows only 0.025 m of pad compression.
FRONT_KICK_MIXED_WALL_SPEC = {
    **T800_TARGET_STRIKE_SPECS["left_front_kick"],
    "target_position": (2.1500544, 0.034, 1.135),
    "contact_frames": (47, 50),
    "recovery_frame": 51,
}

# The active Isaac URDF ankle collision box (not the optional MuJoCo profile)
# reaches max world x=1.4742255 m at frame 61. Keep the front face 0.025 m
# inside that support point, then add the wall's 0.09 m half-thickness.
# A forward plane avoids blocking the remainder of the sideways sweep.
ROUNDHOUSE_MIXED_WALL_SPEC = {
    **T800_TARGET_STRIKE_SPECS["roundhouse_kick"],
    "target_position": (1.5392255, -0.500, 1.570),
    "target_yaw": 0.0,
    "contact_frames": (60, 64),
    "recovery_frame": 65,
}


def _configure_mixed_wall(env_cfg) -> None:
    # The shared plane stays at mu=1; sample the robot's contact materials so
    # separate environments can experience different effective floor friction.
    material = env_cfg.events.physics_material.params
    material["static_friction_range"] = (1.0, 1.7)
    material["dynamic_friction_range"] = (1.0, 1.7)
    material["make_consistent"] = True
    env_cfg.events.reset_target.func = reset_mixed_wall_target
    env_cfg.events.reset_target.params["absent_height"] = -10.0
    env_cfg.events.reset_target.params["strike_depth_range"] = (0.025, 0.10)
    env_cfg.observations.critic.target_pos_b.func = mixed_wall_target_position_b
    env_cfg.rewards.target_contact.func = mixed_wall_contact_reward


@configclass
class T800MixedWallLeftFrontKickEnvCfg(T800FlatTargetLeftFrontKickEnvCfg):
    """A shallow extension-envelope target and an equal free-kick cohort."""

    def __post_init__(self):
        super().__post_init__()
        _configure_target_strike(self, FRONT_KICK_MIXED_WALL_SPEC)
        _configure_mixed_wall(self)


@configclass
class T800MixedWallRoundhouseKickEnvCfg(T800FlatTargetRoundhouseKickEnvCfg):
    """A shallow forward target plus the established landing/recovery rewards."""

    def __post_init__(self):
        super().__post_init__()
        _configure_target_strike(self, ROUNDHOUSE_MIXED_WALL_SPEC)
        _apply_roundhouse_stable_recovery_improvements(self)
        _configure_mixed_wall(self)
