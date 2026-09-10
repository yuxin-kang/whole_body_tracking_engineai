"""SDK get-up randomization; reset perturbations remain reference-relative."""
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg


def apply_sdk_getup_randomization(cfg):
    cfg.events.sdk_contact_offsets = EventTermCfg(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params=dict(asset_cfg=SceneEntityCfg("robot"),
                    rest_offset_distribution_params=None,
                    contact_offset_distribution_params=(0.006, 0.014),
                    distribution="uniform"),
    )
    cfg.events.sdk_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params=dict(asset_cfg=SceneEntityCfg("robot", body_names="LINK_BASE"),
                    mass_distribution_params=(-5.0, 5.0), operation="add"),
    )
    motion = cfg.commands.motion
    motion.joint_position_scale_range = (0.8, 1.2)
    motion.joint_velocity_range = (-1.0, 1.0)
    motion.pose_range = dict(x=(-0.02, 0.02), y=(-0.02, 0.02), z=(0.0, 0.03),
                             roll=(-0.03, 0.03), pitch=(-0.03, 0.03), yaw=(-0.1, 0.1))
    motion.velocity_range = dict(x=(-0.1, 0.1), y=(-0.1, 0.1), z=(-0.1, 0.1),
                                 roll=(-0.2, 0.2), pitch=(-0.2, 0.2), yaw=(-0.2, 0.2))
