"""Prone SDK-reference shaping for direct PPO tasks."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .flat_env_cfg import T800FlatGetUpEnvCfg
from .sdk_getup_randomization import apply_sdk_getup_randomization
from .sdk_supine_env_cfg import REPO
from whole_body_tracking.tasks.tracking.mdp.getup_aggressive_rewards import aggressive_anchor_height_error_exp
from whole_body_tracking.tasks.tracking.mdp.getup_recovery import getup_transition_anchor_orientation_error_exp
from whole_body_tracking.tasks.tracking.mdp.getup_recovery import (
    getup_pose_gated_body_linear_velocity_error_exp,
    getup_pose_gated_body_angular_velocity_error_exp,
)
from whole_body_tracking.tasks.tracking.mdp.getup_aggressive_rewards import aggressive_standing_quality_reward
from whole_body_tracking.tasks.tracking.mdp.sdk_supine_rewards import supine_support_slip


@configclass
class _T800SDKProneBaseCfg(T800FlatGetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(
            REPO / 'data/npz/t800_sdk_get_up/T800_prone_to_stance_sdk_clip_grounded_50hz.npz')
        self.episode_length_s = 143 / 50
        apply_sdk_getup_randomization(self)
        # Reference lifts around frame 60, then straightens around frame 100.
        # phase_end is the end of the ramp, not reward deactivation.
        self.rewards.sdk_prone_height = RewTerm(
            func=aggressive_anchor_height_error_exp, weight=1.25,
            params=dict(command_name='motion', std=0.18, phase_start=0.40, phase_end=0.45))
        self.rewards.sdk_prone_anchor_ori = RewTerm(
            func=getup_transition_anchor_orientation_error_exp, weight=0.5,
            params=dict(command_name='motion', std=0.4, phase_start=0.65, phase_end=0.72))


def apply_prone_extensions(cfg, variant):
    """Add cumulative V2--V4 terms after prone-specific V1 guidance."""
    if variant not in ('V2', 'V3', 'V4'):
        raise ValueError(variant)
    for name, func in (
        ('motion_body_lin_vel', getup_pose_gated_body_linear_velocity_error_exp),
        ('motion_body_ang_vel', getup_pose_gated_body_angular_velocity_error_exp),
    ):
        term = getattr(cfg.rewards, name)
        term.func = func
        term.params.update(phase_start=0.78, phase_end=0.88,
                           height_std=0.25, orientation_std=0.60, floor=0.10)
    if variant in ('V3', 'V4'):
        cfg.rewards.sdk_prone_standing = RewTerm(
            func=aggressive_standing_quality_reward, weight=2.0,
            params=dict(command_name='motion', phase_start=0.85, phase_end=0.93))
    if variant == 'V4':
        sensors = []
        for side in ('L', 'R'):
            name = f'sdk_prone_foot_ground_{side.lower()}'
            sensors.append(name)
            setattr(cfg.scene, name, ContactSensorCfg(
                prim_path=f'{{ENV_REGEX_NS}}/Robot/LINK_ANKLE_ROLL_{side}',
                filter_prim_paths_expr=[f'{cfg.scene.terrain.prim_path}/terrain/GroundPlane/CollisionPlane'],
                update_period=0.0, history_length=0, debug_vis=False))
        cfg.rewards.sdk_prone_foot_slip = RewTerm(
            func=supine_support_slip, weight=-0.5,
            params=dict(command_name='motion', sensor_names=tuple(sensors),
                        foot_cfg=SceneEntityCfg('robot', body_names=['LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R'],
                                               preserve_order=True),
                        phase_start=0.82, phase_end=0.90, speed_std=0.20))


@configclass
class T800SDKProneV3PPOEnvCfg(_T800SDKProneBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        # Direct PPO uses the training actuator gains from the base get-up task.
        self.scene.robot.actuators = T800FlatGetUpEnvCfg().scene.robot.actuators
        apply_prone_extensions(self, 'V3')


@configclass
class T800SDKProneV4PPOEnvCfg(T800SDKProneV3PPOEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_prone_extensions(self, 'V4')
