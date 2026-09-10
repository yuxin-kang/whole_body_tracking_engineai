"""Independent supine reward ablations for direct PPO tasks.

    151 frames at 50 Hz: floor phase 0--0.50, main rise 0.50--0.80,
    near upright 0.80 onward. All windows use native reference phase.
"""

from pathlib import Path
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .flat_env_cfg import T800FlatGetUpEnvCfg
from .sdk_getup_randomization import apply_sdk_getup_randomization
from whole_body_tracking.tasks.tracking.mdp.getup_aggressive_rewards import (
    aggressive_anchor_height_error_exp, aggressive_standing_quality_reward,
)
from whole_body_tracking.tasks.tracking.mdp.getup_recovery import (
    getup_transition_anchor_orientation_error_exp,
    getup_pose_gated_body_linear_velocity_error_exp,
    getup_pose_gated_body_angular_velocity_error_exp,
)
from whole_body_tracking.tasks.tracking.mdp.sdk_supine_rewards import supine_support_slip


REPO = Path(__file__).resolve().parents[7]
MOTION = REPO / 'data/npz/t800_sdk_get_up/T800_supine_to_stance_sdk_clip_50hz.npz'


def apply_supine_shaping(cfg, variant):
    if variant not in ('V1', 'V2', 'V3', 'V4'):
        raise ValueError(variant)
    cfg.rewards.sdk_supine_height = RewTerm(
        func=aggressive_anchor_height_error_exp, weight=1.25,
        params=dict(command_name='motion', std=0.18, phase_start=0.50, phase_end=0.55))
    cfg.rewards.sdk_supine_anchor_ori = RewTerm(
        func=getup_transition_anchor_orientation_error_exp, weight=0.5,
        params=dict(command_name='motion', std=0.4, phase_start=0.35, phase_end=0.45))
    if variant in ('V2', 'V3', 'V4'):
        for name, func in (
            ('motion_body_lin_vel', getup_pose_gated_body_linear_velocity_error_exp),
            ('motion_body_ang_vel', getup_pose_gated_body_angular_velocity_error_exp),
        ):
            term = getattr(cfg.rewards, name)
            term.func = func
            term.params.update(phase_start=0.75, phase_end=0.83,
                               height_std=0.25, orientation_std=0.60, floor=0.10)
    if variant in ('V3', 'V4'):
        cfg.rewards.sdk_supine_standing = RewTerm(
            func=aggressive_standing_quality_reward, weight=2.0,
            params=dict(command_name='motion', phase_start=0.82, phase_end=0.90))
    if variant == 'V4':
        sensors = []
        for side in ('L', 'R'):
            name = f'sdk_supine_foot_ground_{side.lower()}'
            sensors.append(name)
            setattr(cfg.scene, name, ContactSensorCfg(
                prim_path=f'{{ENV_REGEX_NS}}/Robot/LINK_ANKLE_ROLL_{side}',
                filter_prim_paths_expr=[f'{cfg.scene.terrain.prim_path}/terrain/GroundPlane/CollisionPlane'],
                update_period=0.0, history_length=0, debug_vis=False))
        cfg.rewards.sdk_supine_foot_slip = RewTerm(
            func=supine_support_slip, weight=-0.5,
            params=dict(command_name='motion', sensor_names=tuple(sensors),
                        foot_cfg=SceneEntityCfg('robot', body_names=['LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R'],
                                                preserve_order=True),
                        phase_start=0.78, phase_end=0.84, speed_std=0.20))


@configclass
class _SDKSupineBaseCfg(T800FlatGetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(MOTION)
        self.episode_length_s = 151 / 50
        apply_sdk_getup_randomization(self)


@configclass
class T800SDKSupineV1PPOEnvCfg(_SDKSupineBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_supine_shaping(self, 'V1')


@configclass
class T800SDKSupineV2PPOEnvCfg(_SDKSupineBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_supine_shaping(self, 'V2')


@configclass
class T800SDKSupineV3PPOEnvCfg(_SDKSupineBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_supine_shaping(self, 'V3')


@configclass
class T800SDKSupineV4PPOEnvCfg(_SDKSupineBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_supine_shaping(self, 'V4')
