"""T800 get-up recovery experiment configurations."""

from isaaclab.utils import configclass

from whole_body_tracking.robots.t800 import T800_MUJOCO_COLLISION_CFG
from whole_body_tracking.tasks.tracking.mdp.getup_recovery import apply_getup_recovery_experiment

from .flat_env_cfg import T800FlatGetUpSmoothEnvCfg


class _T800GetUpRecoveryBaseCfg(T800FlatGetUpSmoothEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Restore the trusted j9519 reward baseline while retaining every other
        # setting from the current smooth get-up configuration.
        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.getup_anchor_height.weight = 1.25
        self.rewards.getup_anchor_height.params["phase_end"] = 0.90
        self.rewards.getup_transition_joint_pos.weight = 1.5
        for reward_name in (
            "getup_final_anchor_pos",
            "getup_final_anchor_ori",
            "getup_final_anchor_height",
        ):
            setattr(self.rewards, reward_name, None)


@configclass
class T800GetUpRecoveryOriEnvCfg(_T800GetUpRecoveryBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_recovery_experiment(self, "ori")


@configclass
class T800GetUpRecoveryOriVelGateEnvCfg(_T800GetUpRecoveryBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_recovery_experiment(self, "ori_velgate")


@configclass
class T800GetUpMujocoCollisionRecoveryOriEnvCfg(_T800GetUpRecoveryBaseCfg):
    """Fresh get-up training task with the native SDK collision primitives.

    This task intentionally retains the proven recovery-orientation reward
    configuration.  Its only physics-asset change is replacing every URDF
    collision primitive with the MuJoCo SDK profile.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = T800_MUJOCO_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        apply_getup_recovery_experiment(self, "ori")
