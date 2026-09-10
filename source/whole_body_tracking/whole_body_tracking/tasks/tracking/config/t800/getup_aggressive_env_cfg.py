"""Independent from-scratch candidates; do not change the running recovery tasks."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.mdp.getup_aggressive_commands import (
    configure_aggressive_commands,
    getup_delayed_failure,
    getup_motion_timeout,
)
from whole_body_tracking.tasks.tracking.mdp.getup_aggressive_rewards import apply_aggressive_rewards

from .getup_recovery_env_cfg import _T800GetUpRecoveryBaseCfg


def apply_getup_aggressive_experiment(env_cfg, variant):
    if variant not in {"a", "b", "c", "d"}:
        raise ValueError(f"Unknown aggressive get-up variant: {variant}")
    apply_aggressive_rewards(env_cfg, variant)
    configure_aggressive_commands(env_cfg, variant)
    if variant in {"c", "d"}:
        env_cfg.terminations.time_out = DoneTerm(
            func=getup_motion_timeout, time_out=True, params={"command_name": "motion"}
        )
    if variant == "d":
        env_cfg.terminations.getup_aggressive_failure = DoneTerm(
            func=getup_delayed_failure, time_out=False, params={"command_name": "motion"}
        )
    return env_cfg


@configclass
class T800GetUpAggressiveAEnvCfg(_T800GetUpRecoveryBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_aggressive_experiment(self, "a")


@configclass
class T800GetUpAggressiveBEnvCfg(_T800GetUpRecoveryBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_aggressive_experiment(self, "b")


@configclass
class T800GetUpAggressiveCEnvCfg(_T800GetUpRecoveryBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_aggressive_experiment(self, "c")


@configclass
class T800GetUpAggressiveDEnvCfg(_T800GetUpRecoveryBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_getup_aggressive_experiment(self, "d")
