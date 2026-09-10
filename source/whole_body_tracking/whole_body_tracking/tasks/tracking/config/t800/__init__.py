import gymnasium as gym

from . import agents, flat_env_cfg, getup_aggressive_env_cfg, getup_forearm_env_cfg, getup_recovery_env_cfg, mixed_wall_env_cfg
from . import sdk_supine_env_cfg
from . import sdk_prone_env_cfg

for _prone_variant in ('V3', 'V4'):
    _prone_method = 'PPO'
    gym.register(
        id=f'Tracking-Flat-T800-GetUp-SDK-Prone-{_prone_variant}-{_prone_method}-v0',
        entry_point='isaaclab.envs:ManagerBasedRLEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': getattr(sdk_prone_env_cfg, f'T800SDKProne{_prone_variant}{_prone_method}EnvCfg'),
            'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg',
        },
    )

for _supine_variant in ('V1', 'V2', 'V3', 'V4'):
    _supine_method = 'PPO'
    gym.register(
        id=f'Tracking-Flat-T800-GetUp-SDK-Supine-{_supine_variant}-{_supine_method}-v0',
        entry_point='isaaclab.envs:ManagerBasedRLEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': getattr(sdk_supine_env_cfg, f'T800SDKSupine{_supine_variant}{_supine_method}EnvCfg'),
            'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg',
        },
    )

for _kick_name, _kick_cfg in (
    ("LeftFrontKick", mixed_wall_env_cfg.T800MixedWallLeftFrontKickEnvCfg),
    ("RoundhouseKick", mixed_wall_env_cfg.T800MixedWallRoundhouseKickEnvCfg),
):
    gym.register(
        id=f"Tracking-Flat-T800-Target-MixedWall-{_kick_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _kick_cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
        },
    )

for _variant in "ABCD":
    gym.register(
        id=f"Tracking-Flat-T800-GetUp-Aggressive-{_variant}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(getup_aggressive_env_cfg, f"T800GetUpAggressive{_variant}EnvCfg"),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg",
        },
    )

for _forearm_variant in ("F1", "F2", "F3", "F4"):
    gym.register(
        id=f"Tracking-Flat-T800-GetUp-Aggressive-C-Forearm-{_forearm_variant}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(getup_forearm_env_cfg, f"T800GetUpAggressiveCForearm{_forearm_variant}EnvCfg"),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg",
        },
    )

gym.register(
    id="Tracking-Flat-T800-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-GetUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatGetUpEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-GetUp-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatGetUpSmoothEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-GetUp-Recovery-Ori-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": getup_recovery_env_cfg.T800GetUpRecoveryOriEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-GetUp-Recovery-OriVelGate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": getup_recovery_env_cfg.T800GetUpRecoveryOriVelGateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-GetUp-MujocoCollision-Recovery-Ori-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": getup_recovery_env_cfg.T800GetUpMujocoCollisionRecoveryOriEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-StraightPunchL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineStraightPunchLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-StraightPunchR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineStraightPunchREnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-LeftHook-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineLeftHookEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-RearHook-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineRearHookEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-LeftFrontKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineLeftFrontKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-RoundhouseKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineRoundhouseKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-LeftFrontKick-CompleteEpisode-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineLeftFrontKickCompleteEpisodeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-LeftFrontKick-CompleteEpisode-HeightFallOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineLeftFrontKickCompleteEpisodeHeightFallOnlyEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Baseline-RoundhouseKick-CompleteEpisode-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatBaselineRoundhouseKickCompleteEpisodeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Improved-StraightPunchL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatImprovedStraightPunchLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Improved-StraightPunchR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatImprovedStraightPunchREnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Improved-LeftHook-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatImprovedLeftHookEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Improved-RoundhouseKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatImprovedRoundhouseKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Improved-RoundhouseKick-StableRecovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatImprovedRoundhouseKickStableRecoveryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Target-StraightPunchL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatTargetStraightPunchLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Target-StraightPunchR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatTargetStraightPunchREnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Target-LeftHook-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatTargetLeftHookEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Target-LeftFrontKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatTargetLeftFrontKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Target-RoundhouseKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatTargetRoundhouseKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-Start0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1Start0EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-NoEarlyTerminations-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1NoEarlyTerminationsEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-OrigEpisode-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1OrigEpisodeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-KickVelKickPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickVelKickPosEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-KickJointLate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickJointLateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Zhiquan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatZhiquanEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Zhiquan-OrigEpisode-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatZhiquanOrigEpisodeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Zhiquan-Bridge-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatZhiquanBridgeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)
