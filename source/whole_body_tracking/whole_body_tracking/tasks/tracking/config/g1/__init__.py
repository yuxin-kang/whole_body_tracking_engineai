import gymnasium as gym

from . import agents, flat_env_cfg, paper_full_env_cfg


gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Standing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatStandingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-Stage-I-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat1307StageIEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-Stage-II-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat1307StageIIEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-Stage-III-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat1307StageIIIEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-Checkpoint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat1307CheckpointEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": paper_full_env_cfg.G1Flat1307PaperFastSACStageIEnvCfg,
        "fast_sac_cfg_entry_point": f"{agents.__name__}.fast_sac_cfg:G1PaperFastSACRunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": paper_full_env_cfg.G1Flat1307PaperFastSACStageIIEnvCfg,
        "fast_sac_cfg_entry_point": f"{agents.__name__}.fast_sac_cfg:G1PaperFastSACRunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": paper_full_env_cfg.G1Flat1307PaperFastSACStageIIIEnvCfg,
        "fast_sac_cfg_entry_point": f"{agents.__name__}.fast_sac_cfg:G1PaperFastSACRunnerCfg",
    },
)

# A1: single-stage paper-aligned task (replaces the Stage I/II/III curriculum chain).
gym.register(
    id="Tracking-Flat-G1-1307-PaperFastSAC-Unified-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": paper_full_env_cfg.G1Flat1307PaperFastSACUnifiedEnvCfg,
        "fast_sac_cfg_entry_point": f"{agents.__name__}.fast_sac_cfg:G1PaperFastSACRunnerCfg",
    },
)
