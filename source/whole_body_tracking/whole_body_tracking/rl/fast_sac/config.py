from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FastSACAgentCfg:
    """Configuration for the paper-only Isaac FastSAC implementation."""

    seed: int = 42
    device: str = "cuda:0"
    experiment_name: str = "g1_paper_fast_sac"
    run_name: str = ""
    max_steps: int = 30_000
    warmup_steps: int = 10  # paper-aligned: ~learning_starts magnitude (was 10_000)
    updates_per_step: int = 2  # paper-aligned UTD=2 (was 4)
    replay_size: int = 204_800  # FastTD3-style short buffer: ~50 recent steps x 4096 envs (was 1_000_000); scale with num_envs
    batch_size: int = 32_768  # paper-aligned FastSAC reference default (was 256)
    gamma: float = 0.99
    tau: float = 0.1  # paper-aligned FastSAC default (was 0.005)
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    weight_decay: float = 1.0e-3
    target_entropy: float | None = None
    init_alpha: float = 0.001
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512])  # paper ref: 3x512 (was [512,256,128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024, 1024, 1024])  # paper ref: 4x1024 (was [512,256,256,128])
    activation: str = "elu"
    log_std_min: float = -20.0
    log_std_max: float = 0.0
    use_layer_norm: bool = True
    use_observation_normalization: bool = True
    use_distributional_critic: bool = True
    critic_num_atoms: int = 101  # paper-aligned (was 51)
    critic_v_min: float = -250.0  # paper-aligned symmetric reference support (was -50.0)
    critic_v_max: float = 250.0  # symmetric: good tracking policy earns large positive return (was 50.0)
    checkpoint_interval: int = 5_000
    save_replay_buffer: bool = False
    resume_checkpoint: str | None = None
    source_note: str = "Paper-only FastSAC training reproduction for arXiv:2602.13656 with arXiv:2512.01996 recipe defaults."

    def to_dict(self) -> dict:
        return asdict(self)
