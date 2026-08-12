from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def _actor_critic_cfg(**kwargs) -> RslRlPpoActorCriticCfg:
    """Keep per-network normalization on versions that expose those fields."""
    supported_fields = {
        field_name
        for config_type in RslRlPpoActorCriticCfg.__mro__
        for field_name in getattr(config_type, "__annotations__", {})
    }
    for field_name in ("actor_obs_normalization", "critic_obs_normalization"):
        if field_name not in supported_fields:
            kwargs.pop(field_name, None)
    return RslRlPpoActorCriticCfg(**kwargs)


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30001
    save_interval = 500
    experiment_name = "g1_tracking"
    empirical_normalization = True
    policy = _actor_critic_cfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
