from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from .buffer import FastSACBatch
from .config import FastSACAgentCfg
from .networks import SquashedGaussianActor, TwinQCritic
from .normalization import RunningObservationNormalizer


class FastSACAgent:
    """FastSAC agent with paper-aligned observation normalization and twin C51 critics."""

    CHECKPOINT_SCHEMA_VERSION = 2

    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, cfg: FastSACAgentCfg):
        self.cfg = cfg
        requested_device = torch.device(cfg.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.action_dim = action_dim
        self.actor_obs_normalizer = RunningObservationNormalizer(actor_obs_dim).to(self.device)
        self.critic_obs_normalizer = RunningObservationNormalizer(critic_obs_dim).to(self.device)
        self.actor = SquashedGaussianActor(
            actor_obs_dim,
            action_dim,
            cfg.actor_hidden_dims,
            activation=cfg.activation,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            use_layer_norm=cfg.use_layer_norm,
        ).to(self.device)
        self.critic = TwinQCritic(
            critic_obs_dim,
            action_dim,
            cfg.critic_hidden_dims,
            cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
            use_distributional_critic=cfg.use_distributional_critic,
            num_atoms=cfg.critic_num_atoms,
            v_min=cfg.critic_v_min,
            v_max=cfg.critic_v_max,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        optimizer_betas = (cfg.optimizer_beta1, cfg.optimizer_beta2)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=optimizer_betas,
            weight_decay=cfg.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=optimizer_betas,
            weight_decay=cfg.weight_decay,
        )
        self.log_alpha = torch.tensor(float(cfg.init_alpha)).log().to(self.device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr, betas=optimizer_betas)
        self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -(0.5 * float(action_dim))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _normalize_actor_obs(self, obs: torch.Tensor, *, update_stats: bool) -> torch.Tensor:
        obs = obs.to(self.device)
        if self.cfg.use_observation_normalization:
            if update_stats:
                self.actor_obs_normalizer.update(obs)
            obs = self.actor_obs_normalizer.normalize(obs)
        return obs

    def _normalize_critic_obs(self, obs: torch.Tensor, *, update_stats: bool) -> torch.Tensor:
        obs = obs.to(self.device)
        if self.cfg.use_observation_normalization:
            if update_stats:
                self.critic_obs_normalizer.update(obs)
            obs = self.critic_obs_normalizer.normalize(obs)
        return obs

    def observe(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        self._normalize_actor_obs(actor_obs, update_stats=True)
        self._normalize_critic_obs(critic_obs, update_stats=True)

    @torch.no_grad()
    def act(self, actor_obs: torch.Tensor, deterministic: bool = False, update_stats: bool = False) -> torch.Tensor:
        normalized_actor_obs = self._normalize_actor_obs(actor_obs, update_stats=update_stats)
        action, _ = self.actor.sample(normalized_actor_obs, deterministic=deterministic)
        return action

    def update(self, batch: FastSACBatch) -> dict[str, float]:
        actor_obs = batch.actor_obs.to(self.device)
        critic_obs = batch.critic_obs.to(self.device)
        action = batch.action.to(self.device)
        reward = batch.reward.to(self.device)
        done = batch.done.to(self.device).float()
        next_actor_obs = batch.next_actor_obs.to(self.device)
        next_critic_obs = batch.next_critic_obs.to(self.device)

        actor_obs = self._normalize_actor_obs(actor_obs, update_stats=True)
        critic_obs = self._normalize_critic_obs(critic_obs, update_stats=True)
        next_actor_obs = self._normalize_actor_obs(next_actor_obs, update_stats=True)
        next_critic_obs = self._normalize_critic_obs(next_critic_obs, update_stats=True)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_actor_obs)
            target_q1, target_q2 = self.target_critic(next_critic_obs, next_action)
            target_q = TwinQCritic.mean_q(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            backup = reward + self.cfg.gamma * (1.0 - done) * target_q

        if self.cfg.use_distributional_critic:
            target_logits1, target_logits2 = self.target_critic.dist(next_critic_obs, next_action)
            target_probs = 0.5 * (
                torch.softmax(target_logits1, dim=-1) + torch.softmax(target_logits2, dim=-1)
            )
            projected_target = self._project_distribution(
                reward=reward,
                done=done,
                target_probs=target_probs,
                entropy_term=self.alpha.detach() * next_log_prob,
            )
            q1_logits, q2_logits = self.critic.dist(critic_obs, action)
            critic_loss = self._categorical_loss(q1_logits, projected_target) + self._categorical_loss(
                q2_logits, projected_target
            )
        else:
            q1, q2 = self.critic(critic_obs, action)
            critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(actor_obs)
        q1_pi, q2_pi = self.critic(critic_obs, new_action)
        actor_loss = (self.alpha.detach() * log_prob - TwinQCritic.mean_q(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update_targets(self.cfg.tau)

        with torch.no_grad():
            expected_q1, expected_q2 = self.critic(critic_obs, action)
            mean_q_val = float((0.5 * (expected_q1 + expected_q2)).mean().cpu())
            target_q_val = float(target_q.mean().cpu())

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "mean_q": mean_q_val,
            "target_q_mean": target_q_val,
        }

    def _project_distribution(
        self,
        *,
        reward: torch.Tensor,
        done: torch.Tensor,
        target_probs: torch.Tensor,
        entropy_term: torch.Tensor,
    ) -> torch.Tensor:
        support = self.target_critic.support
        delta_z = (support[-1] - support[0]) / float(self.cfg.critic_num_atoms - 1)
        tz = reward + self.cfg.gamma * (1.0 - done) * (support.view(1, -1) - entropy_term)
        tz = tz.clamp(min=self.cfg.critic_v_min, max=self.cfg.critic_v_max)
        b = (tz - self.cfg.critic_v_min) / delta_z
        lower = b.floor().to(dtype=torch.long)
        upper = b.ceil().to(dtype=torch.long)
        lower = lower.clamp(min=0, max=self.cfg.critic_num_atoms - 1)
        upper = upper.clamp(min=0, max=self.cfg.critic_num_atoms - 1)
        projected = torch.zeros_like(target_probs)
        batch_index = torch.arange(projected.shape[0], device=projected.device)
        for atom in range(self.cfg.critic_num_atoms):
            lower_idx = lower[:, atom]
            upper_idx = upper[:, atom]
            lower_mass = target_probs[:, atom] * (upper[:, atom].to(dtype=torch.float32) - b[:, atom])
            upper_mass = target_probs[:, atom] * (b[:, atom] - lower[:, atom].to(dtype=torch.float32))
            same_bucket = lower_idx == upper_idx
            projected[batch_index, lower_idx] += torch.where(
                same_bucket,
                target_probs[:, atom],
                lower_mass,
            )
            projected[batch_index, upper_idx] += torch.where(
                same_bucket,
                torch.zeros_like(upper_mass),
                upper_mass,
            )
        return projected / torch.clamp(projected.sum(dim=-1, keepdim=True), min=1.0e-8)

    @staticmethod
    def _categorical_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        return -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    @torch.no_grad()
    def soft_update_targets(self, tau: float) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def checkpoint_state(self) -> dict:
        return {
            "algorithm": "FastSAC",
            "checkpoint_schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict(),
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict(),
            "critic_distribution": {
                "enabled": self.cfg.use_distributional_critic,
                "num_atoms": self.cfg.critic_num_atoms,
                "v_min": self.cfg.critic_v_min,
                "v_max": self.cfg.critic_v_max,
            },
            "config": self.cfg.to_dict(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        if state.get("algorithm") != "FastSAC":
            raise ValueError("Checkpoint is not a FastSAC checkpoint")
        if int(state.get("checkpoint_schema_version", 0)) < self.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported FastSAC checkpoint schema; strict paper-aligned checkpoints require observation normalizers"
            )
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_obs_normalizer.load_state_dict(state["actor_obs_normalizer"])
        self.critic_obs_normalizer.load_state_dict(state["critic_obs_normalizer"])
        self._move_optimizer_state_to_device(self.actor_optimizer)
        self._move_optimizer_state_to_device(self.critic_optimizer)
        self._move_optimizer_state_to_device(self.alpha_optimizer)

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
