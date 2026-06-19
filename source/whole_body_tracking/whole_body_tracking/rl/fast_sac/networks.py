from __future__ import annotations

import math

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU
    if normalized == "relu":
        return nn.ReLU
    if normalized == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    *,
    activation: str = "elu",
    use_layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    act = _activation(activation)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        last_dim = hidden_dim
    return nn.Sequential(*layers)


def count_hidden_linear_layers(module: nn.Module) -> int:
    return sum(1 for layer in module.modules() if isinstance(layer, nn.Linear))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = build_mlp(obs_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        trunk_dim = hidden_dims[-1] if hidden_dims else obs_dim
        self.head = nn.Linear(trunk_dim, action_dim * 2)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.head(self.trunk(obs)).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        if deterministic:
            pre_tanh = mean
        else:
            std = log_std.exp()
            pre_tanh = mean + std * torch.randn_like(std)
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std, action)
        return action, log_prob

    def _log_prob_from_pre_tanh(
        self,
        pre_tanh: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        std = log_std.exp()
        normal_log_prob = -0.5 * (((pre_tanh - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        normal_log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
        squash_correction = torch.log(1.0 - action.pow(2) + 1.0e-6).sum(dim=-1, keepdim=True)
        return normal_log_prob - squash_correction


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        *,
        activation: str = "elu",
        use_layer_norm: bool = True,
        num_atoms: int = 51,
    ):
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.trunk = build_mlp(obs_dim + action_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        trunk_dim = hidden_dims[-1] if hidden_dims else obs_dim + action_dim
        self.head = nn.Linear(trunk_dim, self.num_atoms)

    def logits(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self.trunk(torch.cat([obs, action], dim=-1))
        return self.head(features)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
        *,
        use_layer_norm: bool = True,
        use_distributional_critic: bool = True,
        num_atoms: int = 51,
        v_min: float = -50.0,
        v_max: float = 50.0,
    ):
        super().__init__()
        self.use_distributional_critic = bool(use_distributional_critic)
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.num_atoms, dtype=torch.float32))
        if self.use_distributional_critic:
            self.q1 = DistributionalQNetwork(
                obs_dim,
                action_dim,
                hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                num_atoms=self.num_atoms,
            )
            self.q2 = DistributionalQNetwork(
                obs_dim,
                action_dim,
                hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                num_atoms=self.num_atoms,
            )
        else:
            self.q1 = nn.Sequential(
                build_mlp(obs_dim + action_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm),
                nn.Linear(hidden_dims[-1] if hidden_dims else obs_dim + action_dim, 1),
            )
            self.q2 = nn.Sequential(
                build_mlp(obs_dim + action_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm),
                nn.Linear(hidden_dims[-1] if hidden_dims else obs_dim + action_dim, 1),
            )

    def dist(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_distributional_critic:
            raise RuntimeError("Distributional outputs are unavailable when use_distributional_critic is False")
        return self.q1.logits(obs, action), self.q2.logits(obs, action)

    def expected_q(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_distributional_critic:
            logits1, logits2 = self.dist(obs, action)
            return self.expected_value_from_logits(logits1, self.support), self.expected_value_from_logits(
                logits2, self.support
            )
        q_input = torch.cat([obs, action], dim=-1)
        return self.q1(q_input), self.q2(q_input)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expected_q(obs, action)

    @staticmethod
    def min_q(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(q1, q2)

    @staticmethod
    def mean_q(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return 0.5 * (q1 + q2)

    @staticmethod
    def expected_value_from_logits(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        values = torch.sum(probs * support.view(1, -1), dim=-1, keepdim=True)
        return values
