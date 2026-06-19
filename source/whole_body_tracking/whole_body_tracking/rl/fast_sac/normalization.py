from __future__ import annotations

import torch
from torch import nn


class RunningObservationNormalizer(nn.Module):
    def __init__(self, obs_dim: int, epsilon: float = 1.0e-6, clip_value: float = 5.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.clip_value = float(clip_value)
        self.register_buffer("mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1.0, dtype=torch.float32))

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        if obs.numel() == 0:
            return
        batch = obs.detach().to(device=self.mean.device, dtype=torch.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = float(batch.shape[0])
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=self.epsilon))
        self.count.copy_(torch.tensor(total_count, dtype=torch.float32, device=self.mean.device))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=obs.device, dtype=obs.dtype)
        std = torch.sqrt(self.var.to(device=obs.device, dtype=obs.dtype) + self.epsilon)
        normalized = (obs - mean) / std
        return torch.clamp(normalized, min=-self.clip_value, max=self.clip_value)
