from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FastSACObs:
    actor: torch.Tensor
    critic: torch.Tensor


def split_actor_critic_obs(obs) -> FastSACObs:
    """Extract policy and critic tensors from IsaacLab/Gym observation outputs."""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "policy" in obs:
            actor_obs = obs["policy"]
            critic_obs = obs.get("critic", actor_obs)
            return FastSACObs(actor=actor_obs, critic=critic_obs)
        if "obs" in obs:
            return split_actor_critic_obs(obs["obs"])
    if hasattr(obs, "keys") and hasattr(obs, "__getitem__"):
        keys = set(obs.keys())
        if "policy" in keys:
            actor_obs = obs["policy"]
            critic_obs = obs["critic"] if "critic" in keys else actor_obs
            return FastSACObs(actor=actor_obs, critic=critic_obs)
        if "obs" in keys:
            return split_actor_critic_obs(obs["obs"])
    if torch.is_tensor(obs):
        return FastSACObs(actor=obs, critic=obs)
    raise TypeError(f"Unsupported observation type for FastSAC: {type(obs)!r}")
