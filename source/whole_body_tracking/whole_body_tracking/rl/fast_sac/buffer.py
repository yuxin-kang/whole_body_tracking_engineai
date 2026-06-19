from __future__ import annotations

from dataclasses import dataclass

import torch


FAST_SAC_TRANSITION_KEYS = (
    "actor_obs",
    "critic_obs",
    "action",
    "reward",
    "terminated",
    "time_out",
    "done",
    "next_actor_obs",
    "next_critic_obs",
)


@dataclass(frozen=True)
class FastSACBatch:
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    time_out: torch.Tensor
    done: torch.Tensor
    next_actor_obs: torch.Tensor
    next_critic_obs: torch.Tensor


class FastSACReplayBuffer:
    """Fixed-size replay buffer for vectorized off-policy IsaacLab rollouts."""

    def __init__(
        self,
        capacity: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.pos = 0
        self.full = False
        self.actor_obs = torch.zeros((capacity, actor_obs_dim), device=self.device)
        self.critic_obs = torch.zeros((capacity, critic_obs_dim), device=self.device)
        self.action = torch.zeros((capacity, action_dim), device=self.device)
        self.reward = torch.zeros((capacity, 1), device=self.device)
        self.terminated = torch.zeros((capacity, 1), dtype=torch.bool, device=self.device)
        self.time_out = torch.zeros((capacity, 1), dtype=torch.bool, device=self.device)
        self.done = torch.zeros((capacity, 1), dtype=torch.bool, device=self.device)
        self.next_actor_obs = torch.zeros((capacity, actor_obs_dim), device=self.device)
        self.next_critic_obs = torch.zeros((capacity, critic_obs_dim), device=self.device)

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos

    def add_batch(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        time_out: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
    ) -> None:
        batch_size = actor_obs.shape[0]
        if batch_size <= 0:
            return
        if batch_size >= self.capacity:
            source_indexes = torch.arange(batch_size - self.capacity, batch_size, device=actor_obs.device)
            target_indexes = torch.arange(self.capacity, device=self.device)
            self.pos = 0
            self.full = True
        else:
            source_indexes = torch.arange(batch_size, device=actor_obs.device)
            target_indexes = (torch.arange(batch_size, device=self.device) + self.pos) % self.capacity
            self.full = self.full or self.pos + batch_size >= self.capacity
            self.pos = (self.pos + batch_size) % self.capacity

        done = terminated.bool()
        values = {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "action": action,
            "reward": reward.reshape(batch_size, 1),
            "terminated": terminated.reshape(batch_size, 1).bool(),
            "time_out": time_out.reshape(batch_size, 1).bool(),
            "done": done.reshape(batch_size, 1).bool(),
            "next_actor_obs": next_actor_obs,
            "next_critic_obs": next_critic_obs,
        }
        for key, value in values.items():
            getattr(self, key)[target_indexes] = value[source_indexes].to(self.device)

    def sample(self, batch_size: int) -> FastSACBatch:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        indexes = torch.randint(0, self.size, (batch_size,), device=self.device)
        return FastSACBatch(
            actor_obs=self.actor_obs[indexes],
            critic_obs=self.critic_obs[indexes],
            action=self.action[indexes],
            reward=self.reward[indexes],
            terminated=self.terminated[indexes],
            time_out=self.time_out[indexes],
            done=self.done[indexes],
            next_actor_obs=self.next_actor_obs[indexes],
            next_critic_obs=self.next_critic_obs[indexes],
        )

    def state_dict(self, include_storage: bool = False) -> dict:
        state = {
            "capacity": self.capacity,
            "pos": self.pos,
            "full": self.full,
            "size": self.size,
            "keys": FAST_SAC_TRANSITION_KEYS,
        }
        if include_storage:
            storage_size = self.size
            state["storage"] = {
                key: getattr(self, key)[:storage_size].detach().cpu()
                for key in FAST_SAC_TRANSITION_KEYS
            }
        return state

    def load_state_dict(self, state: dict) -> None:
        storage = state.get("storage")
        if storage is None:
            self.pos = 0
            self.full = False
            return

        storage_size = int(state["size"])
        if storage_size > self.capacity:
            raise ValueError("Replay storage is larger than buffer capacity")
        self.pos = int(state["pos"])
        self.full = bool(state["full"])
        for key in FAST_SAC_TRANSITION_KEYS:
            if key not in storage:
                raise ValueError(f"Replay storage missing key: {key}")
            value = storage[key]
            getattr(self, key)[:storage_size] = value.to(self.device)
