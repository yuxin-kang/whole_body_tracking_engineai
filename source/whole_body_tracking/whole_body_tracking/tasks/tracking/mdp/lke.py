from __future__ import annotations

import torch


def compute_reference_joint_velocity_energy(joint_vel: torch.Tensor) -> torch.Tensor:
    if joint_vel.ndim != 2:
        raise ValueError("joint_vel must have shape (T, J)")
    # Paper-aligned LKES energy proxy: E = sum |q̇| (L1), not sum q̇^2.
    return torch.sum(torch.abs(joint_vel), dim=-1)


def detect_lke_anchor_indexes(kinetic_energy: torch.Tensor, minimum_window: int = 1) -> torch.Tensor:
    """Detect local-minimum anchors from a 1-D reference kinetic-energy proxy."""
    if kinetic_energy.ndim != 1:
        raise ValueError("kinetic_energy must be a 1-D tensor")
    if kinetic_energy.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=kinetic_energy.device)
    if kinetic_energy.numel() <= 2 * minimum_window:
        return torch.arange(kinetic_energy.numel(), dtype=torch.long, device=kinetic_energy.device)

    anchors: list[int] = []
    for idx in range(int(minimum_window), int(kinetic_energy.numel() - minimum_window)):
        center = kinetic_energy[idx]
        left = kinetic_energy[idx - minimum_window : idx]
        right = kinetic_energy[idx + 1 : idx + 1 + minimum_window]
        if torch.all(center <= left) and torch.all(center <= right):
            anchors.append(idx)
    if not anchors:
        return torch.argmin(kinetic_energy).reshape(1)
    return torch.tensor(anchors, dtype=torch.long, device=kinetic_energy.device)


def initialize_lke_anchor_weights(anchor_indexes: torch.Tensor, initial_weight: float = 1.0) -> torch.Tensor:
    if anchor_indexes.numel() == 0:
        raise ValueError("anchor_indexes must not be empty")
    return torch.full((anchor_indexes.numel(),), float(initial_weight), dtype=torch.float32, device=anchor_indexes.device)


def update_lke_anchor_weights(
    anchor_weights: torch.Tensor,
    failed_steps: torch.Tensor,
    anchor_indexes: torch.Tensor,
    *,
    alpha: float,
    w_min: float,
    w_max: float,
) -> torch.Tensor:
    if anchor_indexes.numel() == 0:
        raise ValueError("anchor_indexes must not be empty")
    updated = anchor_weights.clone()
    if failed_steps.numel() == 0:
        return updated.clamp_(min=w_min, max=w_max)
    failed_anchor_positions = assign_lke_failure_to_previous_anchor(failed_steps, anchor_indexes)
    updated += torch.bincount(
        failed_anchor_positions,
        minlength=anchor_indexes.numel(),
    ).to(device=anchor_weights.device, dtype=anchor_weights.dtype) * float(alpha)
    return updated.clamp_(min=w_min, max=w_max)


def compute_lke_anchor_probabilities(anchor_weights: torch.Tensor) -> torch.Tensor:
    if anchor_weights.ndim != 1 or anchor_weights.numel() == 0:
        raise ValueError("anchor_weights must be a non-empty 1-D tensor")
    positive = anchor_weights.clamp_min(0.0)
    return positive / torch.clamp(positive.sum(), min=1.0e-8)


def assign_lke_failure_to_previous_anchor(failed_steps: torch.Tensor, anchor_indexes: torch.Tensor) -> torch.Tensor:
    """Map failed frame indexes to nearest preceding anchor positions for runtime bincount credit."""
    if anchor_indexes.numel() == 0:
        raise ValueError("anchor_indexes must not be empty")
    insert_positions = torch.searchsorted(anchor_indexes, failed_steps, right=True) - 1
    insert_positions = torch.clamp(insert_positions, min=0, max=anchor_indexes.numel() - 1)
    return insert_positions.to(dtype=torch.long)
