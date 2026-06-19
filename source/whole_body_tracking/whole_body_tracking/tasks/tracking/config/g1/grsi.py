from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .paper_contract import (
    G1_PAPER_EQUIVALENCE_CONTRACT_ID,
    G1_PAPER_EQUIVALENCE_MIN_GRSI_STATES,
)


GRSI_SCHEMA_VERSION = 1
GRSI_PRODUCTION_MODE = "gravity_release"
GRSI_PRODUCTION_SOURCE = "isaaclab-grsi:gravity-release"
GRSI_REQUIRED_KEYS = (
    "version",
    "robot_root_states_xyzw",
    "dof_pos",
    "dof_vel",
    "joint_names",
    "body_names",
    "source",
    "generation_config",
)


@dataclass(frozen=True)
class GRSIValidationReport:
    total: int
    accepted: int
    rejected: int


def _yaw_from_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
    quat[:, 2] = torch.sin(half)
    quat[:, 3] = torch.cos(half)
    return quat


def _quat_mul_xyzw(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = lhs.unbind(dim=-1)
    x2, y2, z2, w2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _rotate_xy_vectors(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    out = xy.clone()
    out[:, 0] = cos_yaw * xy[:, 0] - sin_yaw * xy[:, 1]
    out[:, 1] = sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]
    return out


def select_diverse_grsi_pool_indexes(
    root_states_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    num_select: int,
) -> torch.Tensor:
    count = int(root_states_xyzw.shape[0])
    num_select = min(int(num_select), count)
    if num_select <= 0:
        return torch.empty(0, dtype=torch.long, device=root_states_xyzw.device)
    if num_select >= count:
        return torch.arange(count, dtype=torch.long, device=root_states_xyzw.device)

    features = torch.cat([root_states_xyzw[:, 2:7], dof_pos], dim=1).to(dtype=torch.float32)
    features = features - features.mean(dim=0, keepdim=True)
    features = features / torch.clamp(features.std(dim=0, keepdim=True), min=1.0e-6)

    distance_to_mean = torch.norm(features, dim=1)
    first_index = int(torch.argmax(distance_to_mean).item())
    selected = [first_index]
    min_distance = torch.norm(features - features[first_index], dim=1)
    min_distance[first_index] = -1.0

    while len(selected) < num_select:
        next_index = int(torch.argmax(min_distance).item())
        selected.append(next_index)
        candidate_distance = torch.norm(features - features[next_index], dim=1)
        min_distance = torch.minimum(min_distance, candidate_distance)
        min_distance[selected] = -1.0

    return torch.tensor(selected, dtype=torch.long, device=root_states_xyzw.device)


def summarize_support_contacts(
    body_pos_w: torch.Tensor,
    body_names: list[str],
    contact_height_margin: float,
) -> dict[str, Any]:
    if body_pos_w.ndim != 3:
        raise ValueError("body_pos_w must have shape (N, B, 3)")
    if body_pos_w.shape[1] != len(body_names):
        raise ValueError("body_names length must match body_pos_w body dimension")

    lowest_heights = torch.amin(body_pos_w[:, :, 2], dim=1, keepdim=True)
    support_mask = body_pos_w[:, :, 2] <= (lowest_heights + contact_height_margin)
    body_hit_count = support_mask.sum(dim=0).to(dtype=torch.int64)
    signature_counts: dict[str, int] = {}
    for state_mask in support_mask.cpu():
        names = [body_names[idx] for idx, hit in enumerate(state_mask.tolist()) if hit]
        signature = "|".join(names[:6]) if names else "none"
        signature_counts[signature] = signature_counts.get(signature, 0) + 1

    top_signatures = [
        {"signature": signature, "count": count}
        for signature, count in sorted(signature_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    active_body_counts = {name: int(count) for name, count in zip(body_names, body_hit_count.tolist()) if count > 0}
    return {
        "heuristic": "body_z_within_margin_of_state_min_z",
        "contact_height_margin": float(contact_height_margin),
        "top_support_signatures": top_signatures,
        "body_hit_count": active_body_counts,
    }


def root_state_xyzw_to_wxyz(root_states_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert root states from upstream/GRSI xyzw quaternion layout to Isaac wxyz layout."""
    if root_states_xyzw.shape[-1] < 13:
        raise ValueError("root state tensor must have at least 13 columns")
    converted = root_states_xyzw.clone()
    converted[..., 3:7] = torch.cat([root_states_xyzw[..., 6:7], root_states_xyzw[..., 3:6]], dim=-1)
    return converted


def root_state_wxyz_to_xyzw(root_states_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert root states from Isaac wxyz quaternion layout to upstream/GRSI xyzw layout."""
    if root_states_wxyz.shape[-1] < 13:
        raise ValueError("root state tensor must have at least 13 columns")
    converted = root_states_wxyz.clone()
    converted[..., 3:7] = torch.cat([root_states_wxyz[..., 4:7], root_states_wxyz[..., 3:4]], dim=-1)
    return converted


def make_grsi_state_dict(
    robot_root_states_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    joint_names: list[str],
    body_names: list[str],
    source: str,
    generation_config: dict[str, Any],
    contact_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": GRSI_SCHEMA_VERSION,
        "robot_root_states_xyzw": robot_root_states_xyzw,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "joint_names": list(joint_names),
        "body_names": list(body_names),
        "source": source,
        "generation_config": dict(generation_config),
        "contact_summary": contact_summary or {},
    }


def augment_grsi_with_rotational_recombination(
    root_states_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    *,
    base_indexes: torch.Tensor,
    donor_indexes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if base_indexes.shape != donor_indexes.shape:
        raise ValueError("base_indexes and donor_indexes must have matching shapes")
    base_root = root_states_xyzw[base_indexes].clone()
    donor_root = root_states_xyzw[donor_indexes]
    donor_yaw = _yaw_from_xyzw(donor_root[:, 3:7])
    donor_yaw_quat = _yaw_quat_xyzw(donor_yaw)
    base_root[:, 3:7] = _quat_mul_xyzw(donor_yaw_quat, base_root[:, 3:7])
    base_root[:, 3:7] = base_root[:, 3:7] / torch.clamp(
        torch.linalg.norm(base_root[:, 3:7], dim=1, keepdim=True),
        min=1.0e-8,
    )
    base_root[:, 7:9] = _rotate_xy_vectors(base_root[:, 7:9], donor_yaw)
    base_root[:, 10:12] = _rotate_xy_vectors(base_root[:, 10:12], donor_yaw)
    return (
        base_root,
        dof_pos[base_indexes].clone(),
        dof_vel[base_indexes].clone(),
        {
            "enabled": True,
            "augmented_count": int(base_indexes.numel()),
            "rotation_source": "donor_yaw_component",
            "base_index_count": int(base_indexes.numel()),
            "donor_index_count": int(donor_indexes.numel()),
        },
    )


def _require_exact_int_metadata(path: str | Path, generation_config: dict[str, Any], key: str) -> int:
    value = generation_config.get(key)
    if value is None:
        raise ValueError(f"GRSI production file {path} metadata {key} is required for canonical validation")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"GRSI production file {path} metadata {key}={value!r} must be an integer for canonical validation"
        )
    return value


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _stable_metadata_payload(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {key: _stable_metadata_payload(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_metadata_payload(item) for item in value]
    return value


def compute_grsi_artifact_hash(data: dict[str, Any]) -> str:
    payload = {
        "version": data["version"],
        "source": data["source"],
        "joint_names": list(data["joint_names"]),
        "body_names": list(data["body_names"]),
        "generation_config": _stable_metadata_payload(data["generation_config"]),
        "contact_summary": _stable_metadata_payload(data.get("contact_summary", {})),
        "robot_root_states_xyzw": _stable_metadata_payload(data["robot_root_states_xyzw"]),
        "dof_pos": _stable_metadata_payload(data["dof_pos"]),
        "dof_vel": _stable_metadata_payload(data["dof_vel"]),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def validate_grsi_state_dict(data: dict[str, Any], expected_joint_names: list[str] | None = None) -> GRSIValidationReport:
    missing = [key for key in GRSI_REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"GRSI state file is missing keys: {missing}")
    if not _is_exact_int(data["version"]) or data["version"] != GRSI_SCHEMA_VERSION:
        raise ValueError(f"Unsupported GRSI schema version {data['version']!r}; expected {GRSI_SCHEMA_VERSION}")
    root = data["robot_root_states_xyzw"]
    dof_pos = data["dof_pos"]
    dof_vel = data["dof_vel"]
    if root.ndim != 2 or root.shape[1] < 13:
        raise ValueError("robot_root_states_xyzw must have shape (N, >=13)")
    if dof_pos.shape != dof_vel.shape:
        raise ValueError("dof_pos and dof_vel must have matching shapes")
    if root.shape[0] != dof_pos.shape[0]:
        raise ValueError("root and dof tensors must have matching state counts")
    if expected_joint_names is not None and list(data["joint_names"]) != list(expected_joint_names):
        raise ValueError("GRSI joint_names do not match expected G1 joint order")

    finite = torch.isfinite(root).all(dim=1) & torch.isfinite(dof_pos).all(dim=1) & torch.isfinite(dof_vel).all(dim=1)
    quat_norm = torch.linalg.norm(root[:, 3:7], dim=1)
    valid_quat = torch.abs(quat_norm - 1.0) < 1.0e-2
    accepted_mask = finite & valid_quat
    accepted = int(torch.count_nonzero(accepted_mask).item())
    return GRSIValidationReport(total=int(root.shape[0]), accepted=accepted, rejected=int(root.shape[0]) - accepted)


def validate_grsi_production_contract(data: dict[str, Any], path: str | Path) -> None:
    report = validate_grsi_state_dict(data)
    validate_grsi_production_contract_with_report(data, path, report)


def validate_grsi_production_contract_with_report(
    data: dict[str, Any],
    path: str | Path,
    report: GRSIValidationReport,
) -> None:
    generation_config = data.get("generation_config")
    if not isinstance(generation_config, dict) or not generation_config:
        raise ValueError(f"GRSI production file {path} requires non-empty generation_config metadata")

    mode = generation_config.get("mode")
    if mode != GRSI_PRODUCTION_MODE:
        raise ValueError(
            f"GRSI production file {path} metadata mode={mode!r} does not match "
            f"required production mode={GRSI_PRODUCTION_MODE!r}"
        )

    contract_minimum_states = int(G1_PAPER_EQUIVALENCE_MIN_GRSI_STATES)
    metadata_minimum_states = _require_exact_int_metadata(path, generation_config, "minimum_required_states")
    if metadata_minimum_states != contract_minimum_states:
        raise ValueError(
            f"GRSI production file {path} metadata minimum_required_states={metadata_minimum_states} does not match "
            f"contract minimum_required_states={contract_minimum_states}"
        )
    metadata_accepted = _require_exact_int_metadata(path, generation_config, "accepted")
    actual_accepted = int(report.accepted)
    if metadata_accepted != actual_accepted:
        raise ValueError(
            f"GRSI production file {path} metadata accepted={metadata_accepted} does not match "
            f"validated accepted={actual_accepted}"
        )
    candidate_rejected = _require_exact_int_metadata(path, generation_config, "candidate_rejected")
    if candidate_rejected < 0:
        raise ValueError(f"GRSI production file {path} metadata candidate_rejected={candidate_rejected} must be >= 0")
    candidate_rejection_reasons = generation_config.get("candidate_rejection_reasons")
    if not isinstance(candidate_rejection_reasons, dict) or not candidate_rejection_reasons:
        raise ValueError(
            f"GRSI production file {path} requires non-empty candidate_rejection_reasons metadata for auditability"
        )
    if generation_config.get("zero_torque") is not True:
        raise ValueError(f"GRSI production file {path} requires zero_torque=True production metadata")
    friction_randomization = generation_config.get("friction_randomization")
    if not isinstance(friction_randomization, dict) or not friction_randomization:
        raise ValueError(
            f"GRSI production file {path} requires non-empty friction_randomization metadata for canonical validation"
        )
    rotational_recombination = generation_config.get("rotational_recombination")
    if not isinstance(rotational_recombination, dict) or not rotational_recombination:
        raise ValueError(
            f"GRSI production file {path} requires non-empty rotational_recombination metadata for canonical validation"
        )
    source_contract_id = generation_config.get("source_contract_id")
    if source_contract_id != G1_PAPER_EQUIVALENCE_CONTRACT_ID:
        raise ValueError(
            f"GRSI production file {path} source_contract_id={source_contract_id!r} does not match "
            f"{G1_PAPER_EQUIVALENCE_CONTRACT_ID!r}"
        )
    source = data.get("source")
    if source != GRSI_PRODUCTION_SOURCE:
        raise ValueError(
            f"GRSI production file {path} source={source!r} does not match required Isaac-only provenance "
            f"{GRSI_PRODUCTION_SOURCE!r}"
        )
    contact_summary = data.get("contact_summary")
    if not isinstance(contact_summary, dict) or not contact_summary:
        raise ValueError(f"GRSI production file {path} requires non-empty contact_summary metadata")
    if report.rejected != 0:
        raise ValueError(
            f"GRSI production file {path} contains {report.rejected} rejected states; "
            "canonical pools must contain only schema-valid rows"
        )
    if actual_accepted < contract_minimum_states:
        raise ValueError(
            f"GRSI production file {path} accepted only {actual_accepted} states; "
            f"expected at least {contract_minimum_states}"
        )


def save_grsi_state_file(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_grsi_state_file(path: str | Path, expected_joint_names: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    report = validate_grsi_state_dict(data, expected_joint_names=expected_joint_names)
    if path.name == "grsi_states.pth":
        validate_grsi_production_contract_with_report(data, path, report)
    return data
