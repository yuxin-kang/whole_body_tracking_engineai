from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from whole_body_tracking.rl.fast_sac.recipe_contract import (
    FAST_SAC_LOCAL_STAGED_ACCEPTANCE_STEP_FLOORS,
    FAST_SAC_RECIPE_CONTRACT_ID,
)


G1_PAPER_EQUIVALENCE_CONTRACT_VERSION = "2026-06-18"
G1_PAPER_EQUIVALENCE_CONTRACT_NAME = "g1-paper-fast-sac-isaac-only"
G1_PAPER_EQUIVALENCE_MIN_GRSI_STATES = 512


G1_PAPER_EQUIVALENCE_CONTRACT: dict[str, Any] = {
    "name": G1_PAPER_EQUIVALENCE_CONTRACT_NAME,
    "version": G1_PAPER_EQUIVALENCE_CONTRACT_VERSION,
    "scope": {
        "simulation_platform": "IsaacLab / Isaac Sim",
        "evaluation_boundary": "isaac-only",
        "includes_mujoco_evaluation": False,
        "includes_real_robot_evaluation": False,
    },
    "source_classes": {
        "paper_primary": "arXiv:2602.13656",
        "fastsac_reference": "arXiv:2512.01996",
        "local_parameter": "Paper-underspecified local implementation parameter.",
    },
    "sources_used": ["paper_primary", "fastsac_reference", "local_parameter"],
    "fast_sac_recipe_contract_id": FAST_SAC_RECIPE_CONTRACT_ID,
    "fast_sac_architecture": {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 256, 128],
        "paper_claim": {
            "actor_layers": 3,
            "critic_layers_per_q_network": 4,
        },
        "implementation_status": "paper-aligned-layer-count",
        "implementation_note": (
            "Hidden-layer count matches the paper. Exact hidden widths remain explicit local parameters because the "
            "paper does not publish them."
        ),
    },
    "fast_sac_taxonomy": {
        "actor_critic_depth": {
            "source_class": "paper_primary",
            "local_status": "implemented_required",
            "decision": "Actor uses three hidden layers and each Q-network uses four hidden layers.",
        },
        "average_twin_q": {
            "source_class": "fastsac_reference",
            "local_status": "implemented_required",
            "decision": "Use mean twin-Q aggregation in target and actor updates.",
        },
        "layer_norm": {
            "source_class": "fastsac_reference",
            "local_status": "implemented_required",
            "decision": "Apply LayerNorm on hidden activations in actor and critic trunks.",
        },
        "observation_normalization": {
            "source_class": "fastsac_reference",
            "local_status": "implemented_required",
            "decision": "Normalize actor and critic observations with checkpointed running statistics.",
        },
        "distributional_critic_c51": {
            "source_class": "fastsac_reference",
            "local_status": "implemented_required",
            "decision": "Use twin C51 critics and optimize their projected target distributions.",
        },
        "batch_profile": {
            "source_class": "local_parameter",
            "local_status": "implemented_local_parameter",
            "decision": "Keep batch size and update cadence explicit as paper-underspecified local parameters.",
        },
        "entropy_alpha": {
            "source_class": "fastsac_reference",
            "local_status": "implemented_required",
            "decision": "Use learned entropy temperature with paper-aligned initialization and target entropy policy.",
        },
        "target_update": {
            "source_class": "fastsac_reference",
            "local_status": "implemented_required",
            "decision": "Keep soft target-network updates in the FastSAC runner.",
        },
        "replay_resume": {
            "source_class": "local_parameter",
            "local_status": "implemented_required",
            "decision": "Preserve replay metadata, observation normalizers, and optimizer state in checkpoints.",
        },
    },
    "reward_profile": {
        "undesired_contacts": {
            "weight": -0.5,
            "status": "implemented",
            "source_class": "paper_primary",
        },
        "action_rate_before_stand": {
            "weight": -2.0,
            "status": "implemented",
            "source_class": "paper_primary",
            "note": "Applied only while shoulder-height mismatch stays above the standing threshold.",
        },
    },
    "lke": {
        "anchor_detection": {
            "source_class": "paper_primary",
            "status": "implemented",
            "proxy": "reference_joint_velocity_energy_local_minima",
        },
        "anchor_weight_update": {
            "source_class": "paper_primary",
            "status": "implemented",
            "rule": "clip(w + alpha, w_min, w_max) on the nearest preceding failure anchor",
        },
        "local_parameters": {
            "alpha": 0.2,
            "w_min": 1.0,
            "w_max": 8.0,
            "local_minimum_window": 1,
        },
    },
    "domain_randomization": {
        "mass": {"status": "paper_category_required", "range_source": "local"},
        "friction": {"status": "paper_category_required", "range_source": "local"},
        "base_com": {"status": "paper_category_required", "range_source": "local"},
        "pd_gain": {"status": "paper_category_required", "range_source": "local"},
        "action_delay": {"status": "paper_category_required", "range_source": "local"},
        "torque_disturbance": {"status": "paper_category_required", "range_source": "local"},
        "joint_bias": {"status": "paper_category_required", "range_source": "local"},
        "reset_pose_velocity": {"status": "paper_category_required", "range_source": "local"},
        "push": {"status": "paper_category_required", "range_source": "local"},
        "terrain": {"status": "paper_category_required", "range_source": "local"},
    },
    "evaluation": {
        "success_rate": {
            "formula": (
                "success = no per-joint absolute position error > threshold, "
                "no tracked-body orientation error > threshold, no fall"
            ),
            "joint_position_error_threshold_rad": 0.5,
            "tracked_body_orientation_error_threshold_rad": 0.8,
            "robot_fall_condition": "environment termination before horizon or explicit fall-triggered failure",
            "aggregation": "successful_episodes / total_episodes",
        },
        "orientation_error": {
            "metric_name": "root_relative_mean_per_body_orientation_error",
            "unit": "rad",
            "formula": "mean(quat_error_magnitude(reference_body_quat, robot_body_quat)) across tracked bodies and episode steps",
        },
        "smoothness": {
            "metric_name": "mean_action_rate_l2",
            "unit": "delta_action_l2",
            "formula": "mean(||a_t - a_(t-1)||_2) across evaluation steps after the first action",
            "lower_is_better": True,
        },
    },
    "grsi": {
        "production_minimum_states": G1_PAPER_EQUIVALENCE_MIN_GRSI_STATES,
        "production_source": "isaaclab-grsi:gravity-release",
        "support_summary_method": "body-height proximity heuristic over accepted post-release states",
        "production_rejected_rows_must_equal_zero": True,
        "generation_candidate_rejections_are_audit_only": True,
        "generation_candidate_rejection_audit_required": True,
        "required_generation_evidence": [
            "zero_torque",
            "friction_randomization",
            "rotational_recombination",
            "candidate_rejected",
            "candidate_rejection_reasons",
            "source_contract_id",
        ],
    },
    "artifacts": {
        "local_staged_acceptance_step_floor_by_task": FAST_SAC_LOCAL_STAGED_ACCEPTANCE_STEP_FLOORS,
        "staged_acceptance_note": (
            "30k staged acceptance is local Isaac-only training evidence, not paper long-run parity. "
            "Updates begin after the 10k warmup, so each 30k stage has about 20k update-eligible steps."
        ),
        "accepted_manifest_required_fields": [
            "contract_id",
            "recipe_contract_id",
            "task",
            "stage",
            "checkpoint_path",
            "grsi_hash",
            "source_classes_used",
            "local_staged_acceptance_step_floor",
        ],
    },
}


def _stable_contract_payload(contract: dict[str, Any]) -> str:
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def compute_contract_hash(contract: dict[str, Any] | None = None) -> str:
    payload = _stable_contract_payload(contract or G1_PAPER_EQUIVALENCE_CONTRACT)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


G1_PAPER_EQUIVALENCE_CONTRACT_HASH = compute_contract_hash()
G1_PAPER_EQUIVALENCE_CONTRACT_ID = (
    f"{G1_PAPER_EQUIVALENCE_CONTRACT_NAME}:{G1_PAPER_EQUIVALENCE_CONTRACT_VERSION}:{G1_PAPER_EQUIVALENCE_CONTRACT_HASH}"
)


def get_paper_equivalence_contract() -> dict[str, Any]:
    payload = copy.deepcopy(G1_PAPER_EQUIVALENCE_CONTRACT)
    payload["contract_hash"] = G1_PAPER_EQUIVALENCE_CONTRACT_HASH
    payload["contract_id"] = G1_PAPER_EQUIVALENCE_CONTRACT_ID
    return payload
