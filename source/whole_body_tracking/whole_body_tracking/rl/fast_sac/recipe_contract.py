from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


FAST_SAC_RECIPE_CONTRACT_VERSION = "2026-06-18"
FAST_SAC_RECIPE_CONTRACT_NAME = "fast-sac-paper-reference"
FAST_SAC_LOCAL_STAGED_ACCEPTANCE_STEP_FLOORS = {
    "Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0": 30_000,
    "Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0": 30_000,
    "Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0": 30_000,
}


FAST_SAC_RECIPE_CONTRACT: dict[str, Any] = {
    "name": FAST_SAC_RECIPE_CONTRACT_NAME,
    "version": FAST_SAC_RECIPE_CONTRACT_VERSION,
    "source_classes": {
        "paper_primary": "arXiv:2602.13656",
        "fastsac_reference": "arXiv:2512.01996",
        "local_parameter": "Paper-underspecified local implementation parameter.",
    },
    "paper_explicit_required": {
        "actor_hidden_layer_count": 3,
        "critic_hidden_layer_count_per_q_network": 4,
        "wbt_gamma": 0.99,
    },
    "fastsac_reference_required": {
        "observation_normalization": True,
        "layer_norm": True,
        "mean_twin_q": True,
        "distributional_critic": {
            "kind": "c51",
            "num_atoms": 51,
        },
        "sigma_max": 1.0,
        "log_std_max": 0.0,
        "init_alpha": 0.001,
        "auto_alpha_tuning": True,
        "target_entropy_policy": "-|A|/2",
        "optimizer": {
            "kind": "adam",
            "lr": 3.0e-4,
            "weight_decay": 1.0e-3,
            "beta2": 0.95,
        },
    },
    "paper_underspecified_local_parameter": {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 256, 128],
        "optimizer_beta1": 0.9,
        "distributional_value_support": {
            "v_min": -50.0,
            "v_max": 50.0,
        },
        "batch_size": 256,
        "updates_per_step": 4,
        "local_staged_acceptance_step_floor_by_task": FAST_SAC_LOCAL_STAGED_ACCEPTANCE_STEP_FLOORS,
        "staged_acceptance_note": (
            "30k staged acceptance is local evidence only; it is not paper long-run parity. "
            "With 10k warmup, each 30k stage has about 20k update-eligible steps."
        ),
    },
    "blocked_until_sourced": [],
}


def _stable_contract_payload(contract: dict[str, Any]) -> str:
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def compute_recipe_contract_hash(contract: dict[str, Any] | None = None) -> str:
    payload = _stable_contract_payload(contract or FAST_SAC_RECIPE_CONTRACT)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


FAST_SAC_RECIPE_CONTRACT_HASH = compute_recipe_contract_hash()
FAST_SAC_RECIPE_CONTRACT_ID = (
    f"{FAST_SAC_RECIPE_CONTRACT_NAME}:{FAST_SAC_RECIPE_CONTRACT_VERSION}:{FAST_SAC_RECIPE_CONTRACT_HASH}"
)


def get_fastsac_recipe_contract() -> dict[str, Any]:
    payload = copy.deepcopy(FAST_SAC_RECIPE_CONTRACT)
    payload["contract_hash"] = FAST_SAC_RECIPE_CONTRACT_HASH
    payload["contract_id"] = FAST_SAC_RECIPE_CONTRACT_ID
    return payload
