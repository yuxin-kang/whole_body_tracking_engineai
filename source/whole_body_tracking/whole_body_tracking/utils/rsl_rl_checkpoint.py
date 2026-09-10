"""Compatibility helpers for loading RSL-RL checkpoints for inference."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch


_ACTOR_NORMALIZER_PREFIX = "actor_obs_normalizer."
_CRITIC_NORMALIZER_PREFIX = "critic_obs_normalizer."


def _extract_prefixed_state(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


def _normalizer_accepts_state(normalizer: object | None, state: Mapping[str, Any]) -> bool:
    if normalizer is None or not hasattr(normalizer, "state_dict"):
        return False
    target_state = normalizer.state_dict()
    for key, value in state.items():
        if key not in target_state:
            return False
        target_value = target_state[key]
        if hasattr(value, "shape") and hasattr(target_value, "shape") and value.shape != target_value.shape:
            return False
    return True


def _load_runner_normalizer(runner: object, attr_name: str, state: Mapping[str, Any] | None) -> bool:
    if not state:
        return False
    normalizer = getattr(runner, attr_name, None)
    if not _normalizer_accepts_state(normalizer, state):
        return False
    normalizer.load_state_dict(state)
    normalizer.eval()
    return True


def load_rsl_rl_checkpoint_for_inference(runner: object, path: str | PathLike[str]) -> dict[str, Any]:
    """Load policy and observation normalizers across RSL-RL checkpoint formats.

    RSL-RL 2.x stores actor and critic observation normalizers at checkpoint
    top level. RSL-RL 3.x stores them inside the policy state dictionary. This
    loader maps either representation to the representation expected by the
    current runner and deliberately skips optimizer restoration for inference.
    """

    device = getattr(runner, "device", "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {path}")

    policy = runner.alg.policy
    model_state = dict(checkpoint["model_state_dict"])
    target_state = policy.state_dict()

    actor_state = checkpoint.get("obs_norm_state_dict") or _extract_prefixed_state(
        model_state, _ACTOR_NORMALIZER_PREFIX
    )
    critic_state = checkpoint.get("privileged_obs_norm_state_dict") or _extract_prefixed_state(
        model_state, _CRITIC_NORMALIZER_PREFIX
    )

    target_has_actor_normalizer = any(key.startswith(_ACTOR_NORMALIZER_PREFIX) for key in target_state)
    target_has_critic_normalizer = any(key.startswith(_CRITIC_NORMALIZER_PREFIX) for key in target_state)

    if target_has_actor_normalizer and actor_state:
        model_state.update({_ACTOR_NORMALIZER_PREFIX + key: value for key, value in actor_state.items()})
    elif not target_has_actor_normalizer:
        model_state = {key: value for key, value in model_state.items() if not key.startswith(_ACTOR_NORMALIZER_PREFIX)}

    if target_has_critic_normalizer and critic_state:
        model_state.update({_CRITIC_NORMALIZER_PREFIX + key: value for key, value in critic_state.items()})
    elif not target_has_critic_normalizer:
        model_state = {
            key: value for key, value in model_state.items() if not key.startswith(_CRITIC_NORMALIZER_PREFIX)
        }

    # A target-pad task intentionally adds privileged target information to the
    # critic while keeping the actor observation contract unchanged.  This
    # makes an existing tracking actor usable for a visual pre-training check,
    # but its old critic first layer has a different input width.  Load only
    # state entries that exist with the same shape; actor mismatches still fail
    # through the normalizer check below, while an incompatible critic is safe
    # to skip because play mode never evaluates the value function.
    compatible_model_state: dict[str, Any] = {}
    skipped_model_keys: list[str] = []
    for key, value in model_state.items():
        target_value = target_state.get(key)
        if target_value is None or (
            hasattr(value, "shape") and hasattr(target_value, "shape") and value.shape != target_value.shape
        ):
            skipped_model_keys.append(key)
            continue
        compatible_model_state[key] = value
    policy.load_state_dict(compatible_model_state, strict=False)
    if skipped_model_keys:
        print(f"[WARN] Skipped incompatible checkpoint entries for inference: {', '.join(skipped_model_keys)}")

    actor_loaded = target_has_actor_normalizer and bool(actor_state)
    critic_loaded = target_has_critic_normalizer and bool(critic_state)
    actor_loaded |= _load_runner_normalizer(runner, "obs_normalizer", actor_state)
    critic_loaded |= _load_runner_normalizer(runner, "privileged_obs_normalizer", critic_state)

    if actor_state and not actor_loaded:
        raise RuntimeError(f"Actor observation normalizer from checkpoint was not applied: {path}")
    if critic_state and not critic_loaded:
        print("[WARN] Skipped incompatible critic observation normalizer; actor inference remains usable.")

    runner.current_learning_iteration = checkpoint.get("iter", 0)
    source_format = "top-level" if "obs_norm_state_dict" in checkpoint else "policy-embedded"
    target_format = "policy-embedded" if target_has_actor_normalizer else "runner-level"
    print(f"[INFO] Loaded RSL-RL normalizers: {source_format} -> {target_format}")
    return checkpoint.get("infos") or {}
