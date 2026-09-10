"""Strict, same-format continuation of the trusted j9519 training snapshot."""

from __future__ import annotations

import math

import torch


def config_changes(before, after, prefix=""):
    """Return leaf-level differences for configclass.to_dict() snapshots."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = {}
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                changes[path] = [repr(before.get(key)), repr(after.get(key))]
            else:
                changes.update(config_changes(before[key], after[key], path))
        return changes
    if before == after:
        return {}
    return {prefix: [repr(before), repr(after)]}


def assert_reward_only_changes(changes, variant):
    allowed = ["rewards.getup_transition_anchor_ori"]
    if variant == "ori_velgate":
        allowed += ["rewards.motion_body_lin_vel.func", "rewards.motion_body_lin_vel.params",
                    "rewards.motion_body_ang_vel.func", "rewards.motion_body_ang_vel.params"]
    elif variant != "ori":
        raise ValueError(f"Unknown experiment: {variant}")
    unexpected = [path for path in changes if not any(path == p or path.startswith(p + ".") for p in allowed)]
    if unexpected or not changes:
        raise ValueError(f"Unexpected or empty experiment delta: {unexpected or changes}")


def _assert_state_equal(expected, actual, label):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError(f"{label}: checkpoint keys differ")
        for key in expected:
            _assert_state_equal(expected[key], actual[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError(f"{label}: checkpoint lengths differ")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_state_equal(left, right, f"{label}.{index}")
    elif isinstance(expected, torch.Tensor):
        if not torch.equal(expected.cpu(), actual.detach().cpu()):
            raise RuntimeError(f"{label}: checkpoint tensor differs")
    elif expected != actual:
        raise RuntimeError(f"{label}: checkpoint value differs")


def restore_finetune_checkpoint(runner, path, learning_rate):
    """Restore model, both normalizers and Adam, then enforce the small fixed LR.

    Unlike the inference compatibility loader, no partial actor/critic loading
    is allowed. This deliberately targets the installed RSL-RL runner format.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
                "privileged_obs_norm_state_dict", "iter"}
    if not required <= checkpoint.keys():
        raise ValueError(f"Incomplete training checkpoint: {required - checkpoint.keys()}")
    runner.load(str(path), load_optimizer=True)
    _assert_state_equal(checkpoint["model_state_dict"], runner.alg.policy.state_dict(), "policy")
    _assert_state_equal(checkpoint["obs_norm_state_dict"], runner.obs_normalizer.state_dict(), "actor_normalizer")
    _assert_state_equal(checkpoint["privileged_obs_norm_state_dict"],
                        runner.privileged_obs_normalizer.state_dict(), "critic_normalizer")
    _assert_state_equal(checkpoint["optimizer_state_dict"], runner.alg.optimizer.state_dict(), "optimizer")
    old_lrs = [group["lr"] for group in runner.alg.optimizer.param_groups]
    runner.alg.learning_rate = learning_rate
    runner.alg.schedule = "fixed"
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = learning_rate
    # Checkpoint iter is the last completed update, not the next one.
    runner.current_learning_iteration = int(checkpoint["iter"]) + 1
    return {
        "source_iteration": int(checkpoint["iter"]),
        "first_iteration": runner.current_learning_iteration,
        "source_optimizer_lrs": old_lrs,
        "effective_optimizer_lrs": [group["lr"] for group in runner.alg.optimizer.param_groups],
        "schedule": runner.alg.schedule,
        "model_tensors_verified": len(checkpoint["model_state_dict"]),
        "actor_normalizer_verified": True,
        "critic_normalizer_verified": True,
        "optimizer_state_verified": True,
    }
