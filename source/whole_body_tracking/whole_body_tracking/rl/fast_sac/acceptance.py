from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any
import types

import torch

from whole_body_tracking.rl.fast_sac.recipe_contract import FAST_SAC_RECIPE_CONTRACT_ID


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
G1_CONFIG_ROOT = PACKAGE_ROOT / "tasks" / "tracking" / "config" / "g1"
G1_PACKAGE_PATHS = {
    "whole_body_tracking": PACKAGE_ROOT,
    "whole_body_tracking.tasks": PACKAGE_ROOT / "tasks",
    "whole_body_tracking.tasks.tracking": PACKAGE_ROOT / "tasks" / "tracking",
    "whole_body_tracking.tasks.tracking.config": PACKAGE_ROOT / "tasks" / "tracking" / "config",
    "whole_body_tracking.tasks.tracking.config.g1": G1_CONFIG_ROOT,
}


def _ensure_g1_package_stubs() -> None:
    for package_name, package_path in G1_PACKAGE_PATHS.items():
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package


def _load_g1_module(module_basename: str):
    _ensure_g1_package_stubs()
    module_name = f"whole_body_tracking.tasks.tracking.config.g1.{module_basename}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module_path = G1_CONFIG_ROOT / f"{module_basename}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _paper_contract_module():
    return _load_g1_module("paper_contract")


def _grsi_module():
    return _load_g1_module("grsi")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_is_loadable(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"algorithm", "actor", "critic", "target_critic", "config"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Checkpoint {path} is missing required keys: {sorted(missing)}")
    if payload.get("algorithm") != "FastSAC":
        raise ValueError(f"Checkpoint {path} is not a FastSAC checkpoint")
    return payload


def _checkpoint_payload_step(payload: dict[str, Any], path: Path) -> int:
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError(f"Checkpoint {path} is missing an exact integer payload step")
    return step


def _parse_checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Checkpoint filename {path.name!r} does not match model_<step>.pt")
    return int(match.group(1))


def _resolve_checkpoint_path(root: Path, run_data: dict[str, Any]) -> tuple[Path, int]:
    configured_path = run_data.get("checkpoint_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("run.json checkpoint_path must be a non-empty string")
    checkpoint_path = Path(configured_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise ValueError(f"run.json checkpoint_path does not exist: {checkpoint_path}")
    return checkpoint_path, _parse_checkpoint_step(checkpoint_path)


def evaluate_acceptance(
    *,
    stage: str,
    log_dir: str | Path,
    expected_task_id: str,
    expected_contract_id: str | None = None,
    expected_recipe_contract_id: str = FAST_SAC_RECIPE_CONTRACT_ID,
    expected_grsi_path: str | Path | None = None,
    expected_grsi_hash: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    paper_contract = _paper_contract_module()
    if expected_contract_id is None:
        expected_contract_id = paper_contract.G1_PAPER_EQUIVALENCE_CONTRACT_ID
    root = Path(log_dir)
    run_data = _read_json(root / "params" / "run.json")
    agent_data = _read_json(root / "params" / "agent.json")
    metrics_data = _read_json(root / "metrics.json")
    eval_data = _read_json(root / "eval_summary.json")

    contract = run_data.get("paper_equivalence_contract")
    recipe_contract = run_data.get("fast_sac_recipe_contract")
    if not isinstance(contract, dict) or contract.get("contract_id") != expected_contract_id:
        raise ValueError("run.json paper_equivalence_contract does not match the expected contract id")
    if not isinstance(recipe_contract, dict) or recipe_contract.get("contract_id") != expected_recipe_contract_id:
        raise ValueError("run.json fast_sac_recipe_contract does not match the expected recipe contract id")
    if run_data.get("task") != expected_task_id:
        raise ValueError(f"run.json task {run_data.get('task')!r} does not match expected task {expected_task_id!r}")
    if "success_rate" not in eval_data or "orientation_error_mean" not in eval_data or "smoothness_mean" not in eval_data:
        raise ValueError("eval_summary.json is missing required top-level Isaac-only metrics")
    if "steps" not in metrics_data or "updates" not in metrics_data or "trained_steps" not in metrics_data:
        raise ValueError("metrics.json is missing steps/updates/trained_steps counters")

    staged_step_floor = paper_contract.get_paper_equivalence_contract()["artifacts"][
        "local_staged_acceptance_step_floor_by_task"
    ][expected_task_id]
    checkpoint_path, checkpoint_step = _resolve_checkpoint_path(root, run_data)
    checkpoint_payload = _checkpoint_is_loadable(checkpoint_path)
    payload_step = _checkpoint_payload_step(checkpoint_payload, checkpoint_path)
    if checkpoint_step < int(staged_step_floor):
        raise ValueError(
            f"checkpoint {checkpoint_path.name} step={checkpoint_step} does not reach the local staged acceptance floor {staged_step_floor}"
        )
    if int(metrics_data["steps"]) < int(staged_step_floor):
        raise ValueError(
            f"metrics.json steps={metrics_data['steps']} does not reach the local staged acceptance floor {staged_step_floor}"
        )
    if int(metrics_data["trained_steps"]) < int(staged_step_floor):
        raise ValueError(
            f"metrics.json trained_steps={metrics_data['trained_steps']} does not reach the local staged acceptance floor {staged_step_floor}"
        )
    if checkpoint_step != int(metrics_data["steps"]):
        raise ValueError(
            f"checkpoint step {checkpoint_step} does not match metrics.json steps {metrics_data['steps']}"
        )
    if payload_step != checkpoint_step:
        raise ValueError(f"checkpoint payload step {payload_step} does not match filename step {checkpoint_step}")
    if agent_data.get("batch_size") == 256 and int(metrics_data["steps"]) < 10_000:
        raise ValueError("acceptance refuses smoke-scale runs")

    grsi_hash = None
    if expected_grsi_path is not None:
        grsi = _grsi_module()
        grsi_data = grsi.load_grsi_state_file(expected_grsi_path)
        grsi_hash = grsi.compute_grsi_artifact_hash(grsi_data)
        if expected_grsi_hash is not None and grsi_hash != expected_grsi_hash:
            raise ValueError(
                f"Canonical GRSI hash {grsi_hash} does not match expected hash {expected_grsi_hash}"
            )

    manifest = {
        "status": "accepted",
        "stage": stage,
        "task": expected_task_id,
        "contract_id": expected_contract_id,
        "recipe_contract_id": expected_recipe_contract_id,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "grsi_hash": grsi_hash,
        "local_staged_acceptance_step_floor": int(staged_step_floor),
        "source_classes_used": ["paper_primary", "fastsac_reference", "local_parameter"],
        "metrics": {
            "steps": int(metrics_data["steps"]),
            "start_step": int(metrics_data.get("start_step", 0)),
            "trained_steps": int(metrics_data["trained_steps"]),
            "updates": int(metrics_data["updates"]),
            "success_rate": float(eval_data["success_rate"]),
            "orientation_error_mean": float(eval_data["orientation_error_mean"]),
            "smoothness_mean": float(eval_data["smoothness_mean"]),
        },
    }
    return True, manifest


def write_acceptance_artifacts(
    *,
    stage: str,
    log_dir: str | Path,
    expected_task_id: str,
    expected_contract_id: str | None = None,
    expected_recipe_contract_id: str = FAST_SAC_RECIPE_CONTRACT_ID,
    expected_grsi_path: str | Path | None = None,
    expected_grsi_hash: str | None = None,
) -> tuple[bool, Path]:
    root = Path(log_dir)
    accepted_path = root / "accepted_artifact.json"
    blocker_path = root / "acceptance_blocker.json"
    if accepted_path.exists():
        accepted_path.unlink()
    if blocker_path.exists():
        blocker_path.unlink()
    try:
        accepted, manifest = evaluate_acceptance(
            stage=stage,
            log_dir=root,
            expected_task_id=expected_task_id,
            expected_contract_id=expected_contract_id,
            expected_recipe_contract_id=expected_recipe_contract_id,
            expected_grsi_path=expected_grsi_path,
            expected_grsi_hash=expected_grsi_hash,
        )
    except Exception as exc:
        blocker = {
            "status": "blocked",
            "stage": stage,
            "task": expected_task_id,
            "contract_id": expected_contract_id,
            "recipe_contract_id": expected_recipe_contract_id,
            "reason": str(exc),
            "source_classes_used": ["paper_primary", "fastsac_reference", "local_parameter"],
        }
        _write_json(blocker_path, blocker)
        return False, blocker_path

    _write_json(accepted_path, manifest)
    return accepted, accepted_path
