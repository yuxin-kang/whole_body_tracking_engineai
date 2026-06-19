#!/bin/bash
set -euo pipefail

REPO_DIR="/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/kyx/Workspace/IsaacLab"
CONDA_ENV="/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid"
PYTHON_BIN="$CONDA_ENV/bin/python"
SBATCH_SCRIPT="$REPO_DIR/scripts/slurm/run_g1_paper_fastsac_stage.sbatch"

export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"

cd "$REPO_DIR"
mkdir -p slurm_logs

compute_contract_values() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import importlib.util
import json
import sys
import types

repo = Path("/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai")
package_root = repo / "source" / "whole_body_tracking" / "whole_body_tracking"
g1_root = package_root / "tasks" / "tracking" / "config" / "g1"
package_paths = {
    "whole_body_tracking": package_root,
    "whole_body_tracking.tasks": package_root / "tasks",
    "whole_body_tracking.tasks.tracking": package_root / "tasks" / "tracking",
    "whole_body_tracking.tasks.tracking.config": package_root / "tasks" / "tracking" / "config",
    "whole_body_tracking.tasks.tracking.config.g1": g1_root,
}

for package_name, package_path in package_paths.items():
    if package_name in sys.modules:
        continue
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

paper = load_module(
    "whole_body_tracking.tasks.tracking.config.g1.paper_contract",
    g1_root / "paper_contract.py",
)
grsi = load_module(
    "whole_body_tracking.tasks.tracking.config.g1.grsi",
    g1_root / "grsi.py",
)

from whole_body_tracking.rl.fast_sac.recipe_contract import FAST_SAC_RECIPE_CONTRACT_ID

grsi_data = grsi.load_grsi_state_file(repo / "data" / "g1" / "grsi_states.pth")
payload = {
    "paper_contract_id": paper.G1_PAPER_EQUIVALENCE_CONTRACT_ID,
    "recipe_contract_id": FAST_SAC_RECIPE_CONTRACT_ID,
    "grsi_hash": grsi.compute_grsi_artifact_hash(grsi_data),
}
print(json.dumps(payload))
PY
}

contract_json="$(compute_contract_values)"
EXPECTED_CONTRACT_ID="$(printf '%s' "$contract_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["paper_contract_id"])')"
EXPECTED_RECIPE_CONTRACT_ID="$(printf '%s' "$contract_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["recipe_contract_id"])')"
EXPECTED_GRSI_HASH="$(printf '%s' "$contract_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["grsi_hash"])')"
EXPECTED_GRSI_PATH="data/g1/grsi_states.pth"

NUM_ENVS="${NUM_ENVS:-4096}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1000}"
CURVE_INTERVAL="${CURVE_INTERVAL:-$PROGRESS_INTERVAL}"
FINAL_STEP="${FINAL_STEP:-30000}"
MAX_STEPS="${MAX_STEPS:-$FINAL_STEP}"
TRAIN_STEPS="${TRAIN_STEPS:-$FINAL_STEP}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ACCEPTANCE="${SKIP_ACCEPTANCE:-0}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-paper-fastsac}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_GROUP="${WANDB_GROUP:-g1-paper-fastsac-30k}"
WANDB_TAGS="${WANDB_TAGS:-g1-paper-fastsac-30k}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
MOTION_FILE="${MOTION_FILE:-data/g1/1307.npz}"
EXCLUDE_NODES="${EXCLUDE_NODES:-epyc2}"

submit_stage() {
  local job_name="$1"
  local dependency="$2"
  shift 2
  local cmd=(sbatch --parsable --job-name "$job_name")
  if [[ -n "$EXCLUDE_NODES" ]]; then
    cmd+=(--exclude "$EXCLUDE_NODES")
  fi
  if [[ -n "$dependency" ]]; then
    cmd+=(--dependency "$dependency")
  fi
  while (($#)); do
    cmd+=(--export "$1")
    shift
  done
  cmd+=("$SBATCH_SCRIPT")
  "${cmd[@]}"
}

stage_i_job="$(submit_stage \
  g1_pfsac_stage_i \
  "" \
  "ALL,STAGE_LABEL=stage-i,TASK_ID=Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0,RUN_NAME=g1-paper-fastsac-stage-i-30k,EXPECTED_TASK_ID=Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0,EXPECTED_CONTRACT_ID=$EXPECTED_CONTRACT_ID,EXPECTED_RECIPE_CONTRACT_ID=$EXPECTED_RECIPE_CONTRACT_ID,EXPECTED_GRSI_PATH=$EXPECTED_GRSI_PATH,EXPECTED_GRSI_HASH=$EXPECTED_GRSI_HASH,NUM_ENVS=$NUM_ENVS,EVAL_EPISODES=$EVAL_EPISODES,CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL,PROGRESS_INTERVAL=$PROGRESS_INTERVAL,CURVE_INTERVAL=$CURVE_INTERVAL,FINAL_STEP=$FINAL_STEP,MAX_STEPS=$MAX_STEPS,TRAIN_STEPS=$TRAIN_STEPS,WARMUP_STEPS=$WARMUP_STEPS,BATCH_SIZE=$BATCH_SIZE,UPDATES_PER_STEP=$UPDATES_PER_STEP,SKIP_EVAL=$SKIP_EVAL,SKIP_ACCEPTANCE=$SKIP_ACCEPTANCE,LOGGER=$LOGGER,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_MODE=$WANDB_MODE,WANDB_GROUP=$WANDB_GROUP,WANDB_TAGS=$WANDB_TAGS,WANDB_RUN_ID=$WANDB_RUN_ID,WANDB_RESUME=$WANDB_RESUME,MOTION_FILE=$MOTION_FILE")"

stage_ii_job="$(submit_stage \
  g1_pfsac_stage_ii \
  "afterok:$stage_i_job" \
  "ALL,STAGE_LABEL=stage-ii,TASK_ID=Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0,RUN_NAME=g1-paper-fastsac-stage-ii-30k,RESUME_FROM_RUN_NAME=g1-paper-fastsac-stage-i-30k,EXPECTED_TASK_ID=Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0,EXPECTED_CONTRACT_ID=$EXPECTED_CONTRACT_ID,EXPECTED_RECIPE_CONTRACT_ID=$EXPECTED_RECIPE_CONTRACT_ID,EXPECTED_GRSI_PATH=$EXPECTED_GRSI_PATH,EXPECTED_GRSI_HASH=$EXPECTED_GRSI_HASH,NUM_ENVS=$NUM_ENVS,EVAL_EPISODES=$EVAL_EPISODES,CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL,PROGRESS_INTERVAL=$PROGRESS_INTERVAL,CURVE_INTERVAL=$CURVE_INTERVAL,FINAL_STEP=$FINAL_STEP,MAX_STEPS=$MAX_STEPS,TRAIN_STEPS=$TRAIN_STEPS,WARMUP_STEPS=$WARMUP_STEPS,BATCH_SIZE=$BATCH_SIZE,UPDATES_PER_STEP=$UPDATES_PER_STEP,SKIP_EVAL=$SKIP_EVAL,SKIP_ACCEPTANCE=$SKIP_ACCEPTANCE,LOGGER=$LOGGER,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_MODE=$WANDB_MODE,WANDB_GROUP=$WANDB_GROUP,WANDB_TAGS=$WANDB_TAGS,WANDB_RUN_ID=$WANDB_RUN_ID,WANDB_RESUME=$WANDB_RESUME,MOTION_FILE=$MOTION_FILE")"

stage_iii_job="$(submit_stage \
  g1_pfsac_stage_iii \
  "afterok:$stage_ii_job" \
  "ALL,STAGE_LABEL=stage-iii,TASK_ID=Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0,RUN_NAME=g1-paper-fastsac-stage-iii-30k,RESUME_FROM_RUN_NAME=g1-paper-fastsac-stage-ii-30k,EXPECTED_TASK_ID=Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0,EXPECTED_CONTRACT_ID=$EXPECTED_CONTRACT_ID,EXPECTED_RECIPE_CONTRACT_ID=$EXPECTED_RECIPE_CONTRACT_ID,EXPECTED_GRSI_PATH=$EXPECTED_GRSI_PATH,EXPECTED_GRSI_HASH=$EXPECTED_GRSI_HASH,NUM_ENVS=$NUM_ENVS,EVAL_EPISODES=$EVAL_EPISODES,CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL,PROGRESS_INTERVAL=$PROGRESS_INTERVAL,CURVE_INTERVAL=$CURVE_INTERVAL,FINAL_STEP=$FINAL_STEP,MAX_STEPS=$MAX_STEPS,TRAIN_STEPS=$TRAIN_STEPS,WARMUP_STEPS=$WARMUP_STEPS,BATCH_SIZE=$BATCH_SIZE,UPDATES_PER_STEP=$UPDATES_PER_STEP,SKIP_EVAL=$SKIP_EVAL,SKIP_ACCEPTANCE=$SKIP_ACCEPTANCE,LOGGER=$LOGGER,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_MODE=$WANDB_MODE,WANDB_GROUP=$WANDB_GROUP,WANDB_TAGS=$WANDB_TAGS,WANDB_RUN_ID=$WANDB_RUN_ID,WANDB_RESUME=$WANDB_RESUME,MOTION_FILE=$MOTION_FILE")"

printf 'stage_i_job=%s\nstage_ii_job=%s\nstage_iii_job=%s\n' "$stage_i_job" "$stage_ii_job" "$stage_iii_job"
printf 'expected_contract_id=%s\nexpected_recipe_contract_id=%s\nexpected_grsi_hash=%s\n' \
  "$EXPECTED_CONTRACT_ID" \
  "$EXPECTED_RECIPE_CONTRACT_ID" \
  "$EXPECTED_GRSI_HASH"
