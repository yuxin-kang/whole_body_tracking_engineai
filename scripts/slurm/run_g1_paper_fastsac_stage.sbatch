#!/bin/bash
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/kyx/Workspace/IsaacLab"
CONDA_ENV="/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid"
PYTHON_BIN="$CONDA_ENV/bin/python"

: "${STAGE_LABEL:?STAGE_LABEL is required}"
: "${TASK_ID:?TASK_ID is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${EXPECTED_TASK_ID:?EXPECTED_TASK_ID is required}"
: "${EXPECTED_CONTRACT_ID:?EXPECTED_CONTRACT_ID is required}"
: "${EXPECTED_RECIPE_CONTRACT_ID:?EXPECTED_RECIPE_CONTRACT_ID is required}"
: "${EXPECTED_GRSI_PATH:?EXPECTED_GRSI_PATH is required}"
: "${EXPECTED_GRSI_HASH:?EXPECTED_GRSI_HASH is required}"

MOTION_FILE="${MOTION_FILE:-data/g1/1307.npz}"
NUM_ENVS="${NUM_ENVS:-4096}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1000}"
CURVE_INTERVAL="${CURVE_INTERVAL:-$PROGRESS_INTERVAL}"
RESUME_FROM_RUN_NAME="${RESUME_FROM_RUN_NAME:-}"
FINAL_STEP="${FINAL_STEP:-30000}"
MAX_STEPS="${MAX_STEPS:-$FINAL_STEP}"
TRAIN_STEPS="${TRAIN_STEPS:-$FINAL_STEP}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ISOLATE_OMNI_HOME="${ISOLATE_OMNI_HOME:-0}"
SKIP_ACCEPTANCE="${SKIP_ACCEPTANCE:-0}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-paper-fastsac}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_GROUP="${WANDB_GROUP:-g1-paper-fastsac-30k}"
WANDB_TAGS="${WANDB_TAGS:-g1,paper-fastsac,30k,$STAGE_LABEL}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

export PYTHONUNBUFFERED=1
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

if [[ "$ISOLATE_OMNI_HOME" == "1" ]]; then
  JOB_HOME="/tmp/wbt_omni_home_${SLURM_JOB_ID:-manual}"
  export HOME="$JOB_HOME"
  export XDG_CACHE_HOME="$HOME/.cache"
  export XDG_DATA_HOME="$HOME/.local/share"
  export XDG_RUNTIME_DIR="$HOME/runtime"
  export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
  mkdir -p \
    "$XDG_CACHE_HOME" \
    "$XDG_DATA_HOME" \
    "$XDG_RUNTIME_DIR" \
    "$CUDA_CACHE_PATH" \
    "$HOME/Documents" \
    "$HOME/.nvidia-omniverse/logs"
  chmod 700 "$XDG_RUNTIME_DIR"
else
  mkdir -p "$HOME/.nv/ComputeCache" "$HOME/.nvidia-omniverse/logs"
fi

export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"

cd "$REPO_DIR"
mkdir -p slurm_logs

log_root="$REPO_DIR/logs/fast_sac/g1_paper_fast_sac"

find_latest_run_dir() {
  local target_run_name="$1"
  find "$log_root" -maxdepth 1 -mindepth 1 -type d -name "*_${target_run_name}" -printf '%P\n' | sort | tail -n 1
}

resolve_resume_checkpoint() {
  local prior_run_name="$1"
  local prior_dir
  prior_dir="$(find_latest_run_dir "$prior_run_name")"
  if [[ -z "$prior_dir" ]]; then
    echo "No prior run directory found for $prior_run_name" >&2
    return 1
  fi
  local checkpoint_path="$log_root/$prior_dir/model_${FINAL_STEP}.pt"
  if [[ ! -f "$checkpoint_path" ]]; then
    checkpoint_path="$log_root/$prior_dir/model_$((FINAL_STEP * 2)).pt"
  fi
  if [[ ! -f "$checkpoint_path" ]]; then
    checkpoint_path="$log_root/$prior_dir/model_$((FINAL_STEP * 3)).pt"
  fi
  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Expected resume checkpoint missing: $checkpoint_path" >&2
    return 1
  fi
  printf '%s\n' "$checkpoint_path"
}

resume_checkpoint=""
if [[ -n "$RESUME_FROM_RUN_NAME" ]]; then
  resume_checkpoint="$(resolve_resume_checkpoint "$RESUME_FROM_RUN_NAME")"
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "job_name=${SLURM_JOB_NAME:-manual}"
echo "node=$(hostname)"
echo "stage_label=$STAGE_LABEL"
echo "task_id=$TASK_ID"
echo "run_name=$RUN_NAME"
echo "motion_file=$MOTION_FILE"
echo "num_envs=$NUM_ENVS"
echo "final_step=$FINAL_STEP"
echo "max_steps=$MAX_STEPS"
echo "train_steps=$TRAIN_STEPS"
echo "checkpoint_interval=$CHECKPOINT_INTERVAL"
echo "progress_interval=$PROGRESS_INTERVAL"
echo "curve_interval=$CURVE_INTERVAL"
echo "logger=$LOGGER"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_entity=${WANDB_ENTITY:-<unset>}"
echo "wandb_mode=$WANDB_MODE"
echo "wandb_group=$WANDB_GROUP"
echo "wandb_tags=$WANDB_TAGS"
echo "wandb_resume=$WANDB_RESUME"
echo "isolate_omni_home=$ISOLATE_OMNI_HOME"
echo "skip_acceptance=$SKIP_ACCEPTANCE"
echo "home=$HOME"
echo "xdg_runtime_dir=${XDG_RUNTIME_DIR:-<unset>}"
echo "resume_checkpoint=${resume_checkpoint:-<none>}"
echo "expected_contract_id=$EXPECTED_CONTRACT_ID"
echo "expected_recipe_contract_id=$EXPECTED_RECIPE_CONTRACT_ID"
echo "expected_grsi_path=$EXPECTED_GRSI_PATH"
echo "expected_grsi_hash=$EXPECTED_GRSI_HASH"
date
nvidia-smi -L
"$PYTHON_BIN" -c 'import sys; print(sys.executable); print(sys.version)'

if [[ -z "${WANDB_ENTITY:-}" ]]; then
  unset WANDB_ENTITY
fi
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  unset WANDB_RUN_ID
fi

train_cmd=(
  "$PYTHON_BIN"
  scripts/fast_sac/train.py
  --task "$TASK_ID"
  --run_name "$RUN_NAME"
  --motion_file "$MOTION_FILE"
  --num_envs "$NUM_ENVS"
  --eval_episodes "$EVAL_EPISODES"
  --checkpoint_interval "$CHECKPOINT_INTERVAL"
  --train_steps "$TRAIN_STEPS"
  --progress_interval "$PROGRESS_INTERVAL"
  --curve_interval "$CURVE_INTERVAL"
  --logger "$LOGGER"
  --wandb_project "$WANDB_PROJECT"
  --wandb_mode "$WANDB_MODE"
  --wandb_group "$WANDB_GROUP"
  --wandb_tags "$WANDB_TAGS"
  --wandb_resume "$WANDB_RESUME"
  --headless
  env.scene.terrain.visual_material=null
  env.commands.motion.debug_vis=false
  env.scene.contact_forces.debug_vis=false
)

if [[ -n "$resume_checkpoint" ]]; then
  train_cmd+=(--resume_checkpoint "$resume_checkpoint")
fi
if [[ -n "$MAX_STEPS" ]]; then
  train_cmd+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$WARMUP_STEPS" ]]; then
  train_cmd+=(--warmup_steps "$WARMUP_STEPS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  train_cmd+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$UPDATES_PER_STEP" ]]; then
  train_cmd+=(--updates_per_step "$UPDATES_PER_STEP")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  train_cmd+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  train_cmd+=(--wandb_run_id "$WANDB_RUN_ID")
fi
if [[ "$SKIP_EVAL" == "1" ]]; then
  train_cmd+=(--skip_eval)
fi

"${train_cmd[@]}"

if [[ "$SKIP_ACCEPTANCE" == "1" ]]; then
  echo "Skipping acceptance because SKIP_ACCEPTANCE=1"
  exit 0
fi

run_dir_name="$(find_latest_run_dir "$RUN_NAME")"
if [[ -z "$run_dir_name" ]]; then
  echo "Unable to locate completed run directory for $RUN_NAME" >&2
  exit 1
fi
run_dir="$log_root/$run_dir_name"

echo "run_dir=$run_dir"

"$PYTHON_BIN" scripts/fast_sac/accept_run.py \
  --stage "$STAGE_LABEL" \
  --log_dir "$run_dir" \
  --expected_task_id "$EXPECTED_TASK_ID" \
  --expected_contract_id "$EXPECTED_CONTRACT_ID" \
  --expected_recipe_contract_id "$EXPECTED_RECIPE_CONTRACT_ID" \
  --expected_grsi_path "$EXPECTED_GRSI_PATH" \
  --expected_grsi_hash "$EXPECTED_GRSI_HASH"

if [[ ! -f "$run_dir/accepted_artifact.json" ]]; then
  echo "Acceptance failed for $run_dir" >&2
  if [[ -f "$run_dir/acceptance_blocker.json" ]]; then
    cat "$run_dir/acceptance_blocker.json" >&2
  fi
  exit 1
fi

echo "accepted_artifact=$run_dir/accepted_artifact.json"
