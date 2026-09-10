#!/bin/bash
# Export the current latest checkpoints of the two roundhouse experiments.
# The play entry point writes policy.onnx and, when MNN is installed, policy.mnn
# into each run's exported/ directory.
#SBATCH --job-name=t800_roundhouse_export
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"
TASK="Tracking-Flat-T800-Improved-RoundhouseKick-StableRecovery-v0"
MOTION_FILE="data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz"
PROCESS_HOME="/tmp/wbt_${SLURM_JOB_ID}_roundhouse_export"

RUN_3090="2026-09-04_21-00-21_roundhouse_kick_001_stable_recovery_anchor_slip_com_3090_j9672"
RUN_4090="2026-09-04_22-36-30_roundhouse_kick_001_stable_recovery_anchor_slip_com_4090_from_scratch_j9691"

latest_checkpoint() {
  local run_dir="$1"
  find "$REPO_DIR/logs/rsl_rl/t800_flat/$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V | tail -1
}

CHECKPOINT_3090="$(latest_checkpoint "$RUN_3090")"
CHECKPOINT_4090="$(latest_checkpoint "$RUN_4090")"
if [[ -z "$CHECKPOINT_3090" || -z "$CHECKPOINT_4090" ]]; then
  echo "Unable to find both roundhouse checkpoints." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Training Python is unavailable: $PYTHON_BIN" >&2
  exit 3
fi
if [[ ! -r "$REPO_DIR/$MOTION_FILE" ]]; then
  echo "Motion file is unavailable: $REPO_DIR/$MOTION_FILE" >&2
  exit 4
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export http_proxy="http://10.12.120.125:7897"
export https_proxy="http://10.12.120.125:7897"
export HTTP_PROXY="http://10.12.120.125:7897"
export HTTPS_PROXY="http://10.12.120.125:7897"
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="$PROCESS_HOME/.cache/matplotlib"

mkdir -p \
  "$REPO_DIR/slurm_logs" \
  "$PROCESS_HOME/.cache" \
  "$PROCESS_HOME/.local/share" \
  "$PROCESS_HOME/.nv/ComputeCache" \
  "$PROCESS_HOME/Documents" \
  "$PROCESS_HOME/.nvidia-omniverse/logs" \
  "$PROCESS_HOME/tmp"

export HOME="$PROCESS_HOME"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export TMPDIR="$HOME/tmp"

cd "$REPO_DIR"
echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "task=$TASK"
echo "motion=$MOTION_FILE"
echo "checkpoint_3090=$CHECKPOINT_3090"
echo "checkpoint_4090=$CHECKPOINT_4090"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

export_policy() {
  local run_name="$1"
  local checkpoint="$2"
  echo "export_start run=$run_name checkpoint=$checkpoint"
  "$PYTHON_BIN" scripts/rsl_rl/play.py \
    --task "$TASK" \
    --motion_file "$MOTION_FILE" \
    --load_run "$run_name" \
    --checkpoint "$checkpoint" \
    --num_envs 1 \
    --play_from_start true \
    --max_steps 1 \
    --headless \
    --device cuda:0
  echo "export_end run=$run_name"
}

export_policy "$RUN_3090" "$CHECKPOINT_3090"
export_policy "$RUN_4090" "$CHECKPOINT_4090"
