#!/bin/bash
# Fine-tune the T800 roundhouse policy with stronger landing/standing rewards.
# The base checkpoint is the latest stable-recovery policy; the new run keeps
# the prior experiment intact and records the changed reward configuration.
#SBATCH --job-name=t800_roundhouse_recovery_3090
#SBATCH --partition=rtx3090
# tr2 currently exposes a failed GPU device; use the healthy 3090 nodes.
#SBATCH --exclude=tr2
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"

TASK="${TASK:-Tracking-Flat-T800-Improved-RoundhouseKick-StableRecovery-v0}"
MOTION_FILE="${MOTION_FILE:-data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-15000}"
RESUME="${RESUME:-true}"
BASE_RUN="${BASE_RUN:-2026-09-03_12-58-19_roundhouse_kick_001_stable_recovery_finetune_a4000_j9567}"
CHECKPOINT="${CHECKPOINT:-model_44998.pt}"
RUN_NAME="${RUN_NAME:-roundhouse_kick_001_stable_recovery_anchor_slip_com_3090_j${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-urkl-t800}"
PROCESS_HOME="/tmp/wbt_${SLURM_JOB_ID}_roundhouse_recovery_3090"

if [[ "$MOTION_FILE" = /* ]]; then
  MOTION_PATH="$MOTION_FILE"
else
  MOTION_PATH="$REPO_DIR/$MOTION_FILE"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Training Python is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -r "$MOTION_PATH" ]]; then
  echo "Motion file is unavailable: $MOTION_PATH" >&2
  exit 3
fi
if [[ "$WANDB_MODE" == "online" && ! -r /srv/shared/home/xjh/.netrc ]]; then
  echo "W&B credential file is unavailable: /srv/shared/home/xjh/.netrc" >&2
  exit 4
fi
if [[ "$RESUME" == "true" && ! -r "$REPO_DIR/logs/rsl_rl/t800_flat/$BASE_RUN/$CHECKPOINT" ]]; then
  echo "Resume checkpoint is unavailable: $REPO_DIR/logs/rsl_rl/t800_flat/$BASE_RUN/$CHECKPOINT" >&2
  exit 5
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export http_proxy="http://10.12.120.125:7897"
export https_proxy="http://10.12.120.125:7897"
export HTTP_PROXY="http://10.12.120.125:7897"
export HTTPS_PROXY="http://10.12.120.125:7897"
export WANDB_MODE
export WANDB_PROJECT
export WANDB_ENTITY="gkmbgm-northeastern-university"
export WANDB_RUN_GROUP="t800-roundhouse-stable-recovery-anchor-slip-com-3090"
export WANDB_DISABLE_CODE=true
export WANDB_SAVE_CODE=false
export WANDB_LOG_MODEL=false
export WANDB_IGNORE_GLOBS="*.pt,model_*.pt,*.onnx"
export WBT_WANDB_UPLOAD_FILES=0
export NETRC="/srv/shared/home/xjh/.netrc"
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="$PROCESS_HOME/.cache/matplotlib"
export WANDB_DIR="$PROCESS_HOME/wandb"

mkdir -p \
  "$REPO_DIR/slurm_logs" \
  "$PROCESS_HOME/.cache" \
  "$PROCESS_HOME/.local/share" \
  "$PROCESS_HOME/.nv/ComputeCache" \
  "$PROCESS_HOME/Documents" \
  "$PROCESS_HOME/.nvidia-omniverse/logs" \
  "$PROCESS_HOME/tmp" \
  "$WANDB_DIR"

export HOME="$PROCESS_HOME"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export TMPDIR="$HOME/tmp"

cd "$REPO_DIR"
echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "partition=$SLURM_JOB_PARTITION"
echo "task=$TASK"
echo "motion=$MOTION_PATH"
echo "run_name=$RUN_NAME"
echo "num_envs=$NUM_ENVS"
echo "max_iterations=$MAX_ITERATIONS"
echo "resume=$RESUME"
echo "base_run=$BASE_RUN"
echo "checkpoint=$CHECKPOINT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
fi

train_cmd=(
  "$PYTHON_BIN" scripts/rsl_rl/train.py
  --task "$TASK"
  --motion_file "$MOTION_PATH"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
  --run_name "$RUN_NAME"
  --logger wandb
  --log_project_name "$WANDB_PROJECT"
  --headless
  --device cuda:0
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
  env.scene.terrain.visual_material=null
  env.commands.motion.debug_vis=false
  env.scene.contact_forces.debug_vis=false
)

if [[ "$RESUME" == "true" ]]; then
  train_cmd+=(--resume True --load_run "$BASE_RUN" --checkpoint "$CHECKPOINT")
fi

"${train_cmd[@]}"
