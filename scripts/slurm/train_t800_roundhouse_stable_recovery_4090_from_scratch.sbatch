#!/bin/bash
# Train the updated T800 roundhouse stable-recovery task from random policy
# initialization on one RTX 4090.  This is an independent experiment; it does
# not load the previous A4000/3090 checkpoint.
#SBATCH --job-name=t800_roundhouse_recovery_4090
#SBATCH --partition=rtx4090
# The rtx4090 partition enforces even GPU allocations.  We reserve the
# smallest accepted allocation and expose only its first GPU below.
#SBATCH --gres=gpu:rtx4090d:2
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
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUN_NAME="${RUN_NAME:-roundhouse_kick_001_stable_recovery_anchor_slip_com_4090_from_scratch_j${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-urkl-t800}"
PROCESS_HOME="/tmp/wbt_${SLURM_JOB_ID}_roundhouse_recovery_4090_from_scratch"

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
export WANDB_RUN_GROUP="t800-roundhouse-stable-recovery-anchor-slip-com-4090-from-scratch"
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
echo "initialization=from_scratch"
echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

# The scheduler requires two GPUs on this partition, but this experiment is
# intentionally single-GPU.  Limit Isaac/PyTorch to the first allocated 4090
# so the second GPU is reserved but not used by the training process.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi
echo "process_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
fi

"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "$TASK" \
  --motion_file "$MOTION_PATH" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --run_name "$RUN_NAME" \
  --logger wandb \
  --log_project_name "$WANDB_PROJECT" \
  --headless \
  --device cuda:0 \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
