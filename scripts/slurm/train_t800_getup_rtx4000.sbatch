#!/bin/bash
# Train the T800 get-up tracking task on one RTX 4000 GPU.
# The motion, run name, environment count, and iteration count may be overridden
# through sbatch --export=ALL,NAME=value or shell environment variables.
#SBATCH --job-name=t800_getup
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"
TASK="${TASK:-Tracking-Flat-T800-GetUp-Smooth-v0}"
MOTION_FILE="${MOTION_FILE:-data/npz/t800_get_up/faint_prone_getup_03_t800.npz}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUN_NAME="${RUN_NAME:-t800_getup_smooth_height_faint_prone_getup_03_rtx4000_j${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-urkl-t800}"
process_home="${PROCESS_HOME:-/tmp/wbt_${SLURM_JOB_ID}_getup}"

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
export WANDB_RUN_GROUP="t800-getup-rtx4000"
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
export MPLCONFIGDIR="$process_home/.cache/matplotlib"
export WANDB_DIR="$process_home/wandb"

mkdir -p \
  "$REPO_DIR/slurm_logs" \
  "$process_home/.cache" \
  "$process_home/.local/share" \
  "$process_home/.nv/ComputeCache" \
  "$process_home/Documents" \
  "$process_home/.nvidia-omniverse/logs" \
  "$process_home/tmp" \
  "$WANDB_DIR"

export HOME="$process_home"
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
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
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
