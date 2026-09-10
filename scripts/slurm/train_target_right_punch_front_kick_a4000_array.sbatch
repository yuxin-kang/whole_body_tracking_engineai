#!/bin/bash
# Train right straight-punch and left-front-kick tracking with fixed compliant
# target walls.  One array element consumes one RTX A4000 GPU.
# W&B receives scalar curves/configuration only; checkpoints stay local.
#SBATCH --job-name=t800_target_a4000
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --array=0-1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%A_%a.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"

actions=(
  straight_punch_R_target
  left_front_kick_002_target
)
tasks=(
  Tracking-Flat-T800-Target-StraightPunchR-v0
  Tracking-Flat-T800-Target-LeftFrontKick-v0
)
motions=(
  data/npz/traj_eng_50hz_improved/straight_punch_R_terminal_hold_0p5s.npz
  data/npz/traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz
)

action="${actions[$SLURM_ARRAY_TASK_ID]}"
task="${tasks[$SLURM_ARRAY_TASK_ID]}"
motion="${motions[$SLURM_ARRAY_TASK_ID]}"
run_name="${action}_a4000_j${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
process_home="/tmp/wbt_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${action}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export http_proxy="http://10.12.120.125:7897"
export https_proxy="http://10.12.120.125:7897"
export HTTP_PROXY="http://10.12.120.125:7897"
export HTTPS_PROXY="http://10.12.120.125:7897"

export WANDB_MODE="online"
export WANDB_PROJECT="urkl-t800"
export WANDB_ENTITY="gkmbgm-northeastern-university"
export WANDB_RUN_GROUP="t800-target-a4000-j${SLURM_ARRAY_JOB_ID}"
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
export MPLCONFIGDIR="/srv/shared/home/xjh/.cache/matplotlib"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Training Python is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -r "$NETRC" ]]; then
  echo "W&B credential file is unavailable: $NETRC" >&2
  exit 3
fi
if [[ ! -r "$REPO_DIR/$motion" ]]; then
  echo "Target motion file is unavailable: $REPO_DIR/$motion" >&2
  exit 4
fi

mkdir -p \
  "$REPO_DIR/slurm_logs" \
  "$process_home/.cache" \
  "$process_home/.local/share" \
  "$process_home/.nv/ComputeCache" \
  "$process_home/Documents" \
  "$process_home/.nvidia-omniverse/logs" \
  "$process_home/tmp" \
  "$process_home/wandb"

export HOME="$process_home"
export WANDB_DIR="$process_home/wandb"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export TMPDIR="$HOME/tmp"

# Stagger Isaac Sim cold starts enough to avoid a shared Kit/URDF conversion
# allocator race on tr1; the array elements still use separate A4000 GPUs.
startup_delay=$((SLURM_ARRAY_TASK_ID * 120))
echo "startup_delay_seconds=$startup_delay"
sleep "$startup_delay"

cd "$REPO_DIR"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "node=$(hostname)"
echo "action=$action"
echo "task=$task"
echo "motion=$motion"
echo "run_name=$run_name"
echo "episode_mode=native_target"
echo "num_envs=4096"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_run_group=$WANDB_RUN_GROUP"
echo "wandb_file_uploads=$WBT_WANDB_UPLOAD_FILES"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "$task" \
  --motion_file "$motion" \
  --num_envs 4096 \
  --max_iterations 30000 \
  --run_name "$run_name" \
  --logger wandb \
  --log_project_name "$WANDB_PROJECT" \
  --headless \
  --device cuda:0 \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
