#!/bin/bash
#SBATCH --job-name=t800_fk_bodyfix
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
TASK="Tracking-Flat-T800-v0"
MOTION="data/npz/traj_eng_50hz_canonical/left_front_kick_002.npz"
RUN_NAME="left_front_kick_002_bodyfix_3090_j${SLURM_JOB_ID}"
PROCESS_HOME="/tmp/wbt_${SLURM_JOB_ID}_left_front_kick_bodyfix"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export http_proxy="http://10.12.120.125:7897"
export https_proxy="http://10.12.120.125:7897"
export HTTP_PROXY="http://10.12.120.125:7897"
export HTTPS_PROXY="http://10.12.120.125:7897"
export WANDB_MODE="online"
export WANDB_PROJECT="urkl-t800"
export WANDB_USERNAME="gkmbgm-northeastern-university"
export WANDB_ENTITY="$WANDB_USERNAME"
export WANDB_RUN_GROUP="t800-front-kick-body-order-fix"
export WANDB_DIR="$REPO_DIR/wandb"
export WANDB_DISABLE_CODE=true
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

if [[ ! -r "$NETRC" ]]; then
  echo "W&B online mode requested, but the credential file is unavailable." >&2
  exit 2
fi
if [[ ! -r "$REPO_DIR/$MOTION" ]]; then
  echo "Canonical motion file is unavailable: $REPO_DIR/$MOTION" >&2
  exit 3
fi

mkdir -p \
  "$REPO_DIR/slurm_logs" \
  "$WANDB_DIR" \
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
echo "motion=$MOTION"
echo "run_name=$RUN_NAME"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_run_group=$WANDB_RUN_GROUP"
echo "wandb_file_uploads=$WBT_WANDB_UPLOAD_FILES"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "$TASK" \
  --motion_file "$MOTION" \
  --num_envs 4096 \
  --max_iterations 30000 \
  --run_name "$RUN_NAME" \
  --logger wandb \
  --log_project_name "$WANDB_PROJECT" \
  --headless \
  --device cuda:0 \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
