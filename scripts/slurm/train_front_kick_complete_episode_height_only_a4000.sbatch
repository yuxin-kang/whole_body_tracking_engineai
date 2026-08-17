#!/bin/bash
#SBATCH --job-name=t800_frontkick_heightfall
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
TASK="Tracking-Flat-T800-Baseline-LeftFrontKick-CompleteEpisode-HeightFallOnly-v0"
process_home="/tmp/wbt_${SLURM_JOB_ID}_frontkick_heightfall"

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
export WANDB_RUN_GROUP="t800-frontkick-heightfall-j${SLURM_JOB_ID}"
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

mkdir -p \
  "$REPO_DIR/slurm_logs" \
  "$WANDB_DIR" \
  "$process_home/.cache" \
  "$process_home/.local/share" \
  "$process_home/.nv/ComputeCache" \
  "$process_home/Documents" \
  "$process_home/.nvidia-omniverse/logs" \
  "$process_home/tmp"

export HOME="$process_home"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export TMPDIR="$HOME/tmp"

cd "$REPO_DIR"
"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs 4096 \
  --max_iterations 30000 \
  --run_name "left_front_kick_002_complete_episode_heightfall_a4000_j${SLURM_JOB_ID}" \
  --logger wandb \
  --log_project_name "$WANDB_PROJECT" \
  --headless \
  --device cuda:0 \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
