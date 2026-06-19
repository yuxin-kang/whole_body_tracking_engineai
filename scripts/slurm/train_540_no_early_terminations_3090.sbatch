#!/bin/bash
#SBATCH --job-name=t800_540_noearlyterm_3090
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

TASK="Tracking-Flat-T800-540Huixuanti1-NoEarlyTerminations-v0"
MOTION="data/npz/540/cut/540huixuantitui1.npz"
RUN_NAME="t800-540huixuanti1-no-early-terminations-3090-24h"

export PYTHONUNBUFFERED=1

JOB_HOME="/tmp/wbt_omni_home_${SLURM_JOB_ID:-manual}"
export HOME="$JOB_HOME"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$XDG_DATA_HOME" \
  "$CUDA_CACHE_PATH" \
  "$HOME/Documents" \
  "$HOME/.nvidia-omniverse/logs"

export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"

cd "$REPO_DIR"
mkdir -p slurm_logs

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "job_name=${SLURM_JOB_NAME:-manual}"
echo "node=$(hostname)"
echo "repo=$REPO_DIR"
echo "task=$TASK"
echo "motion=$MOTION"
echo "run_name=$RUN_NAME"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi -L
"$CONDA_ENV/bin/python" -c 'import sys; print(sys.executable); print(sys.version)'

"$CONDA_ENV/bin/python" scripts/rsl_rl/train.py \
  --task "$TASK" \
  --motion_file "$MOTION" \
  --num_envs 4096 \
  --run_name "$RUN_NAME" \
  --headless \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
