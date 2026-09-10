#!/bin/bash
# One-iteration Isaac Sim construction/step smoke for the five target tasks.
# This is intentionally not a training launch and does not enable W&B.
#SBATCH --job-name=t800_target_smoke
#SBATCH --partition=rtx4000
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%A_%a.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"

tasks=(
  Tracking-Flat-T800-Target-StraightPunchL-v0
  Tracking-Flat-T800-Target-StraightPunchR-v0
  Tracking-Flat-T800-Target-LeftHook-v0
  Tracking-Flat-T800-Target-LeftFrontKick-v0
  Tracking-Flat-T800-Target-RoundhouseKick-v0
)
names=(straight_punch_L straight_punch_R left_hook left_front_kick roundhouse_kick)

task="${tasks[$SLURM_ARRAY_TASK_ID]}"
name="${names[$SLURM_ARRAY_TASK_ID]}"
process_home="/tmp/wbt_target_smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
smoke_num_envs="${SMOKE_NUM_ENVS:-1}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
# Kit resolves remote USD references through the proxy configured for this
# workspace.  Set both cases explicitly because urllib/Kit components do not
# consistently prefer the same environment-variable spelling.
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
export HOME="$process_home"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export TMPDIR="$HOME/tmp"

mkdir -p "$HOME/.cache" "$HOME/.local/share" "$HOME/.nv/ComputeCache" "$HOME/tmp" "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

echo "task=$task"
echo "name=$name"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "num_envs=$smoke_num_envs"

"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "$task" \
  --num_envs "$smoke_num_envs" \
  --max_iterations 1 \
  --run_name "${name}_target_smoke" \
  --logger tensorboard \
  --headless \
  --device cuda:0 \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
