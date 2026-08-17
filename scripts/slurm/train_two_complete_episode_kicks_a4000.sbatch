#!/bin/bash
#SBATCH --job-name=t800_kicks_complete_ep
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --gres=gpu:rtx4000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -uo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"

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
export WANDB_RUN_GROUP="t800-complete-episode-kicks-j${SLURM_JOB_ID}"
export WANDB_DIR="$REPO_DIR/wandb"
export WANDB_DISABLE_CODE=true
export WANDB_IGNORE_GLOBS="*.pt,model_*.pt,*.onnx"
export WBT_WANDB_UPLOAD_FILES=0
export NETRC="/srv/shared/home/xjh/.netrc"
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"

mkdir -p "$REPO_DIR/slurm_logs" "$WANDB_DIR"
cd "$REPO_DIR"

run_task() {
  local task="$1"
  local action="$2"
  local process_home="/tmp/wbt_${SLURM_JOB_ID}_${action}"
  local process_log="$REPO_DIR/slurm_logs/${action}-${SLURM_JOB_ID}.out"

  mkdir -p \
    "$process_home/.cache" \
    "$process_home/.local/share" \
    "$process_home/.nv/ComputeCache" \
    "$process_home/Documents" \
    "$process_home/.nvidia-omniverse/logs" \
    "$process_home/tmp"

  (
    export HOME="$process_home"
    export XDG_CACHE_HOME="$HOME/.cache"
    export XDG_DATA_HOME="$HOME/.local/share"
    export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
    export TMPDIR="$HOME/tmp"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export NUMEXPR_NUM_THREADS=8

    "$PYTHON_BIN" scripts/rsl_rl/train.py \
      --task "$task" \
      --num_envs 4096 \
      --max_iterations 30000 \
      --run_name "${action}_complete_episode_hold1s_a4000_j${SLURM_JOB_ID}" \
      --logger wandb \
      --log_project_name "$WANDB_PROJECT" \
      --headless \
      --device cuda:0 \
      env.scene.terrain.visual_material=null \
      env.commands.motion.debug_vis=false \
      env.scene.contact_forces.debug_vis=false
  ) >"$process_log" 2>&1
}

if [[ "${1:-}" == "worker" ]]; then
  shift
  run_task "$@"
  exit $?
fi

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G \
  --gres=gpu:rtx4000:1 --kill-on-bad-exit=0 \
  bash "$0" worker \
    Tracking-Flat-T800-Baseline-LeftFrontKick-CompleteEpisode-v0 \
    left_front_kick_002 &
front_pid=$!

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G \
  --gres=gpu:rtx4000:1 --kill-on-bad-exit=0 \
  bash "$0" worker \
    Tracking-Flat-T800-Baseline-RoundhouseKick-CompleteEpisode-v0 \
    roundhouse_kick_001 &
roundhouse_pid=$!

wait "$front_pid"
front_rc=$?
wait "$roundhouse_pid"
roundhouse_rc=$?

if (( front_rc != 0 || roundhouse_rc != 0 )); then
  exit 1
fi
