#!/bin/bash
#SBATCH --job-name=t800_eval_punch_hook
#SBATCH --partition=rtx4090
#SBATCH --nodelist=epyc4
#SBATCH --gres=gpu:rtx4090d:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -uo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"
TASK="Tracking-Flat-T800-v0"

run_eval() {
  local action="$1"
  local motion="$2"
  local run="$3"
  local process_home="/tmp/wbt_eval_${SLURM_JOB_ID}_${action}"

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

  mkdir -p \
    "$REPO_DIR/slurm_logs" \
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
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "action=$action"
  echo "motion=$motion"
  echo "run=$run"
  echo "checkpoint=model_8000.pt"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

  "$PYTHON_BIN" scripts/rsl_rl/play.py \
    --task "$TASK" \
    --motion_file "$motion" \
    --load_run "$run" \
    --checkpoint model_8000.pt \
    --num_envs 1 \
    --play_from_start true \
    --debug_terminations true \
    --debug_tracking true \
    --max_steps 260 \
    --headless \
    --device cuda:0
}

if [[ "${1:-}" == "worker" ]]; then
  shift
  set -euo pipefail
  run_eval "$@"
  exit 0
fi

mkdir -p "$REPO_DIR/slurm_logs"

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G \
  --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
  bash "$0" worker \
    straight_punch_R \
    data/npz/traj_eng_50hz_canonical/straight_punch_R.npz \
    2026-08-11_14-40-56_straight_punch_R_canonical_a4000_j7324_2 \
  >"$REPO_DIR/slurm_logs/t800_eval_straight_punch_R-${SLURM_JOB_ID}.out" 2>&1 &
punch_pid=$!

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G \
  --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
  bash "$0" worker \
    left_hook_001 \
    data/npz/traj_eng_50hz_canonical/left_hook_001.npz \
    2026-08-11_14-41-58_left_hook_001_canonical_a4000_j7324_4 \
  >"$REPO_DIR/slurm_logs/t800_eval_left_hook_001-${SLURM_JOB_ID}.out" 2>&1 &
hook_pid=$!

wait "$punch_pid"
punch_rc=$?
wait "$hook_pid"
hook_rc=$?
echo "eval_exit_codes punch=$punch_rc hook=$hook_rc"

if (( punch_rc != 0 || hook_rc != 0 )); then
  exit 1
fi
