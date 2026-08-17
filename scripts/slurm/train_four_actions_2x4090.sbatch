#!/bin/bash
#SBATCH --job-name=t800_6act_2x4090
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -uo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"
TASK="Tracking-Flat-T800-v0"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export http_proxy="http://10.12.120.125:7897"
export https_proxy="http://10.12.120.125:7897"
export HTTP_PROXY="http://10.12.120.125:7897"
export HTTPS_PROXY="http://10.12.120.125:7897"
export WANDB_MODE="${WANDB_MODE:-online}"
# Keep W&B limited to scalar curves/configuration; local model checkpoints
# remain available under logs/ but are not uploaded as run files.
export WBT_WANDB_UPLOAD_FILES=0
export WANDB_PROJECT="${WANDB_PROJECT:-urkl-t800}"
export WANDB_USERNAME="${WANDB_USERNAME:-gkmbgm-northeastern-university}"
export WANDB_ENTITY="${WANDB_ENTITY:-$WANDB_USERNAME}"
export WANDB_GROUP="${WANDB_GROUP:-t800-base-6act-j${SLURM_JOB_ID}}"
export WANDB_DIR="${WANDB_DIR:-$REPO_DIR/wandb}"
export NETRC="/srv/shared/home/xjh/.netrc"
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"

if [[ "$WANDB_MODE" == "online" && ! -r "/srv/shared/home/xjh/.netrc" ]]; then
  echo "W&B online mode requested, but the credential file is unavailable." >&2
  exit 2
fi

mkdir -p "$REPO_DIR/slurm_logs" "$WANDB_DIR"
cd "$REPO_DIR"

run_action() {
  local action="$1"
  local motion="$2"
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
    # Three Isaac Lab processes share each allocated GPU. Keep each process
    # from oversubscribing the worker's CPU allocation.
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4

    echo "job_id=$SLURM_JOB_ID"
    echo "node=$(hostname)"
    echo "action=$action"
    echo "task=$TASK"
    echo "motion=$motion"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "python=$PYTHON_BIN"
    echo "wandb_mode=$WANDB_MODE"
    echo "wandb_file_uploads=$WBT_WANDB_UPLOAD_FILES"
    echo "wandb_project=$WANDB_PROJECT"
    echo "wandb_entity=$WANDB_ENTITY"
    echo "wandb_group=$WANDB_GROUP"
    date
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

    "$PYTHON_BIN" scripts/rsl_rl/train.py \
      --task "$TASK" \
      --motion_file "$motion" \
      --num_envs 4096 \
      --max_iterations 30000 \
      --run_name "${action}_base4090_j${SLURM_JOB_ID}" \
      --logger wandb \
      --log_project_name "$WANDB_PROJECT" \
      --headless \
      --device cuda:0 \
      env.scene.terrain.visual_material=null \
      env.commands.motion.debug_vis=false \
      env.scene.contact_forces.debug_vis=false
  ) >"$process_log" 2>&1
}

run_triple() {
  local first_action="$1"
  local first_motion="$2"
  local second_action="$3"
  local second_motion="$4"
  local third_action="$5"
  local third_motion="$6"
  local first_pid second_pid third_pid first_rc second_rc third_rc

  echo "triple_start node=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} actions=$first_action,$second_action,$third_action"
  run_action "$first_action" "$first_motion" &
  first_pid=$!
  run_action "$second_action" "$second_motion" &
  second_pid=$!
  run_action "$third_action" "$third_motion" &
  third_pid=$!

  wait "$first_pid"
  first_rc=$?
  wait "$second_pid"
  second_rc=$?
  wait "$third_pid"
  third_rc=$?
  echo "triple_end actions=$first_action,$second_action,$third_action exit_codes=$first_rc,$second_rc,$third_rc"

  if (( first_rc != 0 || second_rc != 0 || third_rc != 0 )); then
    return 1
  fi
}

if [[ "${1:-}" == "triple" ]]; then
  shift
  run_triple "$@"
  exit $?
fi

echo "launcher_start job_id=$SLURM_JOB_ID node=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi -L

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=48G \
  --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
  bash "$0" triple \
    straight_punch_R data/npz/traj_eng_50hz/straight_punch_R.npz \
    left_hook_001 data/npz/traj_eng_50hz/left_hook_001.npz \
    left_front_kick_002 data/npz/traj_eng_50hz/left_front_kick_002.npz &
triple_0_pid=$!

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=48G \
  --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
  bash "$0" triple \
    roundhouse_kick_001 data/npz/traj_eng_50hz/roundhouse_kick_001.npz \
    straight_punch_L data/npz/traj_eng_50hz/straight_punch_L.npz \
    rear_hook_001 data/npz/traj_eng_50hz/rear_hook_001.npz &
triple_1_pid=$!

wait "$triple_0_pid"
triple_0_rc=$?
wait "$triple_1_pid"
triple_1_rc=$?
echo "launcher_end triple_exit_codes=$triple_0_rc,$triple_1_rc"

if (( triple_0_rc != 0 || triple_1_rc != 0 )); then
  exit 1
fi
