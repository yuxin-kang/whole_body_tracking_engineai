#!/bin/bash
#SBATCH --job-name=t800_2act_tr2
#SBATCH --partition=rtx3090
#SBATCH --nodelist=tr2
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
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
export WANDB_MODE="online"
export WANDB_PROJECT="urkl-t800"
export WANDB_USERNAME="gkmbgm-northeastern-university"
export WANDB_ENTITY="$WANDB_USERNAME"
export WANDB_RUN_GROUP="t800-tr2-remaining"
export WANDB_DIR="$REPO_DIR/wandb"
export NETRC="/srv/shared/home/xjh/.netrc"
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO_DIR/source/whole_body_tracking"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="/srv/shared/home/xjh/.cache/matplotlib"

if [[ ! -r "/srv/shared/home/xjh/.netrc" ]]; then
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

    echo "job_id=$SLURM_JOB_ID"
    echo "node=$(hostname)"
    echo "action=$action"
    echo "task=$TASK"
    echo "motion=$motion"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "wandb_mode=$WANDB_MODE"
    echo "wandb_project=$WANDB_PROJECT"
    echo "wandb_entity=$WANDB_ENTITY"
    date
    nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader

    "$PYTHON_BIN" scripts/rsl_rl/train.py \
      --task "$TASK" \
      --motion_file "$motion" \
      --num_envs 4096 \
      --max_iterations 30000 \
      --run_name "${action}_base3090_tr2_j${SLURM_JOB_ID}" \
      --logger wandb \
      --log_project_name "$WANDB_PROJECT" \
      --headless \
      --device cuda:0 \
      --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
      env.scene.terrain.visual_material=null \
      env.commands.motion.debug_vis=false \
      env.scene.contact_forces.debug_vis=false
  ) >"$process_log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_action "$2" "$3"
  exit $?
fi

echo "launcher_start job_id=$SLURM_JOB_ID node=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi -L

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
  --gres=gpu:rtx3090:1 --cpu-bind=cores \
  /bin/bash "$REPO_DIR/scripts/slurm/train_two_remaining_actions_tr2.sbatch" --worker \
  straight_punch_L data/npz/traj_eng_50hz/straight_punch_L.npz &
straight_pid=$!
# Avoid concurrent cold-start contention in Isaac/Kit caches on tr2. Start the
# second task after the first has returned from SimulationApp initialization.
first_ready=0
for (( attempt=0; attempt<180; attempt++ )); do
  if grep -q "Parsing configuration from" "$REPO_DIR/slurm_logs/straight_punch_L-${SLURM_JOB_ID}.out" 2>/dev/null; then
    first_ready=1
    break
  fi
  if ! kill -0 "$straight_pid" 2>/dev/null; then
    break
  fi
  sleep 5
done
echo "first_action_kit_ready=$first_ready"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
  --gres=gpu:rtx3090:1 --cpu-bind=cores \
  /bin/bash "$REPO_DIR/scripts/slurm/train_two_remaining_actions_tr2.sbatch" --worker \
  rear_hook_001 data/npz/traj_eng_50hz/rear_hook_001.npz &
rear_hook_pid=$!

wait "$straight_pid"
straight_rc=$?
wait "$rear_hook_pid"
rear_hook_rc=$?
echo "launcher_end exit_codes=$straight_rc,$rear_hook_rc"

if (( straight_rc != 0 || rear_hook_rc != 0 )); then
  exit 1
fi
