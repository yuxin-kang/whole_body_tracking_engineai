#!/bin/bash
# Two independent from-scratch mixed-wall T800 kick trainings: one per 4090.
# Preflights are serialized and both must leave passed=true markers before PPO.
# This launcher never changes HOME; every process gets isolated Kit/USD/cache/tmp paths.
#SBATCH --job-name=t800_wall_depth_2x4090
#SBATCH --partition=rtx4090
#SBATCH --account=users
#SBATCH --qos=csc101_gpu_unlimited
#SBATCH --gres=gpu:rtx4090d:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"
NUM_ENVS=4096
MAX_ITERATIONS=30000
WANDB_PROJECT="${WANDB_PROJECT:-urkl-t800}"
WANDB_GROUP="t800_mixedwall50_depth2p5_10cm_mu1_1p7_scratch_4090_j${SLURM_JOB_ID}"
PREFLIGHT_ROOT="$REPO_DIR/logs/preflight/t800_mixedwall50_depth2p5_10cm_mu1_1p7_scratch_4090_j${SLURM_JOB_ID}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Training Python is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
mkdir -p "$REPO_DIR/slurm_logs" "$PREFLIGHT_ROOT"

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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT
export WANDB_ENTITY="gkmbgm-northeastern-university"
export WANDB_RUN_GROUP="$WANDB_GROUP"
export WANDB_DISABLE_CODE=true
export WANDB_SAVE_CODE=false
export WANDB_LOG_MODEL=false
export WANDB_IGNORE_GLOBS="*.pt,model_*.pt,*.onnx"
export WBT_WANDB_UPLOAD_FILES=0
export NETRC=/srv/shared/home/xjh/.netrc
test -r "$NETRC"

cd "$REPO_DIR"

run_preflight() {
  local kind="$1" task="$2" motion="$3"
  local scratch
  scratch="$(mktemp -d "/tmp/wbt_${SLURM_JOB_ID}_${kind}.XXXXXX")"
  mkdir -p "$scratch"/{cache,data,usd,tmp,kit_cache,omniverse,matplotlib}
  export XDG_CACHE_HOME="$scratch/cache"
  export XDG_DATA_HOME="$scratch/data"
  export CUDA_CACHE_PATH="$scratch/cache/cuda"
  export TMPDIR="$scratch/tmp"
  export MPLCONFIGDIR="$scratch/matplotlib"
  export OMNI_USER_PATH="$scratch/omniverse"
  export OMNI_CACHE_PATH="$scratch/kit_cache"
  export OMNI_KIT_CACHE_PATH="$scratch/kit_cache"
  export PREFLIGHT_SCRATCH="$scratch"
  echo "preflight kind=$kind task=$task scratch=$scratch cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --gres=gpu:rtx4090d:1 --gpu-bind=single:1 \
    "$PYTHON_BIN" scripts/diagnostics/smoke_mixed_wall_kicks.py \
    --task "$task" \
    --motion-file "$REPO_DIR/$motion" \
    --output "$PREFLIGHT_ROOT/${kind}.json" \
    --usd-dir "$scratch/usd" \
    --num-envs 32 \
    --device cuda:0 \
    --headless \
    --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
  [[ -s "$PREFLIGHT_ROOT/${kind}.json" ]] || { echo "missing preflight marker: $kind" >&2; return 1; }
  "$PYTHON_BIN" - "$PREFLIGHT_ROOT/${kind}.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    marker = json.load(stream)
if marker.get("passed") is not True:
    raise SystemExit("preflight marker is not passed=true")
PY
}

run_preflight left_front_kick \
  Tracking-Flat-T800-Target-MixedWall-LeftFrontKick-v0 \
  data/npz/traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz
run_preflight roundhouse_kick \
  Tracking-Flat-T800-Target-MixedWall-RoundhouseKick-v0 \
  data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz

launch_training() {
  local kind="$1" task="$2" motion="$3"
  local run_name="t800_${kind}_mixedwall50_depth2p5_10cm_mu1_1p7_scratch_4090_j${SLURM_JOB_ID}"
  local wandb_id="${run_name}"
  local scratch
  scratch="$(mktemp -d "/tmp/wbt_${SLURM_JOB_ID}_${kind}.XXXXXX")"
  mkdir -p "$scratch"/{cache,data,usd,tmp,kit_cache,omniverse,matplotlib,wandb}
  export XDG_CACHE_HOME="$scratch/cache"
  export XDG_DATA_HOME="$scratch/data"
  export CUDA_CACHE_PATH="$scratch/cache/cuda"
  export TMPDIR="$scratch/tmp"
  export MPLCONFIGDIR="$scratch/matplotlib"
  export OMNI_USER_PATH="$scratch/omniverse"
  export OMNI_CACHE_PATH="$scratch/kit_cache"
  export OMNI_KIT_CACHE_PATH="$scratch/kit_cache"
  export WANDB_DIR="$scratch/wandb"
  export WANDB_RUN_ID="$wandb_id"
  export WANDB_RESUME=never
  echo "training kind=$kind task=$task run_name=$run_name scratch=$scratch cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --gres=gpu:rtx4090d:1 --gpu-bind=single:1 \
    "$PYTHON_BIN" scripts/rsl_rl/train.py \
    --task "$task" \
    --motion_file "$REPO_DIR/$motion" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --run_name "$run_name" \
    --logger wandb \
    --log_project_name "$WANDB_PROJECT" \
    --headless \
    --device cuda:0 \
    --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
    "env.scene.robot.spawn.usd_dir=$scratch/usd" \
    agent.resume=false \
    env.scene.terrain.visual_material=null \
    env.commands.motion.debug_vis=false \
    env.scene.contact_forces.debug_vis=false
}

set +e
launch_training left_front_kick \
  Tracking-Flat-T800-Target-MixedWall-LeftFrontKick-v0 \
  data/npz/traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz \
  >"$REPO_DIR/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-left_front_kick.runtime.log" 2>&1 &
front_pid=$!
launch_training roundhouse_kick \
  Tracking-Flat-T800-Target-MixedWall-RoundhouseKick-v0 \
  data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz \
  >"$REPO_DIR/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-roundhouse_kick.runtime.log" 2>&1 &
round_pid=$!
wait "$front_pid"; front_status=$?
wait "$round_pid"; round_status=$?
set -e

require_training_evidence() {
  local kind="$1" status="$2"
  local run_name="t800_${kind}_mixedwall50_depth2p5_10cm_mu1_1p7_scratch_4090_j${SLURM_JOB_ID}"
  local runtime_log="$REPO_DIR/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${kind}.runtime.log"
  [[ "$status" -eq 0 ]] || { echo "training process failed: $kind status=$status" >&2; return 1; }
  rg -q "Logging experiment in directory" "$runtime_log" || {
    echo "training did not provide logging-start evidence: $kind" >&2; return 1;
  }
  rg -q "Learning iteration" "$runtime_log" || {
    echo "training did not provide learning-iteration evidence: $kind" >&2; return 1;
  }
  local checkpoint
  checkpoint="$(find "$REPO_DIR/logs/rsl_rl" -type f -name model_29999.pt -path "*${run_name}*" -print -quit)"
  [[ -n "$checkpoint" ]] || { echo "missing final checkpoint model_29999.pt: $kind" >&2; return 1; }
  echo "verified kind=$kind final_checkpoint=$checkpoint"
}

require_training_evidence left_front_kick "$front_status"
require_training_evidence roundhouse_kick "$round_status"
echo "completed mixed-wall training job ${SLURM_JOB_ID} with both model_29999.pt checkpoints"
