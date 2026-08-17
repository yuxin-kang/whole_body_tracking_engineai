#!/bin/bash
#SBATCH --job-name=t800_6act_a4000
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --array=0-5
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
TASK="Tracking-Flat-T800-v0"
MOTION_DIR="data/npz/traj_eng_50hz_canonical"

actions=(
  left_front_kick_002
  roundhouse_kick_001
  straight_punch_R
  straight_punch_L
  left_hook_001
  rear_hook_001
)
motions=(
  "$MOTION_DIR/left_front_kick_002.npz"
  "$MOTION_DIR/roundhouse_kick_001.npz"
  "$MOTION_DIR/straight_punch_R.npz"
  "$MOTION_DIR/straight_punch_L.npz"
  "$MOTION_DIR/left_hook_001.npz"
  "$MOTION_DIR/rear_hook_001.npz"
)

action="${actions[$SLURM_ARRAY_TASK_ID]}"
motion="${motions[$SLURM_ARRAY_TASK_ID]}"
run_name="${action}_canonical_a4000_j${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
export WANDB_USERNAME="gkmbgm-northeastern-university"
export WANDB_ENTITY="$WANDB_USERNAME"
export WANDB_RUN_GROUP="t800-canonical-a4000-j${SLURM_ARRAY_JOB_ID}"
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
if [[ ! -r "$REPO_DIR/$motion" ]]; then
  echo "Canonical motion file is unavailable: $REPO_DIR/$motion" >&2
  exit 3
fi

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

# Array tasks can be scheduled together; stagger cold Isaac Sim starts.
startup_delay=$((SLURM_ARRAY_TASK_ID * 30))
echo "startup_delay_seconds=$startup_delay"
sleep "$startup_delay"

cd "$REPO_DIR"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "node=$(hostname)"
echo "action=$action"
echo "task=$TASK"
echo "motion=$motion"
echo "run_name=$run_name"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_run_group=$WANDB_RUN_GROUP"
echo "wandb_file_uploads=$WBT_WANDB_UPLOAD_FILES"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "$TASK" \
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
