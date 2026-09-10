#!/bin/bash
# Play the final T800 get-up checkpoint on one RTX 4000 GPU and record a video.
#SBATCH --job-name=t800_getup_play
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai"
ISAACLAB_DIR="/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0"
CONDA_ENV="/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking"
PYTHON_BIN="$CONDA_ENV/bin/python"
TASK="${TASK:-Tracking-Flat-T800-GetUp-Smooth-v0}"
MOTION_FILE="${MOTION_FILE:-$REPO_DIR/data/npz/t800_get_up/faint_prone_getup_03_t800.npz}"
RUN_NAME="${RUN_NAME:-}"
CHECKPOINT="${CHECKPOINT:-model_29999.pt}"
process_home="/tmp/wbt_${SLURM_JOB_ID}_getup_play"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Play Python is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -r "$MOTION_FILE" ]]; then
  echo "Motion file is unavailable: $MOTION_FILE" >&2
  exit 3
fi
if [[ -z "$RUN_NAME" ]]; then
  echo "RUN_NAME is required, e.g. RUN_NAME=<training run directory> sbatch scripts/slurm/play_t800_getup_rtx4000.sbatch" >&2
  exit 4
fi
if [[ ! -r "$REPO_DIR/logs/rsl_rl/t800_flat/$RUN_NAME/$CHECKPOINT" ]]; then
  echo "Checkpoint is unavailable: $REPO_DIR/logs/rsl_rl/t800_flat/$RUN_NAME/$CHECKPOINT" >&2
  exit 5
fi

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
export MPLCONFIGDIR="$process_home/.cache/matplotlib"

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
echo "task=$TASK"
echo "motion=$MOTION_FILE"
echo "run=$RUN_NAME"
echo "checkpoint=$CHECKPOINT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
fi

"$PYTHON_BIN" scripts/rsl_rl/play.py \
  --task "$TASK" \
  --motion_file "$MOTION_FILE" \
  --load_run "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  --num_envs 1 \
  --play_from_start true \
  --debug_terminations true \
  --debug_tracking true \
  --video \
  --video_length 200 \
  --max_steps 200 \
  --headless \
  --device cuda:0 \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
