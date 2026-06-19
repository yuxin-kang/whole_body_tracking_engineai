#!/bin/bash
# FastSAC debug-campaign launcher (run inside srun on a GPU node).
# Usage: run_dbg.sh <run_name> [extra train.py args + agent.* overrides...]
# Sets up the IsaacLab env with an ISOLATED omniverse HOME so parallel jobs
# don't fight over the shared ~/.nvidia-omniverse cache.
set -euo pipefail

REPO=/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai
ISAACLAB_DIR=/srv/shared/home/kyx/Workspace/IsaacLab
CONDA_ENV=/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid
PY="$CONDA_ENV/bin/python"

RUN_NAME="$1"; shift

# Per-job isolated HOME for omniverse/nv caches (avoids parallel-job contention)
JOB_HOME="/tmp/wbt_omni_home_${SLURM_JOB_ID:-manual}_${RUN_NAME}"
export HOME="$JOB_HOME"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_RUNTIME_DIR="$HOME/runtime"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_RUNTIME_DIR" \
  "$CUDA_CACHE_PATH" "$HOME/Documents" "$HOME/.nvidia-omniverse/logs"
chmod 700 "$XDG_RUNTIME_DIR"

export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="/tmp/wbt-minq-pkg:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_mimic:$REPO/source/whole_body_tracking"
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "[run_dbg] node=$(hostname) run_name=$RUN_NAME home=$HOME"
nvidia-smi -L || true
date

"$PY" scripts/fast_sac/train.py \
  --run_name "$RUN_NAME" \
  --headless \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false \
  "$@"
echo "[run_dbg] DONE run_name=$RUN_NAME rc=$?"
