#!/bin/bash
# Bounded policy replay only; no training, export, or HOME reassignment.
#SBATCH --job-name=t800_getup_diag
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail
diag_repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
diag_isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
diag_conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
diag_tmp=$(mktemp -d /tmp/t800-getup-diag.XXXXXX)
diag_output="$diag_repo/outputs/diagnostics/j9519_play_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export CONDA_PREFIX="$diag_conda"
export PATH="$diag_conda/bin:$PATH"
export PYTHONPATH="$diag_isaac/source/isaaclab:$diag_isaac/source/isaaclab_assets:$diag_isaac/source/isaaclab_rl:$diag_isaac/source/isaaclab_tasks:$diag_repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TMPDIR="$diag_tmp" MPLCONFIGDIR="$diag_tmp/matplotlib" CUDA_CACHE_PATH="$diag_tmp/cuda"
export XDG_CACHE_HOME="$diag_tmp/cache" XDG_DATA_HOME="$diag_tmp/data"
mkdir -p "$diag_output" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
cd "$diag_repo"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$diag_conda/bin/python" scripts/diagnostics/play_getup_tracking.py \
    --run "$diag_repo/logs/rsl_rl/t800_flat/2026-09-02_20-23-46_t800_getup_smooth_height_faint_prone_getup_03_rtx4000_j9519" \
    --checkpoint model_29999.pt --output "$diag_output" --usd-dir "$diag_tmp/t800_usd" \
    --episodes 3 --seed 42 --video --headless --device cuda:0 \
    --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
test -s "$diag_output/rollout.npz"
"$diag_conda/bin/python" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("episodes_completed")==3, m' "$diag_output/metadata.json"
