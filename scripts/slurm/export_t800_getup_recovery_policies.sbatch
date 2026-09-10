#!/usr/bin/env bash
# Export the two latest get-up recovery fine-tunes to ONNX/MNN.
# The exported actor uses the same 134-observation/25-action contract as the
# existing SDK trajectory runner (command, anchor orientation, base velocity,
# joint state and previous actions).
#SBATCH --job-name=t800_getup_export
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
python_bin=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking/bin/python
scratch=$(mktemp -d /tmp/wbt_getup_export_${SLURM_JOB_ID}_XXXXXX)
checkpoint="${1:-model_34999.pt}"

export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export CONDA_PREFIX=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$isaac/source/isaaclab:$isaac/source/isaaclab_assets:$isaac/source/isaaclab_rl:$isaac/source/isaaclab_tasks:$isaac/source/isaaclab_mimic:$repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME="$scratch/cache" XDG_DATA_HOME="$scratch/data"
export CUDA_CACHE_PATH="$scratch/cuda" MPLCONFIGDIR="$scratch/mpl" TMPDIR="$scratch/tmp"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"

cd "$repo"
motion=data/npz/t800_get_up/faint_prone_getup_03_t800.npz

export_one() {
    local task="$1" run="$2" checkpoint="$3"
    echo "export_start task=$task run=$run checkpoint=$checkpoint gpu=${CUDA_VISIBLE_DEVICES:-unset}"
    "$python_bin" scripts/rsl_rl/play.py \
        --task "$task" \
        --motion_file "$motion" \
        --load_run "$run" \
        --checkpoint "$checkpoint" \
        --num_envs 1 \
        --play_from_start true \
        --max_steps 1 \
        --headless \
        --device cuda:0
    local output="$repo/logs/rsl_rl/t800_flat/$run/exported"
    test -s "$output/policy.onnx"
    test -s "$output/policy.mnn"
    test -s "$output/deploy_config.yaml"
    sha256sum "$output/policy.onnx" "$output/policy.mnn" "$output/deploy_config.yaml"
    echo "export_complete run=$run output=$output"
}

export_one \
    Tracking-Flat-T800-GetUp-Recovery-Ori-v0 \
    2026-09-06_14-56-36_t800_getup_recovery_ori_finetune_4090_j9740 \
    "$checkpoint"

export_one \
    Tracking-Flat-T800-GetUp-Recovery-OriVelGate-v0 \
    2026-09-06_14-56-36_t800_getup_recovery_ori_velgate_finetune_4090_j9740 \
    "$checkpoint"
