#!/bin/bash
# Four independent PPO runs: one GPU each, no DDP and no checkpoint restoration.
#SBATCH --job-name=t800_getup_ABCD_scratch
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail

scratch_repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
scratch_isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
scratch_conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
scratch_source="$scratch_repo/logs/rsl_rl/t800_flat/2026-09-02_20-23-46_t800_getup_smooth_height_faint_prone_getup_03_rtx4000_j9519"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export CONDA_PREFIX="$scratch_conda"
export PATH="$scratch_conda/bin:$PATH"
export PYTHONPATH="$scratch_isaac/source/isaaclab:$scratch_isaac/source/isaaclab_assets:$scratch_isaac/source/isaaclab_rl:$scratch_isaac/source/isaaclab_tasks:$scratch_repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=urkl-t800
export WANDB_ENTITY=gkmbgm-northeastern-university WANDB_USERNAME=gkmbgm-northeastern-university
export WANDB_RUN_GROUP="t800-getup-aggressive-scratch-${SLURM_JOB_ID}"
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false WBT_WANDB_UPLOAD_FILES=0
export NETRC=/srv/shared/home/xjh/.netrc
test -r "$NETRC"
test -s "$scratch_source/params/env.pkl"
test -s "$scratch_source/params/agent.pkl"
cd "$scratch_repo"

if [[ "${1:-}" == "arm" ]]; then
    scratch_stage="$2"
    scratch_variant="$3"
    scratch_tmp=$(mktemp -d "/tmp/t800-scratch-${SLURM_JOB_ID}-${scratch_variant}.XXXXXX")
    export TMPDIR="$scratch_tmp" XDG_CACHE_HOME="$scratch_tmp/cache" XDG_DATA_HOME="$scratch_tmp/data"
    export CUDA_CACHE_PATH="$scratch_tmp/cuda" MPLCONFIGDIR="$scratch_tmp/matplotlib"
    export WANDB_DIR="$scratch_repo/wandb"
    mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$WANDB_DIR"
    if [[ "$scratch_stage" == "preflight" ]]; then
        scratch_output="$scratch_repo/outputs/diagnostics/getup_aggressive_preflight_${SLURM_JOB_ID}/${scratch_variant}"
        scratch_flags=(--preflight --num-envs 32 --iterations 24)
    elif [[ "$scratch_stage" == "train" ]]; then
        scratch_stamp=$(date +%Y-%m-%d_%H-%M-%S)
        scratch_output="$scratch_repo/logs/rsl_rl/t800_flat/${scratch_stamp}_t800_getup_aggressive_${scratch_variant}_scratch_3090_j${SLURM_JOB_ID}"
        export WANDB_RUN_ID="getup${SLURM_JOB_ID}_scratch_${scratch_variant}" WANDB_RESUME=never
        scratch_flags=(--num-envs 4096 --iterations 30000)
    else
        echo "Unknown stage: $scratch_stage" >&2
        exit 2
    fi
    echo "stage=$scratch_stage variant=$scratch_variant job=$SLURM_JOB_ID visible_gpu=${CUDA_VISIBLE_DEVICES:-unset} output=$scratch_output"
    "$scratch_conda/bin/python" scripts/rsl_rl/train_getup_aggressive.py \
        --variant "$scratch_variant" --source-run "$scratch_source" --output "$scratch_output" \
        --usd-dir "$scratch_tmp/t800_usd" "${scratch_flags[@]}" --headless --device cuda:0 \
        --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    # Kit shutdown can mask a Python exception with exit 0; require completion evidence.
    "$scratch_conda/bin/python" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"] == "completed", m["status"]' "$scratch_output/scratch_manifest.json"
    exit 0
fi

scratch_script="$scratch_repo/scripts/slurm/train_t800_getup_4x3090_aggressive.sbatch"
echo "job=$SLURM_JOB_ID host=$(hostname) allocated_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader

run_four() {
    local stage="$1" variant pid result=0
    local -a pids=()
    for variant in a b c d; do
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
            --gres=gpu:rtx3090:1 --kill-on-bad-exit=0 \
            --output="$scratch_repo/slurm_logs/getup-${SLURM_JOB_ID}-${stage}-scratch-${variant}.out" \
            --error="$scratch_repo/slurm_logs/getup-${SLURM_JOB_ID}-${stage}-scratch-${variant}.err" \
            bash "$scratch_script" arm "$stage" "$variant" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || result=1
    done
    echo "stage=$stage failed=$result"
    return "$result"
}

# No long training starts unless ALL four full-clip + curriculum short tests pass.
run_four preflight
if [[ "${1:-}" == "preflight-only" ]]; then
    exit 0
fi
run_four train
