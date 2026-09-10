#!/bin/bash
# Four independent F1--F4 fine-tunes.  No DDP: one process owns one GPU.
# The formal stage starts only after all four preflight steps pass.
#SBATCH --job-name=t800_getup_forearm_finetune
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail

ft_repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
ft_isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
ft_conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
ft_source="$ft_repo/logs/rsl_rl/t800_flat/2026-09-06_16-08-31_t800_getup_aggressive_c_scratch_3090_j9746"
ft_python="$ft_repo/scripts/rsl_rl/finetune_getup_forearm.py"
ft_launcher="$ft_repo/scripts/slurm/finetune_t800_getup_forearm_4x3090.sbatch"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export CONDA_PREFIX="$ft_conda" PATH="$ft_conda/bin:$PATH"
export PYTHONPATH="$ft_isaac/source/isaaclab:$ft_isaac/source/isaaclab_assets:$ft_isaac/source/isaaclab_rl:$ft_isaac/source/isaaclab_tasks:$ft_repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export WANDB_MODE=online WANDB_PROJECT=urkl-t800
export WANDB_ENTITY=gkmbgm-northeastern-university WANDB_USERNAME=gkmbgm-northeastern-university
export WANDB_RUN_GROUP="t800-getup-forearm-f1-f4-${SLURM_JOB_ID}"
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false WBT_WANDB_UPLOAD_FILES=0
# wandb reads credentials from this file; never print or export the token.
export NETRC=/srv/shared/home/xjh/.netrc
test -r "$NETRC"
for ft_required in "$ft_python" "$ft_launcher" "$ft_source/model_18500.pt" \
    "$ft_source/params/env.pkl" "$ft_source/params/agent.pkl"; do
    test -f "$ft_required"
done
cd "$ft_repo"

if [[ "${1:-}" == "arm" ]]; then
    ft_stage="$2"
    ft_variant="$3"
    ft_tmp=$(mktemp -d "/tmp/t800-getup-forearm-${SLURM_JOB_ID}-${ft_variant}.XXXXXX")
    export TMPDIR="$ft_tmp" XDG_CACHE_HOME="$ft_tmp/cache" XDG_DATA_HOME="$ft_tmp/data"
    export CUDA_CACHE_PATH="$ft_tmp/cuda" MPLCONFIGDIR="$ft_tmp/matplotlib" WANDB_DIR="$ft_repo/wandb"
    mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$WANDB_DIR"
    if [[ "$ft_stage" == "preflight" ]]; then
        ft_output="$ft_repo/outputs/diagnostics/getup_forearm_preflight_${SLURM_JOB_ID}/${ft_variant}"
        ft_run_id="getup-forearm-preflight-${SLURM_JOB_ID}-${ft_variant}"
        ft_flags=(--preflight --num-envs 32 --iterations 2)
    elif [[ "$ft_stage" == "train" ]]; then
        ft_stamp=$(date +%Y-%m-%d_%H-%M-%S)
        ft_output="$ft_repo/logs/rsl_rl/t800_flat/${ft_stamp}_t800_getup_forearm_${ft_variant}_finetune_3090_j${SLURM_JOB_ID}"
        ft_run_id="getup-forearm-${SLURM_JOB_ID}-${ft_variant}"
        ft_flags=(--num-envs 4096 --iterations 5000)
    else
        echo "Unknown stage: $ft_stage" >&2
        exit 2
    fi
    export WANDB_RUN_ID="$ft_run_id" WANDB_RESUME=never
    echo "stage=$ft_stage variant=$ft_variant job=$SLURM_JOB_ID output=$ft_output"
    "$ft_conda/bin/python" "$ft_python" \
        --variant "$ft_variant" --source-run "$ft_source" --output "$ft_output" \
        --usd-dir "$ft_tmp/t800_usd" --learning-rate 1e-5 "${ft_flags[@]}" \
        --headless --device cuda:0 \
        --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    "$ft_conda/bin/python" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"] in {"preflight_completed", "completed"}, m["status"]' \
        "$ft_output/finetune_manifest.json"
    exit 0
fi

run_four() {
    local stage="$1"
    local -a pids=()
    local variant
    for variant in f1 f2 f3 f4; do
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
            --gres=gpu:rtx3090:1 --kill-on-bad-exit=0 \
            --output="$ft_repo/slurm_logs/getup-forearm-${SLURM_JOB_ID}-${stage}-${variant}.out" \
            --error="$ft_repo/slurm_logs/getup-forearm-${SLURM_JOB_ID}-${stage}-${variant}.err" \
            bash "$ft_launcher" arm "$stage" "$variant" &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    return "$rc"
}

echo "job=$SLURM_JOB_ID host=$(hostname) allocated_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader
run_four preflight
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "PREFLIGHT_ONLY=1: all four preflights passed; formal training not launched"
    exit 0
fi
run_four train
