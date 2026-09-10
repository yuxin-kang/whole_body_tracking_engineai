#!/bin/bash
# One allocated GPU per independent experiment; never DDP between A and B.
#SBATCH --job-name=t800_getup_AB_finetune
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail

ft_repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
ft_isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
ft_conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
ft_source="$ft_repo/logs/rsl_rl/t800_flat/2026-09-02_20-23-46_t800_getup_smooth_height_faint_prone_getup_03_rtx4000_j9519"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export CONDA_PREFIX="$ft_conda"
export PATH="$ft_conda/bin:$PATH"
export PYTHONPATH="$ft_isaac/source/isaaclab:$ft_isaac/source/isaaclab_assets:$ft_isaac/source/isaaclab_rl:$ft_isaac/source/isaaclab_tasks:$ft_repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=urkl-t800
export WANDB_ENTITY=gkmbgm-northeastern-university WANDB_USERNAME=gkmbgm-northeastern-university
export WANDB_RUN_GROUP="t800-getup-j9519-recovery-ab-${SLURM_JOB_ID}"
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false
export WBT_WANDB_UPLOAD_FILES=0
export NETRC=/srv/shared/home/xjh/.netrc
test -r "$NETRC"
test -s "$ft_source/model_29999.pt"
cd "$ft_repo"

if [[ "${1:-}" == "arm" ]]; then
    ft_stage="$2"
    ft_variant="$3"
    ft_tmp=$(mktemp -d "/tmp/t800-getup-${SLURM_JOB_ID}-${ft_variant}.XXXXXX")
    export TMPDIR="$ft_tmp" XDG_CACHE_HOME="$ft_tmp/cache" XDG_DATA_HOME="$ft_tmp/data"
    export CUDA_CACHE_PATH="$ft_tmp/cuda" MPLCONFIGDIR="$ft_tmp/matplotlib"
    export WANDB_DIR="$ft_repo/wandb"
    mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$WANDB_DIR"
    ft_flags=()
    if [[ "$ft_stage" == "preflight" ]]; then
        ft_output="$ft_repo/outputs/diagnostics/getup_finetune_preflight_${SLURM_JOB_ID}/${ft_variant}"
        ft_flags=(--preflight --num-envs 32 --iterations 10)
    else
        ft_stamp=$(date +%Y-%m-%d_%H-%M-%S)
        ft_output="$ft_repo/logs/rsl_rl/t800_flat/${ft_stamp}_t800_getup_recovery_${ft_variant}_finetune_4090_j${SLURM_JOB_ID}"
        export WANDB_RUN_ID="getup${SLURM_JOB_ID}_${ft_variant}"
        export WANDB_RESUME=never
        ft_flags=(--num-envs 4096 --iterations 5000)
    fi
    echo "stage=$ft_stage variant=$ft_variant job=$SLURM_JOB_ID visible_gpu=${CUDA_VISIBLE_DEVICES:-unset} output=$ft_output"
    "$ft_conda/bin/python" scripts/rsl_rl/finetune_getup_recovery.py \
        --variant "$ft_variant" --source-run "$ft_source" --output "$ft_output" \
        --usd-dir "$ft_tmp/t800_usd" --learning-rate 1e-5 "${ft_flags[@]}" \
        --headless --device cuda:0 \
        --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    # Kit shutdown can mask a Python exception with exit 0; require proof.
    "$ft_conda/bin/python" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"] == "completed", m["status"]' "$ft_output/finetune_manifest.json"
    exit 0
fi

ft_script="$ft_repo/scripts/slurm/finetune_t800_getup_2x4090.sbatch"
echo "job=$SLURM_JOB_ID host=$(hostname) allocated_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader

run_pair() {
    local stage="$1" pid_a pid_b rc_a=0 rc_b=0
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
        --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
        --output="$ft_repo/slurm_logs/getup-${SLURM_JOB_ID}-${stage}-ori.out" \
        --error="$ft_repo/slurm_logs/getup-${SLURM_JOB_ID}-${stage}-ori.err" \
        bash "$ft_script" arm "$stage" ori &
    pid_a=$!
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
        --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
        --output="$ft_repo/slurm_logs/getup-${SLURM_JOB_ID}-${stage}-ori_velgate.out" \
        --error="$ft_repo/slurm_logs/getup-${SLURM_JOB_ID}-${stage}-ori_velgate.err" \
        bash "$ft_script" arm "$stage" ori_velgate &
    pid_b=$!
    wait "$pid_a" || rc_a=$?
    wait "$pid_b" || rc_b=$?
    echo "stage=$stage exit_a=$rc_a exit_b=$rc_b"
    (( rc_a == 0 && rc_b == 0 ))
}

# Both full-episode smoke runs must pass before either long training starts.
run_pair preflight
run_pair train
