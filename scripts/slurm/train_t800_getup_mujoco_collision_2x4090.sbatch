#!/bin/bash
# Two independent from-scratch seeds, one RTX4090D per PPO process.  RSL-RL
# does not use DDP here: each GPU owns its own 4096-environment simulation.
#SBATCH --job-name=t800_getup_mjcollision_scratch
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail

mj_repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
mj_isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
mj_conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
mj_task=Tracking-Flat-T800-GetUp-MujocoCollision-Recovery-Ori-v0
mj_motion=data/npz/t800_get_up/faint_prone_getup_03_t800.npz
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export CONDA_PREFIX="$mj_conda" PATH="$mj_conda/bin:$PATH"
export PYTHONPATH="$mj_isaac/source/isaaclab:$mj_isaac/source/isaaclab_assets:$mj_isaac/source/isaaclab_rl:$mj_isaac/source/isaaclab_tasks:$mj_repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=urkl-t800
export WANDB_ENTITY=gkmbgm-northeastern-university WANDB_USERNAME=gkmbgm-northeastern-university
export WANDB_RUN_GROUP="t800-getup-mujoco-collision-scratch-${SLURM_JOB_ID}"
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false WBT_WANDB_UPLOAD_FILES=0
export NETRC=/srv/shared/home/xjh/.netrc
test -r "$NETRC"
cd "$mj_repo"

if [[ "${1:-}" == "arm" ]]; then
    mj_stage="$2"
    mj_seed="$3"
    mj_label="$4"
    mj_tmp=$(mktemp -d "/tmp/t800-mjcollision-${SLURM_JOB_ID}-${mj_label}.XXXXXX")
    export TMPDIR="$mj_tmp" XDG_CACHE_HOME="$mj_tmp/cache" XDG_DATA_HOME="$mj_tmp/data"
    export CUDA_CACHE_PATH="$mj_tmp/cuda" MPLCONFIGDIR="$mj_tmp/matplotlib" WANDB_DIR="$mj_repo/wandb"
    mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$WANDB_DIR"
    if [[ "$mj_stage" == "preflight" ]]; then
        mj_envs=32
        mj_iters=2
        mj_run="t800_getup_mjcollision_preflight_${mj_label}_j${SLURM_JOB_ID}"
        mj_logger=tensorboard
    elif [[ "$mj_stage" == "train" ]]; then
        mj_envs=4096
        mj_iters=30000
        mj_run="t800_getup_mjcollision_ori_s${mj_seed}_scratch_4090_j${SLURM_JOB_ID}"
        mj_logger=wandb
        export WANDB_RUN_ID="getup_mjcollision_${SLURM_JOB_ID}_${mj_label}" WANDB_RESUME=never
    else
        echo "Unknown stage: $mj_stage" >&2
        exit 2
    fi
    echo "stage=$mj_stage seed=$mj_seed gpu=${CUDA_VISIBLE_DEVICES:-unset} task=$mj_task run=$mj_run"
    "$mj_conda/bin/python" scripts/rsl_rl/train.py \
        --task "$mj_task" --motion_file "$mj_motion" --num_envs "$mj_envs" \
        --seed "$mj_seed" --max_iterations "$mj_iters" --run_name "$mj_run" \
        --logger "$mj_logger" --log_project_name urkl-t800 --headless --device cuda:0 \
        env.scene.terrain.visual_material=null \
        env.commands.motion.debug_vis=false \
        env.scene.contact_forces.debug_vis=false
    exit 0
fi

echo "job=$SLURM_JOB_ID host=$(hostname) allocated_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader

run_pair() {
    local stage="$1" rc_a=0 rc_b=0
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
        --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
        --output="$mj_repo/slurm_logs/getup-mjcollision-${SLURM_JOB_ID}-${stage}-s42.out" \
        --error="$mj_repo/slurm_logs/getup-mjcollision-${SLURM_JOB_ID}-${stage}-s42.err" \
        bash "$0" arm "$stage" 42 s42 &
    local pid_a=$!
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
        --gres=gpu:rtx4090d:1 --kill-on-bad-exit=0 \
        --output="$mj_repo/slurm_logs/getup-mjcollision-${SLURM_JOB_ID}-${stage}-s2026.out" \
        --error="$mj_repo/slurm_logs/getup-mjcollision-${SLURM_JOB_ID}-${stage}-s2026.err" \
        bash "$0" arm "$stage" 2026 s2026 &
    local pid_b=$!
    wait "$pid_a" || rc_a=$?
    wait "$pid_b" || rc_b=$?
    echo "stage=$stage exit_s42=$rc_a exit_s2026=$rc_b"
    (( rc_a == 0 && rc_b == 0 ))
}

# The long jobs are not started until both asset imports and short PPO updates
# complete successfully under the MuJoCo-aligned collision profile.
run_pair preflight
run_pair train
