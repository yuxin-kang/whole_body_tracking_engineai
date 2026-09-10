#!/bin/bash
#SBATCH --job-name=t800_ridicule_baseline
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail
train_repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
train_env=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
train_isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
cd "$train_repo"
train_motion=${1:-data/npz/ridicule/motion1_t800_50hz.npz}
train_variant=${2:-ridicule}
test -s "$train_motion"
export CONDA_PREFIX="$train_env" PATH="$train_env/bin:$PATH"
export PYTHONPATH="$train_isaac/source/isaaclab:$train_isaac/source/isaaclab_assets:$train_isaac/source/isaaclab_rl:$train_isaac/source/isaaclab_tasks:$train_repo/source/whole_body_tracking"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
train_tmp=$(mktemp -d /tmp/t800-ridicule-XXXXXXXX)
export TMPDIR="$train_tmp" XDG_CACHE_HOME="$train_tmp/cache" XDG_DATA_HOME="$train_tmp/data"
export CUDA_CACHE_PATH="$train_tmp/cuda" MPLCONFIGDIR="$train_tmp/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export NETRC=/srv/shared/home/xjh/.netrc WANDB_MODE=online WANDB_PROJECT=urkl-t800
export WANDB_ENTITY=gkmbgm-northeastern-university
export WANDB_RUN_GROUP=t800-ridicule-baseline WANDB_RUN_ID="ridicule${SLURM_JOB_ID}" WANDB_RESUME=never
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false WBT_WANDB_UPLOAD_FILES=0
"$train_env/bin/python" scripts/rsl_rl/train.py \
  --task Tracking-Flat-T800-v0 \
  --motion_file "$train_motion" \
  --num_envs 4096 --max_iterations 30000 --seed 42 \
  --run_name "t800_${train_variant}_baseline_scratch_3090_j${SLURM_JOB_ID}" \
  --logger wandb --log_project_name urkl-t800 --headless --device cuda:0 \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
  agent.resume=false env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false env.scene.contact_forces.debug_vis=false "${@:3}"
