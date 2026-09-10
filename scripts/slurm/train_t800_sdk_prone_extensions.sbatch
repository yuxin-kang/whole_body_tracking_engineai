#!/usr/bin/env bash
# Direct PPO-only launcher for the prone SDK-reference extensions.
#SBATCH --job-name=t800_prone_v34_ppo
#SBATCH --partition=rtx3090
#SBATCH --array=0-1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%A_%a.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%A_%a.err
set -euo pipefail

repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
python_bin="$conda/bin/python"
variants=(V3 V4)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
task="Tracking-Flat-T800-GetUp-SDK-Prone-${variant}-PPO-v0"
run="$(date +%Y%m%d_%H%M%S)_sdk_prone_${variant}_ppo_j${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export CONDA_PREFIX="$conda" PATH="$conda/bin:$PATH"
export PYTHONPATH="$isaac/source/isaaclab:$isaac/source/isaaclab_assets:$isaac/source/isaaclab_rl:$isaac/source/isaaclab_tasks:$isaac/source/isaaclab_mimic:$repo/source/whole_body_tracking"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
scratch=$(mktemp -d /tmp/t800-prone-ppo.XXXXXXXX)
export TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" XDG_DATA_HOME="$scratch/data"
export CUDA_CACHE_PATH="$scratch/cuda" MPLCONFIGDIR="$scratch/mpl" WANDB_DIR="$scratch/wandb"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$WANDB_DIR" "$scratch/usd"
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export NETRC=/srv/shared/home/xjh/.netrc WANDB_PROJECT=urkl-t800 WANDB_ENTITY=gkmbgm-northeastern-university
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false WBT_WANDB_UPLOAD_FILES=0
export WANDB_RUN_GROUP="t800-sdk-prone-ppo-${SLURM_ARRAY_JOB_ID}" WANDB_RUN_ID="$run" WANDB_RESUME=never
test -r "$NETRC" && test -x "$python_bin"
cd "$repo"

preflight="$repo/outputs/prone_ppo_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$preflight"
kit=(--kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8")
WANDB_MODE=disabled "$python_bin" scripts/diagnostics/smoke_sdk_supine_shaping.py \
  --kind prone --variant "$variant" --output "$preflight/smoke.json" \
  --usd-dir "$scratch/usd" --headless --device cuda:0 "${kit[@]}"
"$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$preflight/smoke.json"

export WANDB_MODE=online
"$python_bin" scripts/rsl_rl/train.py \
  --task "$task" \
  --motion_file data/npz/t800_sdk_get_up/T800_prone_to_stance_sdk_clip_grounded_50hz.npz \
  --num_envs 4096 --max_iterations 30000 --seed 42 --run_name "$run" \
  --logger wandb --log_project_name urkl-t800 --headless --device cuda:0 "${kit[@]}" \
  agent.resume=false env.scene.robot.spawn.usd_dir="$scratch/usd" \
  env.scene.robot.spawn.force_usd_conversion=true env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false env.scene.contact_forces.debug_vis=false

run_output=$(find "$repo/logs/rsl_rl/t800_flat" -type d -name "*_${run}" -print -quit)
test -n "$run_output" && test -d "$run_output" && test -s "$run_output/model_29999.pt"
echo "completed task=$task output=$run_output checkpoint=$run_output/model_29999.pt"
