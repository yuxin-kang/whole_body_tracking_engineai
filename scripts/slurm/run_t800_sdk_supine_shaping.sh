#!/usr/bin/env bash
set -euo pipefail

repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
conda=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
python_bin="$conda/bin/python"
job_id=${SLURM_JOB_ID:-manual}
family=${1:?usage: $0 4090|3090 [worker ...]}

case "$family" in
  4090) gpu_type=rtx4090d; variants=(V1 V1 V2 V2) ;;
  3090) gpu_type=rtx3090; variants=(V3 V3 V4 V4) ;;
  *) echo "unsupported GPU family: $family" >&2; exit 2 ;;
esac

stamp=${T800_SHAPING_STAMP:-$(date +%Y%m%d_%H%M%S)}
manifest_dir="$repo/outputs/sdk_supine_shaping_${job_id}"
manifest="$manifest_dir/launch_manifest.json"
mkdir -p "$repo/slurm_logs" "$manifest_dir"

export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export CONDA_PREFIX="$conda" PATH="$conda/bin:$PATH"
export PYTHONPATH="$isaac/source/isaaclab:$isaac/source/isaaclab_assets:$isaac/source/isaaclab_rl:$isaac/source/isaaclab_tasks:$isaac/source/isaaclab_mimic:$repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export NETRC=/srv/shared/home/xjh/.netrc WANDB_MODE=online WANDB_PROJECT=urkl-t800
export WANDB_ENTITY=gkmbgm-northeastern-university WANDB_RUN_GROUP="t800-sdk-supine-shaping-${family}-${job_id}"
export WANDB_DISABLE_CODE=true WANDB_SAVE_CODE=false WANDB_LOG_MODEL=false WBT_WANDB_UPLOAD_FILES=0
test -r "$NETRC"; test -x "$python_bin"

write_manifest() {
  "$python_bin" - "$manifest" "$family" "$job_id" "$stamp" "${variants[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
family, job_id, stamp = sys.argv[2:5]
variants = sys.argv[5:]
entries = []
for gpu, variant in enumerate(variants):
    if 'SLURM_ARRAY_TASK_ID' in os.environ and gpu != int(os.environ['SLURM_ARRAY_TASK_ID']):
        continue
    run_name = f'{stamp}_{variant}_ppo_gpu{gpu}_j{job_id}'
    log_path = f'logs/rsl_rl/t800_flat/*_{run_name}'
    entries.append({
        'gpu_index': gpu, 'variant': variant, 'method': 'ppo', 'run_name': run_name,
        'smoke_marker': f'outputs/sdk_supine_shaping_{job_id}/preflight/{variant}_gpu{gpu}.json',
        'log_path': log_path,
        'slurm_log': f'slurm_logs/t800_sdk_supine_shaping-{job_id}-{variant}-gpu{gpu}.out',
        'final_checkpoint': f'{log_path}/model_29999.pt',
    })
path.write_text(json.dumps({'status': 'launching', 'job_id': job_id,
                            'gpu_family': family, 'stamp': stamp,
                            'entries': entries}, indent=2) + '\n')
PY
}

worker() {
  local variant="$1" gpu="$2" run_name="$3"
  local tag="${variant}_gpu${gpu}"
  local scratch
  scratch=$(mktemp -d "/tmp/t800-sdk-supine-${job_id}-${tag}.XXXXXX")
  local usd_dir="$scratch/usd" smoke_marker="$manifest_dir/preflight/${tag}.json"
  export TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" XDG_DATA_HOME="$scratch/data"
  export CUDA_CACHE_PATH="$scratch/cuda" MPLCONFIGDIR="$scratch/matplotlib" OMNI_KIT_CACHE_PATH="$scratch/kit_cache" WANDB_DIR="$scratch/wandb"
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$OMNI_KIT_CACHE_PATH" "$WANDB_DIR" "$usd_dir" "$(dirname "$smoke_marker")"
  cd "$repo"
  export WANDB_RUN_ID="$run_name" WANDB_RESUME=never
  echo "start variant=$variant method=ppo gpu=$gpu run_name=$run_name cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader

  "$python_bin" scripts/diagnostics/smoke_sdk_supine_shaping.py \
    --variant "$variant" --output "$smoke_marker" --usd-dir "$usd_dir" \
    --headless --device cuda:0 \
    --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
  test -s "$smoke_marker"
  "$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("passed") is True' "$smoke_marker"
  echo "preflight_complete variant=$variant method=ppo marker=$smoke_marker"

  local run_output
  "$python_bin" scripts/rsl_rl/train.py \
    --task "Tracking-Flat-T800-GetUp-SDK-Supine-${variant}-PPO-v0" \
    --num_envs 4096 --max_iterations 30000 --seed 42 --run_name "$run_name" \
    --logger wandb --log_project_name urkl-t800 --headless --device cuda:0 \
    --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
    agent.resume=false env.scene.robot.spawn.usd_dir="$usd_dir" env.scene.robot.spawn.force_usd_conversion=true \
    env.scene.terrain.visual_material=null env.commands.motion.debug_vis=false env.scene.contact_forces.debug_vis=false
  run_output=$(find "$repo/logs/rsl_rl/t800_flat" -type d -name "*_${run_name}" -print -quit)
  test -n "$run_output" && test -d "$run_output" && test -s "$run_output/model_29999.pt"
  echo "completed variant=$variant method=ppo output=$run_output checkpoint=$run_output/model_29999.pt"
}

if [[ "${2:-}" == worker ]]; then
  worker "$3" "$4" "$5"
  exit 0
fi

write_manifest
pids=()
indices=("${!variants[@]}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then indices=("$SLURM_ARRAY_TASK_ID"); fi
for gpu in "${indices[@]}"; do
  variant="${variants[$gpu]}"
  run_name="${stamp}_${variant}_ppo_gpu${gpu}_j${job_id}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G \
    --gres="gpu:${gpu_type}:1" --gpu-bind=single:1 --kill-on-bad-exit=0 \
    --output="$repo/slurm_logs/t800_sdk_supine_shaping-${job_id}-${variant}-gpu${gpu}.out" \
    --error="$repo/slurm_logs/t800_sdk_supine_shaping-${job_id}-${variant}-gpu${gpu}.err" \
    bash "$repo/scripts/slurm/run_t800_sdk_supine_shaping.sh" "$family" worker "$variant" "$gpu" "$run_name" &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if (( rc != 0 )); then
  "$python_bin" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["status"]="failed"; open(p,"w").write(json.dumps(d,indent=2)+"\n")' "$manifest"
  exit "$rc"
fi
"$python_bin" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["status"]="completed"; open(p,"w").write(json.dumps(d,indent=2)+"\n")' "$manifest"
test -s "$manifest"
echo "completed job=$job_id gpu_family=$family manifest=$manifest"
