#!/usr/bin/env bash
# V3/V4: independent direct-PPO runs; each variant is replicated twice.
# Four independent one-GPU array elements can fill slots on different nodes.
#SBATCH --job-name=t800_sdk_supine_shape_3090
#SBATCH --account=users
#SBATCH --partition=rtx3090
#SBATCH --qos=csc101_gpu_unlimited
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail
exec bash /srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/scripts/slurm/run_t800_sdk_supine_shaping.sh 3090 "$@"
