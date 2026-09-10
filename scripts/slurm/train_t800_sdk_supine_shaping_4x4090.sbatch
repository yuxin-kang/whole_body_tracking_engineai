#!/usr/bin/env bash
# V1/V2: independent direct-PPO runs; each variant is replicated twice.
# The four RTX 4090D cards remain on one node; srun gives each run one card.
#SBATCH --job-name=t800_sdk_supine_shape_4090
#SBATCH --account=users
#SBATCH --qos=csc101_gpu_unlimited
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail
exec bash /srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/scripts/slurm/run_t800_sdk_supine_shaping.sh 4090 "$@"
