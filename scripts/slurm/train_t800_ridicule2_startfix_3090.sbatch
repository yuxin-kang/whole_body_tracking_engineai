#!/bin/bash
#SBATCH --job-name=t800_ridicule2_startfix
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err
set -euo pipefail
cd /srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
# Reuse baseline PPO/rewards; learn complete episodes starting at the repaired first pose.
exec bash scripts/slurm/train_t800_ridicule_baseline_3090.sbatch \
  data/npz/ridicule_2/ridicule2_t800_startfix_50hz.npz ridicule2_startfix \
  env.commands.motion.sampling_mode=start \
  env.commands.motion.resample_at_motion_end=false
