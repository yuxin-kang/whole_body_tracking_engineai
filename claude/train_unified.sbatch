#!/bin/bash
#SBATCH --job-name=paper-unified-3090
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/claude/dbg_logs/paper-unified-3090.%j.out
#SBATCH --error=/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/claude/dbg_logs/paper-unified-3090.%j.out

# Detached single-stage paper-aligned FastSAC training (RTX 3090 fallback; 4090 partition down).
cd /srv/shared/home/kyx/Workspace/whole_body_tracking_engineai
exec bash claude/run_dbg.sh paper-unified-3090 \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Unified-v0 --motion_file data/g1/1307.npz \
  --num_envs 4096 --max_steps 100000 --train_steps 100000 --warmup_steps 10 \
  --eval_episodes 32 \
  --curve_interval 200 --progress_interval 1000 --checkpoint_interval 10000 \
  --logger none
