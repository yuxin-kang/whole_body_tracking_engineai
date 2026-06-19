#!/bin/bash
set -euo pipefail

REPO_DIR="/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai"
JOB_SCRIPT="$REPO_DIR/scripts/slurm/train_540_no_early_terminations_3090.sbatch"

mkdir -p "$REPO_DIR/slurm_logs"
sbatch "$JOB_SCRIPT"
