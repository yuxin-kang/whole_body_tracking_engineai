#!/usr/bin/env bash
# Export the user-selected aggressive get-up scratch checkpoints to the native SDK.
# Update the existing SDK packages to the latest checkpoints from job 9746.
#SBATCH --job-name=t800_getup_cd_export
#SBATCH --partition=rtx4000
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.out
#SBATCH --error=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai/slurm_logs/%x-%j.err

set -euo pipefail

repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
sdk=/srv/shared/home/xjh/ws_kyx/engineai_robotics_native_sdk/assets/config/t800
isaac=/srv/shared/home/xjh/ws_kyx/IsaacLab-2.1.0
python_bin=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking/bin/python
scratch=$(mktemp -d /tmp/wbt_getup_cd_export_${SLURM_JOB_ID}_XXXXXX)
trap 'rm -rf "$scratch"' EXIT

export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export http_proxy=http://10.12.120.125:7897 https_proxy=http://10.12.120.125:7897
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export CONDA_PREFIX=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$isaac/source/isaaclab:$isaac/source/isaaclab_assets:$isaac/source/isaaclab_rl:$isaac/source/isaaclab_tasks:$isaac/source/isaaclab_mimic:$repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME="$scratch/cache" XDG_DATA_HOME="$scratch/data"
export CUDA_CACHE_PATH="$scratch/cuda" MPLCONFIGDIR="$scratch/mpl" TMPDIR="$scratch/tmp"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"

cd "$repo"
motion=data/npz/t800_get_up/faint_prone_getup_03_t800.npz

export_one() {
    local variant="$1" task="$2" run="$3" checkpoint="$4" package="$5"
    local checkpoint_path="$repo/logs/rsl_rl/t800_flat/$run/$checkpoint"
    local output="$repo/logs/rsl_rl/t800_flat/$run/exported"
    local destination="$sdk/$package"

    test -s "$checkpoint_path"
    echo "export_start variant=$variant task=$task run=$run checkpoint=$checkpoint"
    "$python_bin" scripts/rsl_rl/play.py \
        --task "$task" \
        --motion_file "$motion" \
        --load_run "$run" \
        --checkpoint "$checkpoint" \
        --num_envs 1 \
        --play_from_start true \
        --max_steps 1 \
        --headless \
        --device cuda:0

    test -s "$output/policy.onnx"
    test -s "$output/policy.mnn"
    test -s "$output/deploy_config.yaml"
    install -d "$destination/policies" "$destination/trajectories"
    cp "$output/policy.onnx" "$destination/policies/policy.onnx"
    cp "$output/policy.mnn" "$destination/policies/policy.mnn"
    cp "$output/deploy_config.yaml" "$destination/deploy_config.yaml"
    cp "$motion" "$destination/trajectories/faint_prone_getup_03_t800.npz"

    "$python_bin" - "$output/deploy_config.yaml" "$destination" "$package" "$variant" "$task" "$run" "$checkpoint" "$checkpoint_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

deploy_path = Path(sys.argv[1])
destination = Path(sys.argv[2])
package = sys.argv[3]
variant = sys.argv[4]
task = sys.argv[5]
run = sys.argv[6]
checkpoint = sys.argv[7]
checkpoint_path = Path(sys.argv[8])

deploy_config = deploy_path.read_text()
required = (
    'observation_names: ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos",',
    'observation_history_lengths: [1, 1, 1, 1, 1, 1]',
    'action_scale:',
)
if not all(item in deploy_config for item in required):
    raise RuntimeError(f"Unexpected deployment ABI in {deploy_path}")

default_config = (
    f'policy_file: "{package}/policies/policy.mnn"\n'
    f'trajectory_file_npz: "{package}/trajectories/faint_prone_getup_03_t800.npz"\n'
    'trajectory_body_index: 0\n'
    + deploy_config
    + '\nresident_control: true\n'
    'entry_transition_enabled: true\n'
    'entry_transition_duration: 0.16\n'
    'entry_transition_min_duration: 0.10\n'
    'entry_transition_max_duration: 0.28\n'
    'entry_transition_max_joint_velocity: 8.0\n'
    'entry_transition_max_joint_acceleration: 120.0\n'
    'entry_transition_reference_pose_weight: 0.35\n'
    'entry_transition_source_tracking_error: 0.75\n'
    'entry_transition_to_reference_pose: true\n'
    'entry_transition_reference_pose_tolerance: 0.80\n'
    'entry_transition_reference_pose_settle_cycles: 5\n'
    'entry_transition_reference_pose_arm_stiffness_scale: 2.5\n'
    'trajectory_end_behavior: "exit"\n'
)
(destination / 'default.yaml').write_text(default_config)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

policy_onnx = destination / 'policies' / 'policy.onnx'
policy_mnn = destination / 'policies' / 'policy.mnn'
trajectory = destination / 'trajectories' / 'faint_prone_getup_03_t800.npz'
manifest = {
    'variant': variant,
    'task': task,
    'training_run': run,
    'checkpoint': checkpoint,
    'checkpoint_sha256': sha256(checkpoint_path),
    'source_deploy_config_sha256': sha256(deploy_path),
    'onnx_sha256': sha256(policy_onnx),
    'mnn_sha256': sha256(policy_mnn),
    'motion_sha256': sha256(trajectory),
    'observation_size': 134,
    'action_size': 25,
    'motion_frames': 379,
    'motion_fps': 50,
    'export_method': 'scripts/rsl_rl/play.py --max_steps 1 --headless',
    'sdk_sim_keys': ['BACK', 'LB'],
    'sim2sim_rollout_tested': False,
    'notes': 'Policy is exported from the exact named scratch checkpoint. The common get-up entry bridge is retained from the validated recovery SDK packages.',
}
(destination / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

for path in (policy_onnx, policy_mnn, trajectory, destination / 'default.yaml', destination / 'deploy_config.yaml', destination / 'export_manifest.json'):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'Missing or empty SDK artifact: {path}')
print(json.dumps(manifest, indent=2))
PY

    echo "package_complete variant=$variant destination=$destination"
    sha256sum "$checkpoint_path" "$destination/policies/policy.onnx" "$destination/policies/policy.mnn" "$destination/trajectories/faint_prone_getup_03_t800.npz"
}

export_one \
    C \
    Tracking-Flat-T800-GetUp-Aggressive-C-v0 \
    2026-09-06_16-08-31_t800_getup_aggressive_c_scratch_3090_j9746 \
    model_18500.pt \
    rl_getup_aggressive_c_scratch_3090_j9746_m15500

export_one \
    D \
    Tracking-Flat-T800-GetUp-Aggressive-D-v0 \
    2026-09-06_16-08-31_t800_getup_aggressive_d_scratch_3090_j9746 \
    model_17500.pt \
    rl_getup_aggressive_d_scratch_3090_j9746_m14500
