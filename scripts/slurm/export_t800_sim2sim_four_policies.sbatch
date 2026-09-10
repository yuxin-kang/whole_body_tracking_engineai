#!/usr/bin/env bash
# Export the four user-selected T800 policies to the native SDK package format.
# This file is intentionally runnable directly as well as through sbatch.

set -euo pipefail

repo=/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
sdk=/srv/shared/home/xjh/ws_kyx/engineai_robotics_native_sdk/assets/config/t800
isaac=/srv/shared/home/xjh/IsaacLab-2.1.0
python_bin=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking/bin/python
export_device=${EXPORT_DEVICE:-cuda:0}
scratch=$(mktemp -d /tmp/wbt_t800_sdk_export_XXXXXX)
trap 'rm -rf "$scratch"' EXIT

export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMNI_KIT_ACCEPT_EULA=YES
export CONDA_PREFIX=/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$isaac/source/isaaclab:$isaac/source/isaaclab_assets:$isaac/source/isaaclab_rl:$isaac/source/isaaclab_tasks:$isaac/source/isaaclab_mimic:$repo/source/whole_body_tracking"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME="$scratch/cache" XDG_DATA_HOME="$scratch/data"
export CUDA_CACHE_PATH="$scratch/cuda" MPLCONFIGDIR="$scratch/mpl" TMPDIR="$scratch/tmp"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"

cd "$repo"

export_one() {
    local variant="$1" task="$2" motion="$3" run="$4" checkpoint="$5" package="$6"
    local checkpoint_path="$repo/logs/rsl_rl/t800_flat/$run/$checkpoint"
    local output="$repo/logs/rsl_rl/t800_flat/$run/exported"
    local destination="$sdk/$package"
    local motion_name
    motion_name=$(basename "$motion")

    test -s "$checkpoint_path"
    test -s "$motion"
    echo "export_start variant=$variant task=$task run=$run checkpoint=$checkpoint device=$export_device"

    "$python_bin" scripts/rsl_rl/play.py \
        --task "$task" \
        --motion_file "$motion" \
        --load_run "$run" \
        --checkpoint "$checkpoint" \
        --num_envs 1 \
        --play_from_start true \
        --max_steps 1 \
        --headless \
        --device "$export_device"

    test -s "$output/policy.onnx"
    test -s "$output/policy.mnn"
    test -s "$output/deploy_config.yaml"
    install -d "$destination/policies" "$destination/trajectories"
    cp "$output/policy.onnx" "$destination/policies/policy.onnx"
    cp "$output/policy.mnn" "$destination/policies/policy.mnn"
    cp "$output/deploy_config.yaml" "$destination/deploy_config.yaml"
    cp "$motion" "$destination/trajectories/$motion_name"

    "$python_bin" - "$output/deploy_config.yaml" "$destination" "$package" "$variant" "$task" "$motion" "$run" "$checkpoint" "$checkpoint_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

deploy_path = Path(sys.argv[1])
destination = Path(sys.argv[2])
package = sys.argv[3]
variant = sys.argv[4]
task = sys.argv[5]
motion_source = sys.argv[6]
run = sys.argv[7]
checkpoint = sys.argv[8]
checkpoint_path = Path(sys.argv[9])
motion_name = Path(motion_source).name

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
    f'trajectory_file_npz: "{package}/trajectories/{motion_name}"\n'
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
trajectory = destination / 'trajectories' / motion_name
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
    'motion_frames': 379 if 'getup' in motion_name else 179,
    'motion_fps': 50,
    'target_wall_training': 'MixedWall' in task,
    'export_method': 'scripts/rsl_rl/play.py --max_steps 1 --headless',
    'sim2sim_rollout_tested': False,
    'notes': 'Exported from the exact named IsaacLab checkpoint; SDK entry bridge retained for safe policy handoff.',
}
(destination / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

for path in (policy_onnx, policy_mnn, trajectory, destination / 'default.yaml', destination / 'deploy_config.yaml', destination / 'export_manifest.json'):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'Missing or empty SDK artifact: {path}')
print(json.dumps(manifest, indent=2))
PY

    echo "package_complete variant=$variant destination=$destination"
    sha256sum "$checkpoint_path" "$destination/policies/policy.onnx" "$destination/policies/policy.mnn" "$destination/trajectories/$motion_name"
}

export_one \
    getup_recovery_ori_velgate_finetune_4090_j9740_m34999 \
    Tracking-Flat-T800-GetUp-Recovery-OriVelGate-v0 \
    data/npz/t800_get_up/faint_prone_getup_03_t800.npz \
    2026-09-06_14-56-36_t800_getup_recovery_ori_velgate_finetune_4090_j9740 \
    model_34999.pt \
    rl_getup_recovery_ori_velgate_finetune_4090_j9740_m34999

export_one \
    getup_aggressive_c_forearm_f1_finetune_3090_j9768_m23500 \
    Tracking-Flat-T800-GetUp-Aggressive-C-Forearm-F1-v0 \
    data/npz/t800_get_up/faint_prone_getup_03_t800.npz \
    2026-09-07_13-27-29_t800_getup_forearm_f1_finetune_3090_j9768 \
    model_23500.pt \
    rl_getup_aggressive_c_forearm_f1_finetune_3090_j9768_m23500

export_one \
    roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999 \
    Tracking-Flat-T800-Target-MixedWall-RoundhouseKick-v0 \
    data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz \
    2026-09-07_17-32-14_t800_roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770 \
    model_29999.pt \
    rl_roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999

export_one \
    left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999 \
    Tracking-Flat-T800-Target-MixedWall-LeftFrontKick-v0 \
    data/npz/traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz \
    2026-09-07_17-32-14_t800_left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770 \
    model_29999.pt \
    rl_left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999

echo 'all_four_exports_complete'
