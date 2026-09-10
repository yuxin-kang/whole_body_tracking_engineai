"""Export T800 actor checkpoints when IsaacSim cannot start on the host.

The actor ABI is independent of the simulator: 134 observations are normalized
with the checkpoint's empirical statistics and passed through the same
512-256-128 ELU MLP to produce 25 actions.  This helper is only for packaging
the selected checkpoints; it does not replace IsaacLab rollout validation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn


REPO = Path("/srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai")
SDK_T800 = Path("/srv/shared/home/xjh/ws_kyx/engineai_robotics_native_sdk/assets/config/t800")
DEPLOY_CONFIG = SDK_T800 / "rl_getup_recovery_ori_velgate_finetune_4090_j9740_m34999/deploy_config.yaml"
NORMALIZER_EPS = 1.0e-2


POLICIES = (
    *(
        {
            "variant": f"getup_aggressive_{variant}_scratch_3090_j9746_m29999",
            "task": f"Tracking-Flat-T800-GetUp-Aggressive-{variant.upper()}-v0",
            "run": f"2026-09-06_16-08-31_t800_getup_aggressive_{variant}_scratch_3090_j9746",
            "checkpoint": "model_29999.pt",
            "motion": "data/npz/t800_get_up/faint_prone_getup_03_t800.npz",
            "package": f"rl_getup_aggressive_{variant}_scratch_3090_j9746_m29999",
            "sdk_sim_keys": [],
            "sdk_key_description": "sim2sim package; native SDK task key not assigned",
        }
        for variant in "abcd"
    ),
    {
        "variant": "getup_aggressive_c_forearm_f1_finetune_3090_j9768_m23500",
        "task": "Tracking-Flat-T800-GetUp-Aggressive-C-Forearm-F1-v0",
        "run": "2026-09-07_13-27-29_t800_getup_forearm_f1_finetune_3090_j9768",
        "checkpoint": "model_23500.pt",
        "motion": "data/npz/t800_get_up/faint_prone_getup_03_t800.npz",
        "package": "rl_getup_aggressive_c_forearm_f1_finetune_3090_j9768_m23500",
        "sdk_sim_keys": [],
        "sdk_key_description": "sim2sim package; native SDK task key not assigned",
    },
    *(
        {
            "variant": f"getup_aggressive_c_forearm_{variant}_finetune_3090_j9768_m23500",
            "task": f"Tracking-Flat-T800-GetUp-Aggressive-C-Forearm-{variant.upper()}-v0",
            "run": f"2026-09-07_13-27-29_t800_getup_forearm_{variant}_finetune_3090_j9768",
            "checkpoint": "model_23500.pt",
            "motion": "data/npz/t800_get_up/faint_prone_getup_03_t800.npz",
            "package": f"rl_getup_aggressive_c_forearm_{variant}_finetune_3090_j9768_m23500",
            "sdk_sim_keys": [],
            "sdk_key_description": "sim2sim package; native SDK task key not assigned",
        }
        for variant in ("f2", "f3", "f4")
    ),
    {
        "variant": "roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999",
        "task": "Tracking-Flat-T800-Target-MixedWall-RoundhouseKick-v0",
        "run": "2026-09-07_17-32-14_t800_roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770",
        "checkpoint": "model_29999.pt",
        "motion": "data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz",
        "package": "rl_roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999",
        "sdk_sim_keys": ["CROSS_X_UP", "CROSS_X_DOWN", "CROSS_Y_LEFT"],
        "sdk_key_description": "RT + 十字键左",
    },
    {
        "variant": "left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999",
        "task": "Tracking-Flat-T800-Target-MixedWall-LeftFrontKick-v0",
        "run": "2026-09-07_17-32-14_t800_left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770",
        "checkpoint": "model_29999.pt",
        "motion": "data/npz/traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz",
        "package": "rl_left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770_m29999",
        "sdk_sim_keys": ["CROSS_Y_LEFT", "CROSS_Y_RIGHT", "CROSS_X_UP"],
        "sdk_key_description": "LT + 十字键上",
    },
)


class NormalizedActor(nn.Module):
    def __init__(self, actor_state: dict[str, torch.Tensor], norm_state: dict[str, torch.Tensor]):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(134, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 25),
        )
        self.actor.load_state_dict({key.removeprefix("actor."): value for key, value in actor_state.items()})
        self.register_buffer("mean", norm_state["_mean"].detach().clone())
        self.register_buffer("std", norm_state["_std"].detach().clone())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.mean) / (self.std + NORMALIZER_EPS)
        return self.actor(normalized)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_to_csv(values: list[object]) -> str:
    result = []
    for value in values:
        if isinstance(value, (int, float)):
            result.append(f"{value:.3f}")
        else:
            result.append(str(value))
    return ",".join(result)


def export_policy(spec: dict[str, object]) -> None:
    checkpoint_path = REPO / "logs/rsl_rl/t800_flat" / str(spec["run"]) / str(spec["checkpoint"])
    motion_path = REPO / str(spec["motion"])
    destination = SDK_T800 / str(spec["package"])
    destination_policy = destination / "policies"
    destination_motion = destination / "trajectories"
    destination_policy.mkdir(parents=True, exist_ok=True)
    destination_motion.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = {key: value for key, value in checkpoint["model_state_dict"].items() if key.startswith("actor.")}
    actor = NormalizedActor(actor_state, checkpoint["obs_norm_state_dict"]).eval()
    onnx_path = destination_policy / "policy.onnx"
    torch.onnx.export(
        actor,
        torch.zeros(1, 134),
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    metadata = {
        "observation_names": ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions"],
        "observation_history_lengths": [1, 1, 1, 1, 1, 1],
        "action_scale": [0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.05, 0.2, 0.05, 0.2, 0.2, 0.05, 0.2, 0.05, 0.2, 0.2],
    }
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto(key=key, value=list_to_csv(value))
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)

    mnn_path = destination_policy / "policy.mnn"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "MNN.tools.mnnconvert",
            "-f",
            "ONNX",
            "--modelFile",
            str(onnx_path),
            "--MNNModel",
            str(mnn_path),
            "--bizCode",
            "MNN",
        ],
        check=True,
    )

    deploy_config = DEPLOY_CONFIG.read_text()
    shutil.copy2(DEPLOY_CONFIG, destination / "deploy_config.yaml")
    motion_destination = destination_motion / motion_path.name
    shutil.copy2(motion_path, motion_destination)
    motion_meta = np.load(motion_path, allow_pickle=False)
    motion_frames = int(motion_meta["joint_pos"].shape[0])
    motion_fps = float(np.asarray(motion_meta["fps"]).reshape(-1)[0])

    (destination / "default.yaml").write_text(
        f'policy_file: "{spec["package"]}/policies/policy.mnn"\n'
        f'trajectory_file_npz: "{spec["package"]}/trajectories/{motion_path.name}"\n'
        "trajectory_body_index: 0\n"
        + deploy_config
        + "\nresident_control: true\n"
        "entry_transition_enabled: true\n"
        "entry_transition_duration: 0.16\n"
        "entry_transition_min_duration: 0.10\n"
        "entry_transition_max_duration: 0.28\n"
        "entry_transition_max_joint_velocity: 8.0\n"
        "entry_transition_max_joint_acceleration: 120.0\n"
        "entry_transition_reference_pose_weight: 0.35\n"
        "entry_transition_source_tracking_error: 0.75\n"
        "entry_transition_to_reference_pose: true\n"
        "entry_transition_reference_pose_tolerance: 0.80\n"
        "entry_transition_reference_pose_settle_cycles: 5\n"
        "entry_transition_reference_pose_arm_stiffness_scale: 2.5\n"
        'trajectory_end_behavior: "exit"\n'
    )

    manifest = {
        "variant": spec["variant"],
        "task": spec["task"],
        "training_run": spec["run"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_deploy_config_sha256": sha256(DEPLOY_CONFIG),
        "onnx_sha256": sha256(onnx_path),
        "mnn_sha256": sha256(mnn_path),
        "motion_sha256": sha256(motion_destination),
        "observation_size": 134,
        "action_size": 25,
        "motion_frames": motion_frames,
        "motion_fps": motion_fps,
        "target_wall_training": "MixedWall" in str(spec["task"]),
        "export_method": "checkpoint actor + empirical normalizer; CPU export because IsaacSim GPU unavailable",
        "sdk_sim_keys": spec["sdk_sim_keys"],
        "sdk_key_description": spec["sdk_key_description"],
        "sim2sim_rollout_tested": False,
        "notes": "The actor graph and normalizer are reconstructed from the exact named checkpoint; IsaacLab rollout still needs GPU validation.",
    }
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    required = [
        destination / "default.yaml",
        destination / "deploy_config.yaml",
        destination / "export_manifest.json",
        onnx_path,
        mnn_path,
        motion_destination,
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError(f"Incomplete SDK package: {destination}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    getup_policies = tuple(spec for spec in POLICIES if "GetUp" in str(spec["task"]))
    if len(getup_policies) != 8:
        raise RuntimeError(f"Expected eight get-up policies, found {len(getup_policies)}")
    for policy_spec in getup_policies:
        export_policy(policy_spec)
    print("eight_getup_exports_complete")
