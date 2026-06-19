from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.debug_utils import collect_ee_body_violations
from whole_body_tracking.tasks.tracking.mdp.lke import (
    assign_lke_failure_to_previous_anchor,
    compute_lke_anchor_probabilities,
    compute_reference_joint_velocity_energy,
    detect_lke_anchor_indexes,
    initialize_lke_anchor_weights,
    update_lke_anchor_weights,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)


def _lerp_tensor(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    view_shape = [blend.shape[0]] + [1] * a.dim()
    alpha = blend.view(*view_shape)
    return a.unsqueeze(0) * (1.0 - alpha) + b.unsqueeze(0) * alpha


def _quat_slerp_batch(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    dot = torch.sum(q0 * q1, dim=-1)
    q1 = torch.where(dot.unsqueeze(-1) < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1).clamp(-1.0, 1.0)

    linear_mask = torch.abs(dot) > 0.9995
    lerp = _normalize_quat(q0 * (1.0 - blend.unsqueeze(-1)) + q1 * blend.unsqueeze(-1))

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * blend
    sin_theta = torch.sin(theta)

    s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1e-8)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
    slerp = _normalize_quat(s0.unsqueeze(-1) * q0 + s1.unsqueeze(-1) * q1)
    return torch.where(linear_mask.unsqueeze(-1), lerp, slerp)


def _bridge_quat_sequence(last_quat: torch.Tensor, first_quat: torch.Tensor, bridge_frames: int) -> torch.Tensor:
    if bridge_frames <= 0:
        return torch.empty((0,) + last_quat.shape, dtype=last_quat.dtype, device=last_quat.device)
    flat_last = last_quat.reshape(-1, 4)
    flat_first = first_quat.reshape(-1, 4)
    blend = torch.linspace(0.0, 1.0, bridge_frames + 2, device=last_quat.device, dtype=last_quat.dtype)[1:-1]
    bridge = []
    for tau in blend:
        tau_vec = torch.full((flat_last.shape[0],), tau, dtype=last_quat.dtype, device=last_quat.device)
        bridge.append(_quat_slerp_batch(flat_last, flat_first, tau_vec).reshape(last_quat.shape))
    return torch.stack(bridge, dim=0)


def _select_standing_init_pool_indexes(root_states_xyzw: torch.Tensor, dof_pos: torch.Tensor, num_select: int) -> torch.Tensor:
    from whole_body_tracking.tasks.tracking.config.g1.grsi import select_diverse_grsi_pool_indexes

    count = root_states_xyzw.shape[0]
    num_select = min(int(num_select), count)
    if num_select <= 0:
        return torch.empty(0, dtype=torch.long, device=root_states_xyzw.device)
    return select_diverse_grsi_pool_indexes(root_states_xyzw, dof_pos, num_select)


def _validate_grsi_init_data_if_present(
    init_robot_data: dict,
    init_pos_file: str,
    expected_joint_names: Sequence[str] | None,
) -> None:
    if "version" not in init_robot_data or "generation_config" not in init_robot_data:
        return

    from whole_body_tracking.tasks.tracking.config.g1.grsi import (
        validate_grsi_production_contract,
        validate_grsi_state_dict,
    )

    try:
        validate_grsi_state_dict(
            init_robot_data,
            expected_joint_names=list(expected_joint_names) if expected_joint_names is not None else None,
        )
        if Path(init_pos_file).name == "grsi_states.pth":
            validate_grsi_production_contract(init_robot_data, init_pos_file)
    except ValueError as exc:
        raise ValueError(f"Invalid GRSI standing init state file {init_pos_file}: {exc}") from exc


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        min_traj_duration: float | None = None,
        bridge_frames: int = 20,
        compute_kinetic_energy: bool = False,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        fps = data["fps"]
        self.fps = float(np.asarray(fps).reshape(-1)[0])
        motion_tensors = {
            "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
            "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
            "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
            "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
            "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
            "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
        }
        if min_traj_duration is not None and min_traj_duration > 0.0:
            min_frames = int(min_traj_duration * self.fps) + 1
            motion_tensors = self._extend_short_trajectory(
                motion_tensors,
                min_frames=min_frames,
                bridge_frames=bridge_frames,
                source_file=motion_file,
            )
        self.joint_pos = motion_tensors["joint_pos"]
        self.joint_vel = motion_tensors["joint_vel"]
        self._body_pos_w = motion_tensors["body_pos_w"]
        self._body_quat_w = motion_tensors["body_quat_w"]
        self._body_lin_vel_w = motion_tensors["body_lin_vel_w"]
        self._body_ang_vel_w = motion_tensors["body_ang_vel_w"]
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        if compute_kinetic_energy:
            kinetic_energy = compute_reference_joint_velocity_energy(self.joint_vel)
            self.kinetic_energy = kinetic_energy
            self.lke_anchor_indexes = detect_lke_anchor_indexes(kinetic_energy)
            self.lke_anchor_weights = initialize_lke_anchor_weights(
                self.lke_anchor_indexes,
                initial_weight=1.0,
            )
            self.kinetic_energy_prob = compute_lke_anchor_probabilities(self.lke_anchor_weights)

    def _extend_short_trajectory(
        self,
        motion: dict[str, torch.Tensor],
        min_frames: int,
        bridge_frames: int,
        source_file: str,
    ) -> dict[str, torch.Tensor]:
        frame_count = motion["joint_pos"].shape[0]
        if frame_count >= min_frames:
            return motion

        first = {k: v[0] for k, v in motion.items()}
        last = {k: v[-1] for k, v in motion.items()}
        blend = torch.linspace(
            0.0, 1.0, bridge_frames + 2, device=motion["joint_pos"].device, dtype=motion["joint_pos"].dtype
        )[1:-1]

        bridge = {
            "joint_pos": _lerp_tensor(last["joint_pos"], first["joint_pos"], blend),
            "joint_vel": _lerp_tensor(last["joint_vel"], first["joint_vel"], blend),
            "body_pos_w": _lerp_tensor(last["body_pos_w"], first["body_pos_w"], blend),
            "body_quat_w": _bridge_quat_sequence(last["body_quat_w"], first["body_quat_w"], bridge_frames),
            "body_lin_vel_w": _lerp_tensor(last["body_lin_vel_w"], first["body_lin_vel_w"], blend),
            "body_ang_vel_w": _lerp_tensor(last["body_ang_vel_w"], first["body_ang_vel_w"], blend),
        }

        pieces = [{k: v for k, v in motion.items()}]
        total_frames = frame_count
        while total_frames < min_frames:
            if bridge_frames > 0:
                pieces.append(bridge)
                total_frames += bridge_frames
            pieces.append(motion)
            total_frames += frame_count

        result = {key: torch.cat([piece[key] for piece in pieces], dim=0) for key in motion.keys()}
        orig_duration = max(frame_count - 1, 0) / self.fps
        new_duration = max(result["joint_pos"].shape[0] - 1, 0) / self.fps
        print(
            f"[INFO] Extended short motion '{os.path.basename(source_file)}': "
            f"{frame_count} -> {result['joint_pos'].shape[0]} frames "
            f"({orig_duration:.2f}s -> {new_duration:.2f}s, bridge={bridge_frames})"
        )
        return result

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.root_body_name = self.cfg.root_body_name or self.cfg.anchor_body_name
        self.robot_root_body_index = self.robot.body_names.index(self.root_body_name)
        motion_body_names = self.cfg.motion_body_names if self.cfg.motion_body_names is not None else self.cfg.body_names
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.motion_root_body_index = self.cfg.body_names.index(self.root_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion_body_indexes = torch.tensor(
            [motion_body_names.index(name) for name in self.cfg.body_names], dtype=torch.long, device=self.device
        )

        if self.cfg.motion_joint_names is not None:
            self.robot_joint_indexes = torch.tensor(
                self.robot.find_joints(self.cfg.motion_joint_names, preserve_order=True)[0],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.robot_joint_indexes = None

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.motion_body_indexes,
            device=self.device,
            min_traj_duration=self.cfg.min_traj_duration,
            bridge_frames=self.cfg.bridge_frames,
            compute_kinetic_energy=self.cfg.sampling_mode == "lke",
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_standing_task = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        self._termination_debug_anchor_pos_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._termination_debug_anchor_ori_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._termination_debug_body_pos_error = torch.zeros(
            self.num_envs, len(cfg.body_names), dtype=torch.float32, device=self.device
        )
        self._play_from_start = bool(self.cfg.play_from_start)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_root_body_index] + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_root_body_index]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_root_body_index]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_root_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        if self.robot_joint_indexes is None:
            return self.robot.data.joint_pos
        return self.robot.data.joint_pos[:, self.robot_joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        if self.robot_joint_indexes is None:
            return self.robot.data.joint_vel
        return self.robot.data.joint_vel[:, self.robot_joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_root_body_index]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_root_body_index]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_root_body_index]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_root_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        if self.cfg.phase_sampling_windows:
            bin_centers = (torch.arange(self.bin_count, device=self.device, dtype=torch.float32) + 0.5) / float(self.bin_count)
            phase_weights = torch.ones_like(sampling_probabilities)
            for phase_start, phase_end, weight in self.cfg.phase_sampling_windows:
                if weight <= 0.0:
                    continue
                in_window = (bin_centers >= phase_start) & (bin_centers <= phase_end)
                phase_weights = torch.where(in_window, phase_weights * float(weight), phase_weights)
            sampling_probabilities = sampling_probabilities * phase_weights

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else torch.ones((), device=self.device)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_sampling(self, env_ids: torch.Tensor):
        self.time_steps[env_ids] = torch.randint(
            0,
            self.motion.time_step_total,
            (len(env_ids),),
            device=self.device,
        )
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.bin_count, 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _lke_sampling(self, env_ids: torch.Tensor):
        if not hasattr(self.motion, "lke_anchor_indexes"):
            raise RuntimeError("LKE sampling requires MotionLoader(compute_kinetic_energy=True).")
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_steps = self.time_steps[env_ids][episode_failed]
            self.motion.lke_anchor_weights = update_lke_anchor_weights(
                self.motion.lke_anchor_weights,
                failed_steps,
                self.motion.lke_anchor_indexes,
                alpha=0.2,
                w_min=1.0,
                w_max=8.0,
            )

        sampling_probabilities = compute_lke_anchor_probabilities(self.motion.lke_anchor_weights)
        sampled_anchor_positions = torch.multinomial(
            sampling_probabilities,
            env_ids.numel(),
            replacement=True,
        )
        self.time_steps[env_ids] = self.motion.lke_anchor_indexes[sampled_anchor_positions]

        h_norm = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        h_norm = h_norm / math.log(sampling_probabilities.numel()) if sampling_probabilities.numel() > 1 else h_norm
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = h_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = self.motion.lke_anchor_indexes[imax].float() / max(
            self.motion.time_step_total, 1
        )

    def _sample_time_steps(self, env_ids: torch.Tensor):
        if self._play_from_start or self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        elif self.cfg.sampling_mode == "lke":
            self._lke_sampling(env_ids)
        else:
            self._adaptive_sampling(env_ids)

    def update_termination_debug_anchor_pos(self, error: torch.Tensor):
        self._termination_debug_anchor_pos_error.copy_(error.detach())

    def update_termination_debug_anchor_ori(self, error: torch.Tensor):
        self._termination_debug_anchor_ori_error.copy_(error.detach())

    def update_termination_debug_body_pos(self, error: torch.Tensor, body_indexes: Sequence[int] | torch.Tensor):
        self._termination_debug_body_pos_error.zero_()
        if isinstance(body_indexes, torch.Tensor):
            body_index_tensor = body_indexes.to(device=self.device, dtype=torch.long)
        else:
            if len(body_indexes) == 0:
                return
            body_index_tensor = torch.tensor(body_indexes, dtype=torch.long, device=self.device)
        if body_index_tensor.numel() == 0:
            return
        self._termination_debug_body_pos_error[:, body_index_tensor] = error.detach()

    def set_play_from_start_mode(self):
        self._play_from_start = True
        self.time_steps.zero_()
        self.bin_failed_count.zero_()
        self._current_bin_failed.zero_()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_envs_from_motion(env_ids)
        self._refresh_relative_motion_state(env_ids)
        self._update_metrics()

    def get_termination_debug_info(self, env_ids: Sequence[int] | torch.Tensor) -> list[dict[str, object]]:
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() == 0:
            return []

        infos: list[dict[str, object]] = []
        terminations_cfg = self._env.cfg.terminations
        anchor_pos_cfg = getattr(terminations_cfg, "anchor_pos", None)
        anchor_ori_cfg = getattr(terminations_cfg, "anchor_ori", None)
        ee_body_pos_cfg = getattr(terminations_cfg, "ee_body_pos", None)

        ee_body_indexes: list[int] = []
        ee_body_names: list[str] = []
        ee_body_threshold: float | None = None
        if ee_body_pos_cfg is not None:
            ee_body_threshold = float(ee_body_pos_cfg.params["threshold"])
            ee_body_names_filter = ee_body_pos_cfg.params.get("body_names")
            ee_body_indexes = [
                i for i, name in enumerate(self.cfg.body_names) if (ee_body_names_filter is None) or (name in ee_body_names_filter)
            ]
            ee_body_names = [self.cfg.body_names[i] for i in ee_body_indexes]

        for env_id in env_ids_tensor.tolist():
            triggered_terms: list[dict[str, object]] = []
            if anchor_pos_cfg is not None:
                threshold = float(anchor_pos_cfg.params["threshold"])
                error = float(self._termination_debug_anchor_pos_error[env_id].item())
                if error > threshold:
                    triggered_terms.append({"term": "anchor_pos", "error": error, "threshold": threshold})

            if anchor_ori_cfg is not None:
                threshold = float(anchor_ori_cfg.params["threshold"])
                error = float(self._termination_debug_anchor_ori_error[env_id].item())
                if error > threshold:
                    triggered_terms.append({"term": "anchor_ori", "error": error, "threshold": threshold})

            if ee_body_threshold is not None and ee_body_indexes:
                error_z_values = self._termination_debug_body_pos_error[env_id, ee_body_indexes].detach().cpu().tolist()
                violations = [
                    violation
                    for violation in collect_ee_body_violations(ee_body_names, error_z_values, ee_body_threshold)
                    if violation["triggered"]
                ]
                if violations:
                    triggered_terms.append(
                        {
                            "term": "ee_body_pos",
                            "threshold": ee_body_threshold,
                            "violations": violations,
                        }
                    )

            infos.append({"env_id": env_id, "triggered_terms": triggered_terms})

        return infos

    def _refresh_relative_motion_state(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = env_ids
        if env_ids_tensor.numel() == 0:
            return

        anchor_pos_w_repeat = self.anchor_pos_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w[env_ids_tensor] = quat_mul(delta_ori_w, self.body_quat_w[env_ids_tensor])
        self.body_pos_relative_w[env_ids_tensor] = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w[env_ids_tensor] - anchor_pos_w_repeat
        )

    def _pd_stand_reset_would_terminate(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        terminated = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        terminations_cfg = self._env.cfg.terminations

        anchor_pos_cfg = getattr(terminations_cfg, "anchor_pos", None)
        if anchor_pos_cfg is not None:
            threshold = float(anchor_pos_cfg.params["threshold"])
            terminated |= torch.abs(self.anchor_pos_w[env_ids, -1] - self.robot_anchor_pos_w[env_ids, -1]) > threshold

        anchor_ori_cfg = getattr(terminations_cfg, "anchor_ori", None)
        if anchor_ori_cfg is not None:
            threshold = float(anchor_ori_cfg.params["threshold"])
            gravity_vec_w = self.robot.data.GRAVITY_VEC_W
            if gravity_vec_w.ndim > 1:
                gravity_vec_w = gravity_vec_w[env_ids]
            motion_projected_gravity_b = quat_apply_inverse(self.anchor_quat_w[env_ids], gravity_vec_w)
            robot_projected_gravity_b = quat_apply_inverse(self.robot_anchor_quat_w[env_ids], gravity_vec_w)
            terminated |= torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > threshold

        ee_body_pos_cfg = getattr(terminations_cfg, "ee_body_pos", None)
        if ee_body_pos_cfg is not None:
            threshold = float(ee_body_pos_cfg.params["threshold"])
            body_names = ee_body_pos_cfg.params.get("body_names")
            body_indexes = [
                i for i, name in enumerate(self.cfg.body_names) if (body_names is None) or (name in body_names)
            ]
            if body_indexes:
                body_indexes = torch.tensor(body_indexes, dtype=torch.long, device=self.device)
                error_z = torch.abs(
                    self.body_pos_relative_w[env_ids][:, body_indexes, -1] - self.robot_body_pos_w[env_ids][:, body_indexes, -1]
                )
                terminated |= torch.any(error_z > threshold, dim=-1)

        return terminated

    def _reset_envs_from_motion(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        root_pos = self.root_pos_w[env_ids].clone()
        root_ori = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        if self.robot_joint_indexes is None:
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            full_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.robot_joint_indexes]
            clipped_joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            full_joint_pos[:, self.robot_joint_indexes] = clipped_joint_pos
            full_joint_vel[:, self.robot_joint_indexes] = joint_vel
            self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _reset_envs_from_pd_stand(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return env_ids

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._env.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._env.scene.update(self._env.physics_dt)
        self._refresh_relative_motion_state(env_ids)

        return env_ids[self._pd_stand_reset_would_terminate(env_ids)]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self._play_from_start:
            self._sample_time_steps(env_ids)
            self._reset_envs_from_motion(env_ids)
            self._refresh_relative_motion_state(env_ids)
            return
        self._sample_time_steps(env_ids)
        if self.cfg.reset_preroll_frames > 0:
            self.time_steps[env_ids] = torch.clamp(
                self.time_steps[env_ids] - self.cfg.reset_preroll_frames,
                min=0,
            )

        pd_stand_ratio = float(self.cfg.pd_stand_reset_ratio)
        if pd_stand_ratio <= 0.0:
            self._reset_envs_from_motion(env_ids)
            return

        pd_stand_mask = torch.rand(len(env_ids), device=self.device) < pd_stand_ratio
        motion_env_ids = env_ids[~pd_stand_mask]
        pd_stand_env_ids = env_ids[pd_stand_mask]

        self._reset_envs_from_motion(motion_env_ids)
        fallback_env_ids = self._reset_envs_from_pd_stand(pd_stand_env_ids)
        if fallback_env_ids.numel() > 0:
            self._reset_envs_from_motion(fallback_env_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if len(env_ids) > 0:
            if self.cfg.resample_at_motion_end:
                self._resample_command(env_ids)
            else:
                self.time_steps[env_ids] = self.motion.time_step_total - 1
        self._refresh_relative_motion_state()

        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])


class MotionStandingCommand(MotionCommand):
    cfg: MotionStandingCommandCfg

    def __init__(self, cfg: MotionStandingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not os.path.exists(cfg.init_pos_file):
            raise FileNotFoundError(f"Can't find standing init state file: {cfg.init_pos_file}")

        self.init_robot_data = torch.load(cfg.init_pos_file, map_location="cpu", weights_only=True)
        expected_joint_names = (
            list(self.cfg.motion_joint_names) if self.cfg.motion_joint_names is not None else None
        )
        _validate_grsi_init_data_if_present(self.init_robot_data, cfg.init_pos_file, expected_joint_names)
        expected_joint_count = len(expected_joint_names) if expected_joint_names is not None else self.robot.num_joints
        dof_pos = self.init_robot_data["dof_pos"]
        if dof_pos.ndim != 2 or dof_pos.shape[1] != expected_joint_count:
            raise ValueError(
                f"Standing init dof_pos must have shape (N, {expected_joint_count}); got {tuple(dof_pos.shape)}."
            )
        dof_vel = self.init_robot_data.get("dof_vel")
        if dof_vel is not None and (dof_vel.ndim != 2 or dof_vel.shape[1] != expected_joint_count):
            raise ValueError(
                f"Standing init dof_vel must have shape (N, {expected_joint_count}); got {tuple(dof_vel.shape)}."
            )
        root_states_xyzw = self.init_robot_data["robot_root_states_xyzw"]
        if root_states_xyzw.ndim != 2 or root_states_xyzw.shape[1] < 13:
            raise ValueError(
                f"Standing init robot_root_states_xyzw must have shape (N, >=13); got {tuple(root_states_xyzw.shape)}."
            )
        pool_size = min(self.cfg.standing_init_sample_pool_size, root_states_xyzw.shape[0])
        self.standing_init_pool_ids = _select_standing_init_pool_indexes(root_states_xyzw, dof_pos, pool_size).cpu()

        self.is_standing_task = torch.multinomial(
            torch.tensor(cfg.tracking_standing_weight, device=self.device, dtype=torch.float32),
            num_samples=self.num_envs,
            replacement=True,
        ).bool()
        self.prev_anchor_pos = torch.zeros_like(self.robot_anchor_pos_w)
        self.current_anchor_pos = self.robot_anchor_pos_w.clone()
        self.root_indexes = [self.cfg.body_names.index(name) for name in self.cfg.root_body_names]
        self.root_index = self.root_indexes[0] if self.root_indexes else self.motion_root_body_index
        self.shoulders_indexes = [self.cfg.body_names.index(name) for name in self.cfg.shoulders_body_names]
        self.feet_indexes = [self.cfg.body_names.index(name) for name in self.cfg.feet_body_names]

    def _sample_standing_init(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.standing_init_pool_ids.numel() > 0:
            pool_indexes = torch.randint(0, self.standing_init_pool_ids.numel(), (count,))
            sampled_ids = self.standing_init_pool_ids[pool_indexes].long()
        else:
            sampled_ids = torch.randint(0, self.init_robot_data["robot_root_states_xyzw"].shape[0], (count,))

        root_states_xyzw = self.init_robot_data["robot_root_states_xyzw"][sampled_ids].to(self.device)
        dof_pos = self.init_robot_data["dof_pos"][sampled_ids].to(self.device)
        dof_vel = self.init_robot_data.get("dof_vel")
        if dof_vel is None:
            dof_vel = torch.zeros_like(dof_pos)
        else:
            dof_vel = dof_vel[sampled_ids].to(self.device)
        return root_states_xyzw, dof_pos, dof_vel

    def _write_standing_or_motion_state(self, env_ids: torch.Tensor, standing_mask: torch.Tensor):
        if env_ids.numel() == 0:
            return

        root_pos = self.root_pos_w[env_ids].clone()
        root_ori = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        standing_root_xyzw, standing_joint_pos, standing_joint_vel = self._sample_standing_init(len(env_ids))
        if standing_mask.any():
            root_pos = torch.where(
                standing_mask.unsqueeze(1),
                torch.cat([root_pos[:, :2], standing_root_xyzw[:, 2:3]], dim=1),
                root_pos,
            )
            root_ori = torch.where(
                standing_mask.unsqueeze(1),
                torch.cat([standing_root_xyzw[:, 6:7], standing_root_xyzw[:, 3:6]], dim=1),
                root_ori,
            )
            root_lin_vel = torch.where(standing_mask.unsqueeze(1), standing_root_xyzw[:, 7:10], root_lin_vel)
            root_ang_vel = torch.where(standing_mask.unsqueeze(1), standing_root_xyzw[:, 10:13], root_ang_vel)

            joint_pos = torch.where(standing_mask.unsqueeze(1), standing_joint_pos, joint_pos)
            joint_vel = torch.where(standing_mask.unsqueeze(1), standing_joint_vel, joint_vel)

        if self.robot_joint_indexes is None:
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            full_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.robot_joint_indexes]
            clipped_joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            full_joint_pos[:, self.robot_joint_indexes] = clipped_joint_pos
            full_joint_vel[:, self.robot_joint_indexes] = joint_vel
            self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._sample_time_steps(env_ids)
        if self.cfg.reset_preroll_frames > 0:
            self.time_steps[env_ids] = torch.clamp(self.time_steps[env_ids] - self.cfg.reset_preroll_frames, min=0)

        standing_mask = torch.multinomial(
            torch.tensor(self.cfg.tracking_standing_weight, device=self.device, dtype=torch.float32),
            num_samples=env_ids.numel(),
            replacement=True,
        ).bool()
        self.is_standing_task[env_ids] = standing_mask
        self._write_standing_or_motion_state(env_ids, standing_mask)
        self._refresh_relative_motion_state(env_ids)

    def _update_command(self):
        self.prev_anchor_pos[:] = self.current_anchor_pos
        super()._update_command()
        self.current_anchor_pos[:] = self.robot_anchor_pos_w


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    root_body_name: str | None = None
    body_names: list[str] = MISSING
    motion_body_names: list[str] | None = None
    motion_joint_names: list[str] | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    min_traj_duration: float | None = None
    bridge_frames: int = 20

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    sampling_mode: Literal["adaptive", "uniform", "start", "lke"] = "adaptive"
    phase_sampling_windows: list[tuple[float, float, float]] = []
    pd_stand_reset_ratio: float = 0.0
    reset_preroll_frames: int = 0
    play_from_start: bool = False
    resample_at_motion_end: bool = True

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class MotionStandingCommandCfg(MotionCommandCfg):
    """Motion command that can reset a subset of envs from sampled standing states."""

    class_type: type = MotionStandingCommand

    init_pos_file: str = MISSING
    root_body_names: list[str] = []
    shoulders_body_names: list[str] = []
    feet_body_names: list[str] = []
    tracking_standing_weight: tuple[float, float] = (1.0, 1.0)
    standing_init_sample_pool_size: int = 2048
