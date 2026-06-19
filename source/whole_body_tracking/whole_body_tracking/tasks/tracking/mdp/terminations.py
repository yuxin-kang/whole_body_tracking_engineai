from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


class TolerantTermination(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.bad_tracking_episode_length = None
        self.last_triggered_terms = {}
        self.final_trigger_mask = None
        self.env = env

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        bad_tracking_time_threshold_s: float,
        command_name: str,
        terms: list,
        recovering_shoulder_threshold: float = 1.0,
    ) -> torch.Tensor:
        self.env = env
        if self.bad_tracking_episode_length is None:
            self.bad_tracking_episode_length = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        current_combined_bad = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        command: MotionCommand = env.command_manager.get_term(command_name)
        for term in terms:
            term_name, func, params = _resolve_tolerant_term(term)
            raw_bad = func(env, **params)
            current_combined_bad |= raw_bad
            self.last_triggered_terms[term_name] = raw_bad

        # Paper Eq.21: gate the bad-tracking tolerance on the per-frame recovering state
        # (I_recovering = shoulder-height deviation from the reference exceeds a threshold),
        # NOT on the per-episode Bernoulli init flag command.is_standing_task. This makes a
        # tracking-init episode that is pushed down also get recovery tolerance (fall-resilient),
        # and a recovery-init episode that has stood up lose the tolerance (strict tracking).
        shoulder_indexes = getattr(command, "shoulders_indexes", [])
        if shoulder_indexes:
            height_diff = torch.norm(
                command.body_pos_relative_w[:, shoulder_indexes, 2]
                - command.robot_body_pos_w[:, shoulder_indexes, 2],
                dim=-1,
            )
            is_recovering = height_diff > recovering_shoulder_threshold
        else:
            is_recovering = command.is_standing_task

        max_bad_steps = math.ceil(bad_tracking_time_threshold_s / env.step_dt)
        self.bad_tracking_episode_length = torch.where(
            current_combined_bad,
            self.bad_tracking_episode_length + 1,
            torch.zeros_like(self.bad_tracking_episode_length),
        )
        self.final_trigger_mask = torch.where(
            is_recovering,
            self.bad_tracking_episode_length >= max_bad_steps,
            current_combined_bad,
        )
        return self.final_trigger_mask

    def reset(self, env_ids=None):
        if self.bad_tracking_episode_length is None:
            return

        if env_ids is None:
            mask = self.final_trigger_mask
            self.bad_tracking_episode_length.zero_()
        else:
            mask = self.final_trigger_mask[env_ids] if self.final_trigger_mask is not None else None
            self.bad_tracking_episode_length[env_ids] = 0

        if self.env is None or mask is None:
            return
        self.env.extras.setdefault("log", {})
        for name, value in self.last_triggered_terms.items():
            term_value = value if env_ids is None else value[env_ids]
            count = torch.count_nonzero(term_value & mask).item()
            key = f"Episode_Termination/TolerantTermination/{name}"
            self.env.extras[key] = count
            self.env.extras["log"][key] = count


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1)
    command.update_termination_debug_anchor_pos(error)
    return error > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    command.update_termination_debug_anchor_pos(error)
    return error > threshold


def _resolve_tolerant_term(term):
    if isinstance(term, dict):
        func = term["func"]
        if isinstance(func, str):
            func = globals()[func]
        return term["name"], func, term["params"]
    return term


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg | None = None,
    asset_name: str = "robot",
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name if asset_cfg is not None else asset_name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()
    command.update_termination_debug_anchor_ori(error)
    return error > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    command.update_termination_debug_body_pos(error, body_indexes)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    command.update_termination_debug_body_pos(error, body_indexes)
    return torch.any(error > threshold, dim=-1)


def bad_hip_dof(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    hip_indexes = [0, 1, 2, 6, 7, 8]
    valid_indexes = [idx for idx in hip_indexes if idx < command.joint_pos.shape[1]]
    if not valid_indexes:
        return torch.zeros(command.num_envs, device=command.device, dtype=torch.bool)
    error = torch.abs(command.joint_pos[:, valid_indexes] - command.robot_joint_pos[:, valid_indexes])
    return torch.any(error > threshold, dim=-1)
