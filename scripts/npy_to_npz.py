#!/usr/bin/env python3
"""Convert EngineAI T800 npy motion to the standard training npz format.

NPY format:
  32 columns:
    0-2:   base_pos (x, y, z)
    3-6:   base_quat (x, y, z, w)
    7-31:  joint_pos (25 joints, DFS order)
  40 columns:
    0-2:   base_pos (x, y, z)
    3-6:   base_quat (w, x, y, z)
    7-31:  joint_pos (25 joints, DFS order)
    32-35: contacts (left_toe, left_heel, right_toe, right_heel)

This script replays the motion through IsaacLab's T800 robot and records the
canonical motion tensors used by training:
  fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay EngineAI npy motion and output a standard npz file.")
parser.add_argument(
    "--input_file",
    "--input",
    "-i",
    dest="input_file",
    type=str,
    required=True,
    help="The path to the input motion .npy file.",
)
parser.add_argument(
    "--output_file",
    "--output",
    "-o",
    dest="output_file",
    type=str,
    required=True,
    help="The path to the output motion npz file.",
)
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument("--output_fps", "--fps", dest="output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "Frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument(
    "--engineai_lab",
    type=str,
    default=None,
    help="Deprecated and ignored. The conversion no longer depends on the external engineaimuaythailab repository.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

assert Path(args_cli.input_file).resolve() != Path(args_cli.output_file).resolve(), "Input and output must differ."

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.t800 import T800_CFG
from whole_body_tracking.tasks.tracking.config.t800.t800_mdp import T800_DFS_JOINT_NAMES, T800_MOTION_BODY_NAMES

MISSING_MOTION_BODY_FALLBACKS = {
    "LINK_ANKLE_ROLL_L_TOE": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_HEEL": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R_TOE": "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_HEEL": "LINK_ANKLE_ROLL_R",
    "LINK_WRIST_PITCH_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_ROLL_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_R": "LINK_ELBOW_YAW_R",
    "LINK_WRIST_ROLL_R": "LINK_ELBOW_YAW_R",
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_npy_motion(self):
        motion = np.load(self.motion_file)
        if motion.ndim != 2 or motion.shape[1] not in (32, 40):
            raise ValueError(
                f"Expected an [T, 32] or [T, 40] motion array with base/joint columns, got shape {motion.shape}."
            )

        start = None if self.frame_range is None else self.frame_range[0] - 1
        end = None if self.frame_range is None else self.frame_range[1]
        return motion[start:end]

    def _load_motion(self):
        """Loads the motion from the npy file."""
        if not self.motion_file.lower().endswith(".npy"):
            raise ValueError(f"Expected a .npy motion file, got: {self.motion_file}")

        motion = torch.from_numpy(self._load_npy_motion()).to(dtype=torch.float32).to(device=self.device)

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        if motion.shape[1] == 32:
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_base_rots_input = self.motion_base_rots_input / torch.clamp(
            torch.norm(self.motion_base_rots_input, dim=-1, keepdim=True), min=1e-8
        )
        self.motion_dof_poss_input = motion[:, 7 : 7 + 25]
        self.motion_contacts = motion[:, 32:36] if motion.shape[1] >= 36 else None

        if self.motion_dof_poss_input.shape[1] != len(T800_DFS_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(T800_DFS_JOINT_NAMES)} joint positions, got {self.motion_dof_poss_input.shape[1]}."
            )

        self.input_frames = motion.shape[0]
        if self.input_frames < 2:
            raise ValueError("Motion must contain at least 2 frames.")
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(T800_DFS_JOINT_NAMES, preserve_order=True)[0]
    body_name_to_index = {name: idx for idx, name in enumerate(robot.body_names)}
    motion_body_indexes = []
    for body_name in T800_MOTION_BODY_NAMES:
        resolved_name = body_name if body_name in body_name_to_index else MISSING_MOTION_BODY_FALLBACKS.get(body_name)
        if resolved_name is None or resolved_name not in body_name_to_index:
            raise ValueError(
                f"Failed to resolve motion body '{body_name}'. Available robot bodies: {robot.body_names}"
            )
        motion_body_indexes.append(body_name_to_index[resolved_name])
    motion_body_indexes = torch.tensor(motion_body_indexes, dtype=torch.long, device=sim.device)

    log = {
        "fps": np.array([args_cli.output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        if not file_saved:
            # Persist joint data in the DFS motion order expected by replay/training,
            # not the articulation's internal storage order.
            log["joint_pos"].append(robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, robot_joint_indexes].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, motion_body_indexes].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, motion_body_indexes].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, motion_body_indexes].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, motion_body_indexes].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[key] = np.stack(log[key], axis=0)

            output_path = Path(args_cli.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **log)
            print(f"[INFO] Motion saved to {output_path}")
            return


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
