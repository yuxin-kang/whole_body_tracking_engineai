"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --input_file /path/to/motion.npz --robot t800
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
import time
import sys
from importlib import util
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--input_file", type=str, default=None, help="Path to a local .npz motion file.")
parser.add_argument("--robot", type=str, default="t800", choices=["pm01", "t800"], help="Robot type to use.")
parser.add_argument(
    "--report_540_landmarks_only",
    action="store_true",
    help="Print the T800 540 right-ankle landmark summary and exit.",
)
parser.add_argument(
    "--announce_540_landmarks",
    action="store_true",
    help="Print the T800 540 right-ankle keyframe hits while replaying.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def _load_motion_landmarks_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "utils"
        / "motion_landmarks.py"
    )
    spec = util.spec_from_file_location("motion_landmarks", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

if args_cli.report_540_landmarks_only:
    if args_cli.input_file is None:
        raise ValueError("--input_file is required when using --report_540_landmarks_only.")
    module = _load_motion_landmarks_module()
    print(module.summarize_t800_right_ankle_540_landmarks(args_cli.input_file).format())
    raise SystemExit(0)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.pm01 import PM01_CYLINDER_CFG
from whole_body_tracking.robots.t800 import T800_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

ROBOT_CFGS = {
    "pm01": PM01_CYLINDER_CFG,
    "t800": T800_CFG,
}

T800_MOTION_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
]

ROBOT_MOTION_JOINT_NAMES = {
    "t800": T800_MOTION_JOINT_NAMES,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (will be overridden in main based on --robot)
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _print_replay_timing_summary(motion_file: str, motion: MotionLoader, sim_dt: float):
    metadata_duration_s = max(motion.time_step_total - 1, 0) / max(motion.fps, 1e-8)
    logical_replay_fps = 1.0 / sim_dt
    logical_cycle_duration_s = motion.time_step_total * sim_dt

    print("[INFO] Replay timing summary")
    print(f"  motion_file: {motion_file}")
    print(f"  metadata_fps: {motion.fps:.4f}")
    print(f"  num_frames: {motion.time_step_total}")
    print(f"  metadata_duration_s: {metadata_duration_s:.4f}")
    print(f"  logical_replay_fps: {logical_replay_fps:.4f}")
    print(f"  logical_cycle_duration_s: {logical_cycle_duration_s:.4f}")
    print("  NOTE: filename labels like '12hz' are not trusted here; use the numbers above instead.")
    if abs(motion.fps - logical_replay_fps) > 1e-3:
        print(
            "  WARN: metadata_fps differs from logical_replay_fps, so file metadata and replay cadence do not match."
        )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.input_file is not None:
        motion_file = args_cli.input_file
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Either --input_file or --registry_name must be provided.")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    _print_replay_timing_summary(motion_file, motion, sim_dt)
    landmark_report = None
    landmark_hits = {}
    announced_landmarks = set()
    if args_cli.announce_540_landmarks:
        if args_cli.robot != "t800":
            raise ValueError("--announce_540_landmarks is only supported for --robot t800.")
        landmark_module = _load_motion_landmarks_module()
        landmark_report = landmark_module.summarize_t800_right_ankle_540_landmarks(motion_file)
        landmark_hits = {
            landmark_report.qishi_end: "qishi_end",
            landmark_report.pre_peak_local_min: "pre_peak_local_min",
            landmark_report.peak_frame: "peak_frame",
            landmark_report.post_peak_local_min: "post_peak_local_min",
            landmark_report.return_zero_end: "return_zero_end",
        }
        print(landmark_report.format())
    motion_joint_names = ROBOT_MOTION_JOINT_NAMES.get(args_cli.robot)
    robot_joint_indexes = None
    if motion_joint_names is not None:
        robot_joint_indexes = robot.find_joints(motion_joint_names, preserve_order=True)[0]
        if len(robot_joint_indexes) != motion.joint_pos.shape[1]:
            raise ValueError(
                f"Motion joint count ({motion.joint_pos.shape[1]}) does not match mapped robot joints "
                f"({len(robot_joint_indexes)}) for robot '{args_cli.robot}'."
            )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    cycle_wall_clock_start = time.perf_counter()
    wall_clock_reported = False

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        current_step = int(time_steps[0].item())
        if landmark_report is not None and current_step in landmark_hits and current_step not in announced_landmarks:
            print(f"[INFO] 540 keyframe hit: {landmark_hits[current_step]}={current_step}")
            announced_landmarks.add(current_step)
        reset_ids = time_steps >= motion.time_step_total
        if torch.any(reset_ids) and not wall_clock_reported:
            wall_clock_cycle_duration_s = time.perf_counter() - cycle_wall_clock_start
            print(f"  wall_clock_cycle_duration_s: {wall_clock_cycle_duration_s:.4f}")
            wall_clock_reported = True
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        if robot_joint_indexes is None:
            robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        else:
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps]
            joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps]
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([-2.0, -2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
