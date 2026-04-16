"""Interactively trim a motion npz file while replaying it in Isaac Sim.

.. code-block:: bash

    python scripts/trim_npz.py --input_file data/riot_combo_50hz.npz --robot t800 --start_paused
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Interactively trim a replayable motion npz file.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the source motion .npz file.")
parser.add_argument("--output_file", type=str, default=None, help="Optional output path for the trimmed motion.")
parser.add_argument("--robot", type=str, default="t800", choices=["pm01", "t800"], help="Robot type to use.")
parser.add_argument("--start_paused", action="store_true", help="Start replay paused at frame 0.")
parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists.")
parser.add_argument(
    "--debug_print_every",
    type=int,
    default=0,
    help="Print frame/root/joint debug info every N rendered frames. Disabled when set to 0.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

input_path = Path(args_cli.input_file).resolve()
output_path = None if args_cli.output_file is None else Path(args_cli.output_file).resolve()
if input_path.suffix.lower() != ".npz":
    raise ValueError(f"Expected a .npz motion file, got: {input_path}")
if output_path is not None and output_path == input_path:
    raise ValueError("Input and output motion files must be different.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb.input
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.pm01 import PM01_CYLINDER_CFG
from whole_body_tracking.robots.t800 import T800_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


def _load_motion_trim_helpers():
    module_path = Path(__file__).resolve().parents[1] / "source/whole_body_tracking/whole_body_tracking/utils/motion_trim.py"
    spec = spec_from_file_location("motion_trim_helpers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load motion trim helpers from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_motion_trim = _load_motion_trim_helpers()
default_trim_output_path = _motion_trim.default_trim_output_path
infer_total_frames = _motion_trim.infer_total_frames
load_motion_npz = _motion_trim.load_motion_npz
save_trimmed_motion_npz = _motion_trim.save_trimmed_motion_npz
trim_motion_dict = _motion_trim.trim_motion_dict


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
    """Configuration for an interactive trim scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class InteractiveTrimSession:
    def __init__(
        self,
        motion_file: Path,
        motion_data: dict[str, np.ndarray],
        fps: float,
        start_paused: bool,
        output_file: Path | None,
        force: bool,
    ):
        self.motion_file = motion_file
        self.motion_data = motion_data
        self.fps = fps
        self.total_frames = infer_total_frames(motion_data)
        self.current_frame = 0
        self.is_playing = not start_paused
        self.start_frame: int | None = None
        self.end_frame: int | None = None
        self.output_file = output_file
        self.force = force
        self.should_exit = False
        self.last_export_path: Path | None = None

    def print_controls(self):
        print("[INFO] Interactive trim controls")
        print("  SPACE: play/pause")
        print("  LEFT / RIGHT: step backward/forward while paused")
        print("  [ or S: mark start")
        print("  ] or E: mark end")
        print("  C: clear markers")
        print("  ENTER: export trimmed motion")
        print("  Q or ESCAPE: quit")
        self.print_status(prefix="[INFO] Initial state")

    def print_status(self, prefix: str = "[INFO] State"):
        logical_time = self.current_frame / max(self.fps, 1e-8)
        start = "-" if self.start_frame is None else str(self.start_frame)
        end = "-" if self.end_frame is None else str(self.end_frame)
        playback_state = "PLAYING" if self.is_playing else "PAUSED"
        print(
            f"{prefix}: frame={self.current_frame}/{self.total_frames - 1} "
            f"time={logical_time:.3f}s state={playback_state} start={start} end={end}"
        )

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.print_status(prefix="[INFO] Toggled playback")

    def step(self, delta: int):
        if self.is_playing:
            return
        new_frame = int(np.clip(self.current_frame + delta, 0, self.total_frames - 1))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.print_status(prefix="[INFO] Stepped frame")

    def advance_playback(self):
        if not self.is_playing:
            return
        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.print_status(prefix="[INFO] Reached final frame")
            return
        self.current_frame += 1

    def mark_start(self):
        self.start_frame = self.current_frame
        self.print_status(prefix="[INFO] Marked start")

    def mark_end(self):
        self.end_frame = self.current_frame
        self.print_status(prefix="[INFO] Marked end")

    def clear_markers(self):
        self.start_frame = None
        self.end_frame = None
        self.print_status(prefix="[INFO] Cleared markers")

    def request_exit(self):
        self.should_exit = True
        print("[INFO] Exit requested")

    def export_trim(self):
        if self.start_frame is None or self.end_frame is None:
            print("[WARN] Both start and end markers must be set before export.")
            return
        if self.start_frame > self.end_frame:
            print("[WARN] Start marker must be less than or equal to end marker.")
            return

        output_path = self.output_file or default_trim_output_path(self.motion_file, self.start_frame, self.end_frame)
        if output_path.resolve() == self.motion_file.resolve():
            print("[WARN] Refusing to overwrite the source motion file.")
            return

        trimmed = trim_motion_dict(self.motion_data, self.start_frame, self.end_frame)
        try:
            saved_path = save_trimmed_motion_npz(trimmed, output_path, force=self.force)
        except FileExistsError as exc:
            print(f"[WARN] {exc}")
            return

        trimmed_frames = infer_total_frames(trimmed)
        trimmed_duration = max(trimmed_frames - 1, 0) / max(self.fps, 1e-8)
        self.last_export_path = saved_path
        print("[INFO] Trim export complete")
        print(f"  input_file: {self.motion_file}")
        print(f"  output_file: {saved_path}")
        print(f"  start_frame: {self.start_frame}")
        print(f"  end_frame: {self.end_frame}")
        print(f"  source_frames: {self.total_frames}")
        print(f"  trimmed_frames: {trimmed_frames}")
        print(f"  fps: {self.fps:.4f}")
        print(f"  trimmed_duration_s: {trimmed_duration:.4f}")

    def handle_key(self, key_name: str):
        if key_name == "SPACE":
            self.toggle_play_pause()
        elif key_name == "LEFT":
            self.step(-1)
        elif key_name == "RIGHT":
            self.step(1)
        elif key_name in {"LEFT_BRACKET", "S"}:
            self.mark_start()
        elif key_name in {"RIGHT_BRACKET", "E"}:
            self.mark_end()
        elif key_name == "C":
            self.clear_markers()
        elif key_name == "ENTER":
            self.export_trim()
        elif key_name in {"Q", "ESCAPE"}:
            self.request_exit()


def _apply_frame(
    frame_index: int,
    motion: MotionLoader,
    robot: Articulation,
    scene: InteractiveScene,
    sim: sim_utils.SimulationContext,
    robot_joint_indexes: torch.Tensor | None,
    sim_dt: float,
    debug_print_every: int = 0,
):
    time_steps = torch.full((scene.num_envs,), frame_index, dtype=torch.long, device=sim.device)
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
    sim.forward()
    sim.render()
    scene.update(sim_dt)

    if debug_print_every > 0 and frame_index % debug_print_every == 0:
        joint_sample = motion.joint_pos[time_steps][0, : min(3, motion.joint_pos.shape[1])].detach().cpu().tolist()
        root_sample = root_states[0, :3].detach().cpu().tolist()
        print(
            f"[DEBUG] frame={frame_index} "
            f"root_xyz={[round(v, 4) for v in root_sample]} "
            f"joint_pos_0_2={[round(v, 4) for v in joint_sample]}"
        )

    pos_lookat = root_states[0, :3].cpu().numpy()
    sim.set_camera_view(pos_lookat + np.array([-2.0, -2.0, 0.5]), pos_lookat)


def _unsubscribe_keyboard(input_interface, keyboard, subscription):
    if subscription is None:
        return
    if hasattr(input_interface, "unsubscribe_to_keyboard_events"):
        input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
    elif hasattr(input_interface, "unsubscribe_from_keyboard_events"):
        input_interface.unsubscribe_from_keyboard_events(keyboard, subscription)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion_data = load_motion_npz(input_path)
    motion = MotionLoader(
        str(input_path),
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    motion_joint_names = ROBOT_MOTION_JOINT_NAMES.get(args_cli.robot)
    robot_joint_indexes = None
    if motion_joint_names is not None:
        robot_joint_indexes = robot.find_joints(motion_joint_names, preserve_order=True)[0]
        if len(robot_joint_indexes) != motion.joint_pos.shape[1]:
            raise ValueError(
                f"Motion joint count ({motion.joint_pos.shape[1]}) does not match mapped robot joints "
                f"({len(robot_joint_indexes)}) for robot '{args_cli.robot}'."
            )

    session = InteractiveTrimSession(
        motion_file=input_path,
        motion_data=motion_data,
        fps=motion.fps,
        start_paused=args_cli.start_paused,
        output_file=output_path,
        force=args_cli.force,
    )
    session.print_controls()

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()

    def _on_keyboard_event(event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            session.handle_key(event.input.name)
        return True

    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)

    try:
        while simulation_app.is_running() and not session.should_exit:
            if session.is_playing:
                session.advance_playback()

            _apply_frame(
                session.current_frame,
                motion,
                robot,
                scene,
                sim,
                robot_joint_indexes,
                sim_dt,
                debug_print_every=args_cli.debug_print_every,
            )
    finally:
        _unsubscribe_keyboard(input_interface, keyboard, keyboard_subscription)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
