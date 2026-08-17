"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import sys

import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _disable_robot_terminations(env_cfg):
    if not hasattr(env_cfg, "terminations") or env_cfg.terminations is None:
        return
    for name in list(vars(env_cfg.terminations).keys()):
        if name != "time_out":
            setattr(env_cfg.terminations, name, None)


def _configure_t800_motion_episode(env_cfg, motion_file: str, play_from_start: bool) -> None:
    with np.load(motion_file, allow_pickle=False) as motion:
        if "joint_pos" not in motion or "fps" not in motion:
            raise ValueError(f"Motion file must contain joint_pos and fps: {motion_file}")
        num_frames = int(motion["joint_pos"].shape[0])
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])

    if num_frames <= 0 or fps <= 0.0:
        raise ValueError(f"Invalid motion length/fps in {motion_file}: frames={num_frames}, fps={fps}")

    motion_cfg = env_cfg.commands.motion
    motion_cfg.min_traj_duration = None
    motion_cfg.bridge_frames = 0
    motion_cfg.pd_stand_reset_ratio = 0.0
    if play_from_start:
        motion_cfg.play_from_start = True
        motion_cfg.resample_at_motion_end = False
    env_cfg.episode_length_s = num_frames / fps
    phase_mode = "start" if motion_cfg.play_from_start else motion_cfg.sampling_mode
    end_mode = "resample" if motion_cfg.resample_at_motion_end else "hold"
    print(
        f"[INFO] T800 native motion playback: {num_frames} frames / {fps:g} Hz "
        f"= {env_cfg.episode_length_s:.4f}s; phase={phase_mode}, motion_end={end_mode}"
    )


def _extract_obs(observations):
    if isinstance(observations, tuple):
        return observations[0]
    return observations


def _configure_play_viewer(env_cfg, follow_camera: bool) -> None:
    """Use a nearby static camera so interactive mouse controls remain effective."""
    if follow_camera:
        return
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.body_name = None
    env_cfg.viewer.eye = (-1.8, -1.8, 1.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.8)
    print("[INFO] Free viewer camera enabled; use --follow_camera true to track the robot.")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many policy steps even when video recording is disabled.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--no_terminations",
    type=_str2bool,
    default=False,
    help="Disable all robot terminations except time_out.",
)
parser.add_argument(
    "--debug_terminations",
    type=_str2bool,
    default=False,
    help="Print triggered termination terms and error values when an environment ends.",
)
parser.add_argument(
    "--debug_tracking",
    type=_str2bool,
    default=False,
    help="Print finite-rollout tracking and per-joint error statistics.",
)
parser.add_argument(
    "--play_from_start",
    type=_str2bool,
    default=False,
    help="In play mode, always start each motion rollout from frame 0 instead of sampling a random segment.",
)
parser.add_argument(
    "--follow_camera",
    type=_str2bool,
    default=False,
    help="Continuously lock the viewer to the robot. By default the nearby camera remains freely movable.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import subprocess

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.debug_utils import format_termination_debug_report
from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    resolve_rsl_rl_normalizer,
)
from whole_body_tracking.utils.rsl_rl_checkpoint import load_rsl_rl_checkpoint_for_inference


def _print_termination_debug(base_env, dones: torch.Tensor):
    done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
    if done_ids.numel() == 0:
        return

    motion_command = base_env.command_manager.get_term("motion")
    for info in motion_command.get_termination_debug_info(done_ids):
        for line in format_termination_debug_report(info):
            print(line)


def _enable_play_from_start(base_env):
    motion_command = base_env.command_manager.get_term("motion")
    motion_command.set_play_from_start_mode()


def _capture_tracking_sample(motion_command):
    metrics = {
        name: value.detach().cpu()
        for name, value in motion_command.metrics.items()
        if name.startswith("error_")
    }
    return metrics, motion_command.joint_pos.detach().cpu(), motion_command.robot_joint_pos.detach().cpu()


def _print_tracking_summary(motion_command, metric_samples, target_joint_samples, actual_joint_samples, done_count):
    print(f"[TRACKING] steps={len(target_joint_samples)} done_count={done_count}")
    for name in sorted(metric_samples):
        values = torch.cat(metric_samples[name])
        print(
            f"[TRACKING] {name}: mean={values.mean().item():.4f} "
            f"p95={torch.quantile(values, 0.95).item():.4f} max={values.max().item():.4f}"
        )

    targets = torch.cat(target_joint_samples, dim=0)
    actuals = torch.cat(actual_joint_samples, dim=0)
    joint_rmse = torch.sqrt(torch.mean(torch.square(targets - actuals), dim=0))
    target_range = targets.max(dim=0).values - targets.min(dim=0).values
    actual_range = actuals.max(dim=0).values - actuals.min(dim=0).values

    if motion_command.robot_joint_indexes is None:
        joint_names = motion_command.robot.joint_names
    else:
        joint_names = [
            motion_command.robot.joint_names[index]
            for index in motion_command.robot_joint_indexes.detach().cpu().tolist()
        ]

    for index in torch.argsort(joint_rmse, descending=True).tolist():
        print(
            f"[TRACKING_JOINT] {joint_names[index]}: rmse={joint_rmse[index].item():.4f} "
            f"target_range={target_range[index].item():.4f} actual_range={actual_range[index].item():.4f}"
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    if args_cli.task and "t800" in args_cli.task.lower():
        _configure_t800_motion_episode(
            env_cfg,
            env_cfg.commands.motion.motion_file,
            play_from_start=args_cli.play_from_start,
        )

    if args_cli.no_terminations:
        _disable_robot_terminations(env_cfg)

    _configure_play_viewer(env_cfg, follow_camera=args_cli.follow_camera)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.play_from_start:
        _enable_play_from_start(env.unwrapped)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_rsl_rl_checkpoint_for_inference(ppo_runner, resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    normalizer = resolve_rsl_rl_normalizer(ppo_runner)

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # 导出后立即转换为MNN
    onnx_file = os.path.join(export_model_dir, "policy.onnx")
    mnn_file = os.path.join(export_model_dir, "policy.mnn")

    if os.path.exists(onnx_file):
        try:
            mnn_converter_spec = importlib.util.find_spec("MNN.tools.mnnconvert")
        except ModuleNotFoundError:
            mnn_converter_spec = None

        if mnn_converter_spec is None:
            print("[WARN] MNN.tools.mnnconvert is not available in the current Python environment. Skipping MNN export.")
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "MNN.tools.mnnconvert",
                    "-f",
                    "ONNX",
                    "--modelFile",
                    onnx_file,
                    "--MNNModel",
                    mnn_file,
                    "--bizCode",
                    "MNN",
                ],
                check=True,
            )
            print(f"Successfully converted to MNN: {mnn_file}")
    else:
        print(f"ONNX file not found: {onnx_file}")
    # reset environment
    obs = _extract_obs(env.get_observations())
    timestep = 0
    metric_samples = {}
    target_joint_samples = []
    actual_joint_samples = []
    done_count = 0
    motion_command = env.unwrapped.command_manager.get_term("motion") if args_cli.debug_tracking else None
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if motion_command is not None:
                metrics, target_joints, actual_joints = _capture_tracking_sample(motion_command)
                for name, values in metrics.items():
                    metric_samples.setdefault(name, []).append(values)
                target_joint_samples.append(target_joints)
                actual_joint_samples.append(actual_joints)
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            done_count += int(dones.sum().item())
            if args_cli.debug_terminations:
                _print_termination_debug(env.unwrapped, dones)
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

    if motion_command is not None and target_joint_samples:
        _print_tracking_summary(
            motion_command,
            metric_samples,
            target_joint_samples,
            actual_joint_samples,
            done_count,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
