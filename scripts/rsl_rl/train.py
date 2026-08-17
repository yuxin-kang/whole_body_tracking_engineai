# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
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


def _configure_t800_motion_episode(env_cfg, motion_file: str) -> None:
    """Keep T800 motions native-length without overriding task reset semantics."""
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
    env_cfg.episode_length_s = num_frames / fps
    phase_mode = "start" if motion_cfg.play_from_start else motion_cfg.sampling_mode
    end_mode = "resample" if motion_cfg.resample_at_motion_end else "hold"
    print(
        f"[INFO] T800 native motion episode: {num_frames} frames / {fps:g} Hz "
        f"= {env_cfg.episode_length_s:.4f}s; no bridge, phase={phase_mode}, motion_end={end_mode}"
    )


def _configure_t800_punch_normal_episode(env_cfg) -> None:
    """Restore the training distribution used by the successful Punch_normal run."""
    motion_cfg = env_cfg.commands.motion
    env_cfg.episode_length_s = 10.0
    motion_cfg.min_traj_duration = 10.0
    motion_cfg.bridge_frames = 20
    motion_cfg.sampling_mode = "adaptive"
    motion_cfg.phase_sampling_windows = []
    motion_cfg.pd_stand_reset_ratio = 0.2
    motion_cfg.reset_preroll_frames = 0
    motion_cfg.play_from_start = False
    motion_cfg.resample_at_motion_end = True
    env_cfg.events.push_robot.interval_range_s = (1.0, 3.0)
    print(
        "[INFO] T800 Punch_normal episode: 10.0000s; min_traj_duration=10.0000s, "
        "bridge=20, phase=adaptive, motion_end=resample, pd_stand_reset_ratio=0.2, "
        "push_interval=(1.0, 3.0)s"
    )


def _learn_with_code_state_fallback(
    runner, num_learning_iterations: int, init_at_random_ep_len: bool = True
):
    """Retry training without git snapshot logging if diff encoding fails."""
    try:
        runner.learn(
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )
    except UnicodeEncodeError as err:
        git_status_repos = getattr(runner, "git_status_repos", None)
        if git_status_repos and "surrogates not allowed" in str(err):
            print(
                "[WARN] Failed to store git diff due to non-UTF8 content. "
                "Retrying with code state logging disabled."
            )
            runner.git_status_repos = []
            runner.learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
            return
        raise


def dump_pickle(filename: str, data):
    """Persist data as pickle without relying on IsaacLab utility exports."""
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(data, file)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a local motion .npz file.")
parser.add_argument(
    "--t800_episode_mode",
    choices=("native", "punch_normal"),
    default="native",
    help="T800 trajectory organization: native single clip or the successful Punch_normal 10 s bridged setup.",
)
parser.add_argument(
    "--no_terminations",
    type=_str2bool,
    default=False,
    help="Disable all robot terminations except time_out.",
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
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # load the motion file
    registry_name = args_cli.registry_name
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    elif registry_name is not None:
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    elif isinstance(env_cfg.commands.motion.motion_file, (str, os.PathLike)) and os.fspath(
        env_cfg.commands.motion.motion_file
    ):
        print(f"[INFO] Using motion file configured by task: {env_cfg.commands.motion.motion_file}")
    else:
        raise ValueError("Provide --motion_file or --registry_name, or select a task with a configured motion file.")

    is_t800_task = bool(args_cli.task and "t800" in args_cli.task.lower())
    if is_t800_task:
        if args_cli.t800_episode_mode == "punch_normal":
            _configure_t800_punch_normal_episode(env_cfg)
        else:
            _configure_t800_motion_episode(env_cfg, env_cfg.commands.motion.motion_file)

    if args_cli.no_terminations:
        _disable_robot_terminations(env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # Motion-phase sampling already de-synchronizes T800 rollouts. Avoid also
    # shortening the first native-length episode through runner bookkeeping.
    is_native_t800_episode = is_t800_task and args_cli.t800_episode_mode == "native"
    _learn_with_code_state_fallback(
        runner,
        agent_cfg.max_iterations,
        init_at_random_ep_len=not is_native_t800_episode,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
