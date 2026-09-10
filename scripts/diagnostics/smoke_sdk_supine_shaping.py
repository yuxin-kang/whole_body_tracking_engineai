"""Construction/reset/step smoke for one direct-PPO SDK get-up task."""

from __future__ import annotations

import argparse
import json
import copy
import hashlib
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kind', choices=('supine', 'prone'), default='supine')
parser.add_argument("--variant", choices=("V1", "V2", "V3", "V4"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

task = f"Tracking-Flat-T800-GetUp-SDK-{args.kind.title()}-{args.variant}-PPO-v0"
if not args.usd_dir.is_dir():
    parser.error(f"USD directory is unavailable: {args.usd_dir}")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import whole_body_tracking.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner
from whole_body_tracking.tasks.tracking.config.t800.flat_env_cfg import T800FlatGetUpEnvCfg
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import T800FlatPPORunnerCfg


def differences(old, new, prefix=''):
    result = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in old.keys() | new.keys():
            name = f'{prefix}.{key}' if prefix else key
            if key not in old or key not in new:
                result.append(name)
            else:
                result.extend(differences(old[key], new[key], name))
    elif old != new:
        result.append(prefix)
    return result


def main() -> None:
    spec = gym.spec(task)
    cfg = spec.kwargs["env_cfg_entry_point"]()
    baseline = T800FlatGetUpEnvCfg()
    baseline.commands.motion.motion_file = cfg.commands.motion.motion_file
    baseline.episode_length_s = cfg.episode_length_s
    changes = differences(baseline.to_dict(), cfg.to_dict())
    allowed = (f'rewards.sdk_{args.kind}_', 'rewards.motion_body_lin_vel.',
               'rewards.motion_body_ang_vel.', f'scene.sdk_{args.kind}_',
               'events.sdk_contact_offsets', 'events.sdk_base_mass',
               'commands.motion.pose_range.', 'commands.motion.velocity_range.',
               'commands.motion.joint_position_scale_range', 'commands.motion.joint_velocity_range')
    assert all(name.startswith(allowed) for name in changes), changes
    assert cfg.rewards.motion_global_anchor_pos.weight == 0.5
    assert cfg.rewards.motion_global_anchor_ori.weight == 0.5
    cfg.scene.num_envs = 32
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_dir = str(args.usd_dir.resolve())
    cfg.scene.robot.spawn.force_usd_conversion = True
    cfg.scene.terrain.visual_material = None
    cfg.scene.contact_forces.debug_vis = False
    cfg.commands.motion.debug_vis = False

    env = None
    try:
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
        observations, extras = env.reset()
        assert observations.shape == (32, 134), observations.shape
        assert extras['observations']['critic'].shape == (32, 284)
        action_dim = env.num_actions
        assert action_dim == 25
        actions = torch.zeros((cfg.scene.num_envs, action_dim), device=args.device)
        command = env.unwrapped.command_manager.get_term('motion')
        peaks, sensor_shapes = {}, {}
        max_frame = 0
        for _ in range(152):
            observations, rewards, dones, extras = env.step(actions)
            assert torch.isfinite(observations).all() and torch.isfinite(rewards).all()
            max_frame = max(max_frame, int(command.time_steps.max()))
            for name in env.unwrapped.reward_manager.active_terms:
                if name.startswith(f'sdk_{args.kind}_'):
                    term = env.unwrapped.reward_manager.get_term_cfg(name)
                    value = term.func(env.unwrapped, **term.params)
                    assert torch.isfinite(value).all(), name
                    peaks[name] = max(peaks.get(name, 0.), float(value.abs().max()))
            if args.variant == 'V4':
                for name in (f'sdk_{args.kind}_foot_ground_l', f'sdk_{args.kind}_foot_ground_r'):
                    force = env.unwrapped.scene[name].data.force_matrix_w
                    assert force is not None and torch.isfinite(force).all(), name
                    sensor_shapes[name] = list(force.shape)
        assert max_frame >= (149 if args.kind == 'supine' else 141), max_frame
        agent = T800FlatPPORunnerCfg()
        agent.logger = 'tensorboard'
        agent.device = args.device
        runner = MotionOnPolicyRunner(env, copy.deepcopy(agent.to_dict()),
                                     log_dir=str(args.output.with_suffix('')), device=args.device)
        assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
        initial = [p.detach().clone() for p in runner.alg.policy.parameters()]
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
        assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(initial, runner.alg.policy.parameters()))
        runner.save(str(args.output.with_suffix('.pt')))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        marker = {
            "status": "completed",
            "passed": True,
            "task": task,
            "variant": args.variant,
            "method": "ppo",
            "num_envs": cfg.scene.num_envs,
            "action_dim": action_dim,
            "usd_dir": str(args.usd_dir.resolve()),
            "config_changes": changes,
            "max_reference_frame": max_frame,
            "reward_peaks": peaks,
            "ground_sensor_shapes": sensor_shapes,
            "ppo_updates": 2,
            "motion_sha256": hashlib.sha256(Path(cfg.commands.motion.motion_file).read_bytes()).hexdigest(),
        }
        args.output.write_text(json.dumps(marker, indent=2) + "\n")
        print(json.dumps(marker, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()


try:
    main()
    simulation_app.close()
except Exception:
    traceback.print_exc()
    os._exit(1)
