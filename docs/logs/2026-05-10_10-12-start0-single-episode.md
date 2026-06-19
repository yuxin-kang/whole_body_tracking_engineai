# 2026-05-10 10:12 Start0 Single-Episode Validation Log

## Summary

This session added a minimal `single-episode-start0` validation task for the
T800 540 huixuanti cut. The purpose is to isolate whether the short 5.8 second
episode is itself problematic, or whether the failure comes from random reset
start frames and random motion resampling at the end of the clip.

## Hypothesis

The short episode alone should be testable if each reset starts at frame 0 and
the command does not jump to a random new motion frame when the clip reaches
its end. If this variant trains better than the previous shortened-episode
runs, the main issue is likely random start / end-of-clip resampling rather
than episode length.

## Code Changes

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
  - Added `MotionCommandCfg.play_from_start`, default `False`.
  - Added `MotionCommandCfg.resample_at_motion_end`, default `True`.
  - Wired `_play_from_start` to the config instead of hardcoding `False`.
  - When `resample_at_motion_end=False`, command time is clamped to the last
    motion frame instead of calling `_resample_command`.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py`
  - Added `T800Flat540Huixuanti1Start0EnvCfg`.
  - This config inherits the existing 540 huixuanti config and sets:
    - `commands.motion.play_from_start = True`
    - `commands.motion.resample_at_motion_end = False`
    - `commands.motion.pd_stand_reset_ratio = 0.0`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py`
  - Registered `Tracking-Flat-T800-540Huixuanti1-Start0-v0`.
- `.gitignore`
  - Added an exception for the new focused regression test.
- `tests/test_540_start0_single_episode.py`
  - Added a lightweight static regression test covering the task registration,
    start0 config, and no-resample command support.

## Validation

Commands run:

```bash
/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid/bin/python -m py_compile \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py \
  tests/test_540_start0_single_episode.py

/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid/bin/python -m pytest \
  tests/test_540_start0_single_episode.py -v
```

Result:

```text
3 passed in 0.02s
```

The system Python did not have `pytest`; the IsaacLab conda environment was
used for the passing test run.

## Slurm Job Started

```text
704 t800_540_huixuanti1_start0
  task:      Tracking-Flat-T800-540Huixuanti1-Start0-v0
  motion:    data/npz/540/cut/540huixuantitui1.npz
  run:       t800-540huixuanti1-start0-24h
  partition: rtx3090
  gres:      gpu:rtx3090:1
  envs:      4096
  cpus:      8
  memory:    32G
  time:      24h
  node:      epyc2
```

The job used the same IsaacLab/Slurm stability overrides as the previous
training runs:

```bash
env.scene.terrain.visual_material=null
env.scene.terrain.physics_material=null
env.commands.motion.debug_vis=false
env.scene.contact_forces.debug_vis=false
```

## Saved Config Check

Saved Hydra configs under:

```text
logs/rsl_rl/t800_flat/2026-05-10_10-05-48_t800-540huixuanti1-start0-24h
```

confirmed:

```text
num_envs: 4096
episode_length_s: 5.8
motion_file: /srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/data/npz/540/cut/540huixuantitui1.npz
play_from_start: true
resample_at_motion_end: false
pd_stand_reset_ratio: 0.0
max_iterations: 30000
run_name: t800-540huixuanti1-start0-24h
```

## Runtime Check

At `2026-05-10 10:12:55 CST`, Slurm reported:

```text
704 t800_540_huixuanti1_start0 RUNNING 00:07:29 epyc2 gres/gpu
AllocTRES: billing=8,cpu=8,gres/gpu=1,mem=32G,node=1
```

Training had entered the RSL-RL loop and reached:

```text
Learning iteration 183/30000
Mean reward: 1.29
Metrics/motion/sampling_entropy: 0.0000
Episode_Termination/anchor_pos: 1.0000
ETA: 13:43:38
```

`sampling_entropy: 0.0000` is expected for this validation because all resets
start from frame 0. The early `anchor_pos` termination is still high and should
be compared against the previous shortened-episode runs after more training
iterations.

## Follow-up

After the job has run longer, compare the following metrics against the four
2026-05-09 runs:

- `Episode_Termination/anchor_pos`
- `Metrics/motion/error_anchor_pos`
- `Metrics/motion/error_body_pos`
- `Metrics/motion/error_body_rot`
- `Metrics/motion/error_joint_pos`
- mean reward and checkpoint progression

The key question is whether fixed start / no end resampling lowers termination
or improves tracking while keeping the same short episode length.

## 16:25 Zhiquan Velocity-Matched Fast Playback

The zhiquan motion was converted the same way as the 540 cut fast-playback
variant: keep the same frames, preserve the original file, set the saved fps to
50, and scale stored velocity fields by `50 / 12` so velocity targets match the
50 Hz policy-step playback.

Generated data:

```text
data/npz/zhiquan/cut/zhiquan_quanji1_fast50vel.npz
```

Source data was left unchanged:

```text
data/npz/zhiquan/cut/zhiquan_quanji1.npz
```

Data check:

```text
source fps: 12
target fps: 50
velocity scale: 4.1666667
frames: 666
target duration: 13.32s
```

Submitted Slurm job:

```text
736 t800_zhiquan_fast50vel
  task:      Tracking-Flat-T800-Zhiquan-v0
  motion:    data/npz/zhiquan/cut/zhiquan_quanji1_fast50vel.npz
  run:       t800-zhiquan-fast50vel-24h
  partition: rtx3090
  gres:      gpu:rtx3090:1
  envs:      4096
  cpus:      8
  memory:    32G
  time:      24h
  node:      epyc2
```

Saved config under:

```text
logs/rsl_rl/t800_flat/2026-05-10_16-25-13_t800-zhiquan-fast50vel-24h
```

confirmed:

```text
num_envs: 4096
episode_length_s: 13.32
motion_file: /srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/data/npz/zhiquan/cut/zhiquan_quanji1_fast50vel.npz
max_iterations: 30000
run_name: t800-zhiquan-fast50vel-24h
```

Runtime check at startup:

```text
Learning iteration 15/30000
Mean reward: -0.53
Metrics/motion/error_anchor_pos: 0.2838
Metrics/motion/error_joint_vel: 28.5456
Episode_Termination/anchor_pos: 0.0481
Episode_Termination/ee_body_pos: 0.9963
```

This run had entered the RSL-RL loop successfully. The early issue to watch is
not anchor termination yet, but high end-effector termination and high velocity
tracking error.

## 19:33 Zhiquan Bridge / Min-Traj Validation

Added a zhiquan-only config variant to test whether the previous single-episode
change accidentally removed stabilizing motion stitching behavior.

Code changes:

```text
T800FlatZhiquanBridgeEnvCfg
  inherits: T800FlatZhiquanEnvCfg
  commands.motion.min_traj_duration = episode_length_s
  commands.motion.bridge_frames = 20

Tracking-Flat-T800-Zhiquan-Bridge-v0
```

Submitted Slurm job:

```text
765 t800_zhiquan_bridge
  task:      Tracking-Flat-T800-Zhiquan-Bridge-v0
  motion:    data/npz/zhiquan/cut/zhiquan_quanji1.npz
  run:       t800-zhiquan-bridge-24h
  partition: rtx3090
  gres:      gpu:rtx3090:1
  envs:      4096
  cpus:      8
  memory:    32G
  time:      24h
  node:      epyc2
```

Saved config under:

```text
logs/rsl_rl/t800_flat/2026-05-10_19-33-13_t800-zhiquan-bridge-24h
```

confirmed:

```text
episode_length_s: 13.32
motion_file: /srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/data/npz/zhiquan/cut/zhiquan_quanji1.npz
min_traj_duration: 13.32
bridge_frames: 20
pd_stand_reset_ratio: 0.2
play_from_start: false
resample_at_motion_end: true
max_iterations: 30000
run_name: t800-zhiquan-bridge-24h
```

Startup check:

```text
Learning iteration 16/30000
Mean reward: -0.50
Metrics/motion/error_anchor_pos: 0.2808
Metrics/motion/error_body_pos: 0.2745
Metrics/motion/error_joint_pos: 1.3701
Episode_Termination/anchor_pos: 0.0368
Episode_Termination/ee_body_pos: 0.9977
```

The run entered training successfully. Early behavior still terminates mostly
on `ee_body_pos`, so the discriminating comparison should be made after a few
thousand iterations against `733 t800_zhiquan_orig_episode` and
`736 t800_zhiquan_fast50vel`.

## 19:39 Zhiquan Bridge 24s

Cancelled the first zhiquan bridge probe:

```text
765 t800_zhiquan_bridge
state: CANCELLED
elapsed: 00:05:35
```

Updated `T800FlatZhiquanBridgeEnvCfg` to use:

```text
episode_length_s = 24.0
commands.motion.min_traj_duration = 24.0
commands.motion.bridge_frames = 20
```

Submitted replacement Slurm job:

```text
766 t800_zhiquan_bridge24
  task:      Tracking-Flat-T800-Zhiquan-Bridge-v0
  motion:    data/npz/zhiquan/cut/zhiquan_quanji1.npz
  run:       t800-zhiquan-bridge24-24h
  partition: rtx3090
  gres:      gpu:rtx3090:1
  envs:      4096
  cpus:      8
  memory:    32G
  time:      24h
  node:      epyc2
```

Saved config under:

```text
logs/rsl_rl/t800_flat/2026-05-10_19-39-15_t800-zhiquan-bridge24-24h
```

confirmed:

```text
episode_length_s: 24.0
motion_file: /srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/data/npz/zhiquan/cut/zhiquan_quanji1.npz
min_traj_duration: 24.0
bridge_frames: 20
pd_stand_reset_ratio: 0.2
play_from_start: false
resample_at_motion_end: true
max_iterations: 30000
run_name: t800-zhiquan-bridge24-24h
```

Startup check:

```text
Learning iteration 16/30000
Mean reward: -0.53
Metrics/motion/error_anchor_pos: 0.2774
Metrics/motion/error_body_pos: 0.2713
Metrics/motion/error_joint_pos: 1.3881
Episode_Termination/anchor_pos: 0.0379
Episode_Termination/ee_body_pos: 0.9966
```
