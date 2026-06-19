# 2026-05-11 Punch Swing Baseline Comparison

## Purpose

Started a clean baseline comparison to check whether recent changes broke the
general T800 tracking setup, or whether the 540 failure is specific to the 540
motion/task variants.

## Slurm Job

```text
852 t800_punch_swing_l_baseline
  task:      Tracking-Flat-T800-v0
  motion:    data/Punch_Swing_L_50hz.npz
  run:       t800-punch-swing-l-50hz-baseline-24h
  partition: rtx3090
  gres:      gpu:rtx3090:1
  envs:      4096
  cpus:      8
  memory:    32G
  time:      24h
  node:      epyc2
```

The job used the same IsaacLab/Slurm stability overrides as the recent T800
runs:

```bash
env.scene.terrain.visual_material=null
env.scene.terrain.physics_material=null
env.commands.motion.debug_vis=false
env.scene.contact_forces.debug_vis=false
```

## Saved Config

Run directory:

```text
logs/rsl_rl/t800_flat/2026-05-11_09-43-52_t800-punch-swing-l-50hz-baseline-24h
```

Confirmed from saved configs:

```text
seed: 42
num_envs: 4096
motion_file: /srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/data/Punch_Swing_L_50hz.npz
experiment_name: t800_flat
run_name: t800-punch-swing-l-50hz-baseline-24h
max_iterations: 30000
```

Slurm logs:

```text
slurm_logs/t800_punch_swing_l_baseline-852.out
slurm_logs/t800_punch_swing_l_baseline-852.err
```

## Runtime Check

At `2026-05-11 09:56:46 CST`, Slurm reported:

```text
852 t800_punch_swing_l_baseline RUNNING 13:16 epyc2 gres/gpu
```

Training had entered the RSL-RL loop and reached:

```text
Learning iteration 347/30000
Mean reward: 0.76
Mean episode length: 11.59
Metrics/motion/error_anchor_pos: 0.2505
Metrics/motion/error_body_pos: 0.2283
Metrics/motion/error_body_rot: 0.1455
Metrics/motion/error_joint_pos: 0.5324
Metrics/motion/error_joint_vel: 6.8224
Episode_Termination/anchor_pos: 0.8432
Episode_Termination/ee_body_pos: 0.9274
```

`model_0.pt` and TensorBoard event output were created in the run directory.
stderr contains an Isaac ground-plane `ChangePropertyCommand` stack, but the
job continued, completed environment setup, and started training.

## Interpretation

This run is the control branch:

- If `Tracking-Flat-T800-v0` with `Punch_Swing_L_50hz.npz` trains normally, the
  current 540 issue is more likely tied to the 540 data, task-specific
  settings, phase/termination choices, or motion preprocessing.
- If this baseline also fails to improve, the regression is more likely in the
  shared T800 tracking environment, command logic, rewards, or recent global
  config changes.

## 19:39 Base 540 / Zhiquan Physmat Single-Segment Runs

Started two more comparison runs using the current `physmat_fixed` launch
method. These are the base single-segment tasks, not the `OrigEpisode`,
`Bridge`, or other episode-lengthened variants.

Launch method difference from the earlier null-material workaround:

```bash
env.scene.terrain.visual_material=null
env.commands.motion.debug_vis=false
env.scene.contact_forces.debug_vis=false
```

The terrain `physics_material` was intentionally kept enabled.

Jobs:

```text
1195 t800_540_base_physmat_1seg
  task:      Tracking-Flat-T800-540Huixuanti1-v0
  motion:    data/npz/540/cut/540huixuantitui1.npz
  run:       t800-540huixuanti1-base-physmat-1seg-24h
  log dir:   logs/rsl_rl/t800_flat/2026-05-11_19-39-45_t800-540huixuanti1-base-physmat-1seg-24h

1196 t800_zhiquan_base_physmat_1seg
  task:      Tracking-Flat-T800-Zhiquan-v0
  motion:    data/npz/zhiquan/cut/zhiquan_quanji1.npz
  run:       t800-zhiquan-base-physmat-1seg-24h
  log dir:   logs/rsl_rl/t800_flat/2026-05-11_19-39-59_t800-zhiquan-base-physmat-1seg-24h
```

Saved config checks confirmed:

```text
540:     episode_length_s: 5.8,  num_envs: 4096, max_iterations: 30000
zhiquan: episode_length_s: 13.32, num_envs: 4096, max_iterations: 30000
```

Runtime check at `2026-05-11 19:45:05 CST`:

```text
1195 t800_540_base_physmat_1seg     RUNNING 5:42 epyc2 gres/gpu
1196 t800_zhiquan_base_physmat_1seg RUNNING 5:26 epyc2 gres/gpu
```

Early metrics:

```text
540 iteration 86/30000
  Mean reward: -0.41
  Mean episode length: 37.15
  Metrics/motion/error_anchor_pos: 1.1511
  Metrics/motion/error_body_pos: 0.6002
  Metrics/motion/error_joint_pos: 2.2579
  Metrics/motion/error_joint_vel: 32.0430
  Episode_Termination/anchor_pos: 1.0000

zhiquan iteration 63/30000
  Mean reward: 0.18
  Mean episode length: 13.72
  Metrics/motion/error_anchor_pos: 0.2062
  Metrics/motion/error_body_pos: 0.2111
  Metrics/motion/error_joint_pos: 1.8850
  Metrics/motion/error_joint_vel: 26.4535
  Episode_Termination/anchor_pos: 0.0010
  Episode_Termination/ee_body_pos: 0.9997
```

Tomorrow's check should compare final or late-run TensorBoard metrics for
`1195` and `1196` against the earlier null-physics-material and episode-variant
runs.
