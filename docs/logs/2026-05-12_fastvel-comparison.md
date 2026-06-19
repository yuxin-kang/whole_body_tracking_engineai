# 2026-05-12 Fastvel Comparison Log

## Summary

This session reviewed the previous T800 base physmat runs and started matched
fastvel comparison runs for zhiquan and 540 huixuanti.

The goal is to compare the same task/environment setup against the
`*_fast50vel.npz` motion variants, without changing task config, Slurm
resources, or the physmat launch method.

## Previous Run Data Checks

Saved configs confirmed the 2026-05-11 base physmat runs did not use fastvel
data:

```text
2026-05-11_19-39-59_t800-zhiquan-base-physmat-1seg-24h
  motion: data/npz/zhiquan/cut/zhiquan_quanji1.npz
  task:   Tracking-Flat-T800-Zhiquan-v0

2026-05-11_19-39-45_t800-540huixuanti1-base-physmat-1seg-24h
  motion: data/npz/540/cut/540huixuantitui1.npz
  task:   Tracking-Flat-T800-540Huixuanti1-v0
```

The 540 run also confirmed:

```text
episode_length_s: 5.8
pd_stand_reset_ratio: 0.2
resample_at_motion_end: true
seed: 42
max_iterations: 30000
```

## Fastvel Data Interpretation

The `fast50vel` motion files keep the same pose frames as the original data,
but change playback metadata and stored velocity targets:

```text
fps: 12 -> 50
velocity fields: scaled by 50 / 12
pose, orientation, joint position frames: unchanged
```

This explains why `scripts/replay_npz.py` looks visually identical for the
original and fastvel files: replay writes the pose/joint state directly for
rendering. Training can still differ because command targets, rewards, and
reset state use stored velocity fields.

## Jobs Started

Both jobs reused the same physmat launch method as the 2026-05-11 base runs:

```bash
env.scene.terrain.visual_material=null
env.commands.motion.debug_vis=false
env.scene.contact_forces.debug_vis=false
```

The terrain `physics_material` was intentionally kept enabled.

```text
1199 t800_zhiquan_base_fastvel_physmat_1seg
  task:      Tracking-Flat-T800-Zhiquan-v0
  motion:    data/npz/zhiquan/cut/zhiquan_quanji1_fast50vel.npz
  run:       t800-zhiquan-base-fastvel-physmat-1seg-24h
  log dir:   logs/rsl_rl/t800_flat/2026-05-12_10-12-20_t800-zhiquan-base-fastvel-physmat-1seg-24h

1200 t800_540_base_fastvel_physmat_1seg
  task:      Tracking-Flat-T800-540Huixuanti1-v0
  motion:    data/npz/540/cut/540huixuantitui1_fast50vel.npz
  run:       t800-540huixuanti1-base-fastvel-physmat-1seg-24h
  log dir:   logs/rsl_rl/t800_flat/2026-05-12_10-30-44_t800-540huixuanti1-base-fastvel-physmat-1seg-24h
```

Saved config checks confirmed:

```text
seed: 42
num_envs: 4096
experiment_name: t800_flat
max_iterations: 30000
```

## Runtime Check

At `2026-05-12 10:52 CST`, Slurm reported:

```text
1199 t800_zhiquan_base_fastvel_physmat_1seg RUNNING 40:00 epyc2
1200 t800_540_base_fastvel_physmat_1seg     RUNNING 21:35 epyc2
```

Zhiquan fastvel had started improving by the latest check:

```text
Learning iteration 757/30000
Mean reward: 11.66
Mean episode length: 202.75
Metrics/motion/error_anchor_pos: 0.5200
Metrics/motion/error_body_pos: 0.2726
Metrics/motion/error_joint_pos: 1.7007
Metrics/motion/error_joint_vel: 21.4158
Episode_Termination/anchor_pos: 0.0546
Episode_Termination/ee_body_pos: 0.9202
```

The 540 fastvel run was still poor early:

```text
Learning iteration 486/30000
Mean reward: 1.37
Mean episode length: 39.93
Metrics/motion/error_anchor_pos: 1.0291
Metrics/motion/error_body_pos: 0.5526
Metrics/motion/error_joint_pos: 2.4735
Metrics/motion/error_joint_vel: 30.8595
Episode_Termination/anchor_pos: 0.9998
```

## Interpretation

The current evidence supports treating zhiquan and 540 separately:

- Zhiquan fastvel is not obviously harmful in the early run and is improving.
- 540 fastvel still shows near-constant anchor position termination early, so
  the 540 issue is not solved by velocity-matched fast playback alone.
- Since the previous 540 base physmat run also failed badly, the likely issue
  remains tied to the 540 motion/task configuration, termination thresholds,
  reset distribution, or motion content rather than only the fastvel conversion.

## Follow-up

Compare late-run TensorBoard curves and play checkpoints for:

```text
1195 t800-540huixuanti1-base-physmat-1seg-24h
1196 t800-zhiquan-base-physmat-1seg-24h
1199 t800-zhiquan-base-fastvel-physmat-1seg-24h
1200 t800-540huixuanti1-base-fastvel-physmat-1seg-24h
```

The main metrics to compare are:

- `Episode_Termination/anchor_pos`
- `Metrics/motion/error_anchor_pos`
- `Metrics/motion/error_body_pos`
- `Metrics/motion/error_joint_pos`
- `Metrics/motion/error_joint_vel`
- mean reward and playable checkpoint quality
