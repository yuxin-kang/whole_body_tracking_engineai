# Ridicule2 startup repair — Job 9884

## Motivation and evidence

User reports baseline `2026-09-08_20-32-49_t800_ridicule2_baseline_scratch_3090_j9830/model_29999.pt`
fails during client Play. Previous diagnostic rollout matched the reference at reset but
diverged within ten steps. This does not by itself prove a unique cause; the client and
training simulator versions also differ.

Original first-frame maximum absolute single-joint velocity: 25.14117 rad/s;
base angular speed: 18.003075 rad/s. At original output frame 20 (0.4 s),
maximum single-joint speed is 0.22332 rad/s and base angular speed 0.04312 rad/s.
The source CSV itself contains the early discontinuity. Treating the first 0.4 s as
retargeting startup transient is an experimental assumption, not confirmed provenance.
Previously reported 39.2 and 134.8 are aggregate norms over all body nodes, not
individual body speeds.

## Data

- Original source: `data/npz/ridicule_2/g1_lafan1_dance1_subject2_upperbody_f060_240_t800.csv`, 30 FPS.
- New motion: `data/npz/ridicule_2/ridicule2_t800_startfix_50hz.npz`.
- Remove first 12 source frames (0.4 s); retain remaining 168 poses.
- Start hold 0.4 s; ease playback speed over 0.4 s at both ends; end hold 1.0 s.
- Smooth time mapping integrates smoothstep speed. Root position and joint angles
  interpolate linearly; root quaternion uses SLERP; all bodies recomputed by URDF FK.
- Recompute all velocities from the retimed poses; seven standard NPZ keys, 50 Hz,
  370 output frames (timestamp span 7.38 s; environment episode length 7.4 s).
- First-frame velocities are zero. Full motion peak single-joint speed 6.174787 rad/s.
- Original files and checkpoints preserved.
- SHA256: `a0f82df232ce6dc2f866fb3cfbf1fbf30275b49c7169b36c1ff0ebaba249a541`.

Reproduce:

```bash
python scripts/convert_t800_pose_csv.py \
  --input data/npz/ridicule_2/g1_lafan1_dance1_subject2_upperbody_f060_240_t800.csv \
  --output data/npz/ridicule_2/ridicule2_t800_startfix_50hz.npz \
  --input-fps 30 --output-fps 50 \
  --trim-start-frames 12 --start-hold 0.4 --end-hold 1.0 --speed-ramp 0.4
```

Converter refuses to overwrite an existing output.

## Training

- Submit: `sbatch scripts/slurm/train_t800_ridicule2_startfix_3090.sbatch`.
- Job 9884: 1 RTX 3090 on epyc2.
- Verified RUNNING and PPO iteration 18/30000. Initial mean episode length 9.32
  steps (target 370); this is a scratch initialization check, not a success result.
- Run: `2026-09-09_23-20-29_t800_ridicule2_startfix_baseline_scratch_3090_j9884`.
- W&B: https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/ridicule9884
- Task `Tracking-Flat-T800-v0`, baseline PPO and reward weights.
- Scratch (`agent.resume=false`), seed 42, 4096 environments, 30000 iterations.
- `env.commands.motion.sampling_mode=start`: all episodes learn from frame zero.
- `env.commands.motion.resample_at_motion_end=false`: hold final pose instead of
  resampling/teleporting to another phase at trajectory end.
- Other baseline randomization and termination settings retained.
- Tradeoff: explicit start sampling tests complete-action ability but can slow learning
  of later phases. Training curves alone are not sufficient evidence of success.

## Validation and client

Four timing tests pass. NPZ validation checks seven keys, finiteness, unit quaternions,
pose-derived linear/joint/angular velocities, stationary start/end and matching the
retained original pose. A physics success test remains pending a trained checkpoint.

New motion and metadata copied to client
`/home/kyx/robot/mimic/whole_body_tracking_engineai/data/npz/ridicule_2/`;
server/client NPZ SHA256 match. Preview reference (not learned policy):

```bash
python scripts/replay_npz.py \
  --robot t800 \
  --input_file data/npz/ridicule_2/ridicule2_t800_startfix_50hz.npz \
  --follow_camera false \
  --device cuda:0
```

Acceptance: evaluate checkpoint on complete frame-zero episodes with normal failure
terminations, recording completion rate, anchor/body errors, and ending stability.
Compare training-stack and client-stack playback before claiming the original failure
fully resolved. Do not judge success from a forced no-termination rollout alone.
