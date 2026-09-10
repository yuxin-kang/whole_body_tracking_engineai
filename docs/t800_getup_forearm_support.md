# C get-up forearm support experiments

User request (2026-09-07): add brief forearm-vertical guidance during get-up,
then run four independent experiments on four RTX 3090 GPUs.

All four fine-tune the same C checkpoint, preserving its policy, critic,
observation normalization and optimizer state:

- Run: `2026-09-06_16-08-31_t800_getup_aggressive_c_scratch_3090_j9746`
- Checkpoint: `model_18500.pt`
- SHA256: `42e1660dde180350f935cd8a72aa757b2c2b42c1b58e86f466a7a9565f037d90`
- Motion: `data/npz/t800_get_up/faint_prone_getup_03_t800.npz`
- Motion SHA256: `e5a1711e492c7fadac84741d11fb09f0ebaa8fe659b68ab3674cb6b8d482812d`

The phrase about support sliding was interpreted as **reducing support slip**
when preparing these four experiments; the user subsequently authorized
implementing and launching the experiments. No prior simulation results are
claimed for the new rewards.

| Variant | Task suffix after `Tracking-Flat-T800-GetUp-Aggressive-C-` | Vertical weight | Slip weight |
| --- | --- | ---: | ---: |
| F1 | `Forearm-F1-v0` | 1.0 | 0.0 |
| F2 | `Forearm-F2-v0` | 3.0 | 0.0 |
| F3 | `Forearm-F3-v0` | 1.0 | -0.5 |
| F4 | `Forearm-F4-v0` | 3.0 | -0.5 |

## Reward definitions

`getup_forearm_vertical` measures the vector from the real wrist support sphere
center to `LINK_ELBOW_PITCH_L/R` in world coordinates. The score is
`exp(-(angle_to_world_up / 0.35)^2)`, averaged over both arms. An upside-down
forearm or airborne wrist does not earn the upright support score.

Do not use the NPZ `LINK_WRIST_PITCH` / `LINK_WRIST_ROLL` entries as physical
wrist-end locations: this motion schema aliases them to `LINK_ELBOW_YAW`.
The runtime uses named `LINK_WRIST_END_L/R` bodies and their actual collision
sphere center offsets: left `(0.026, -0.009, -0.060)`, right
`(0.026, 0.009, -0.060)` m, with sphere radius 0.05 m.

`getup_wrist_support_slip` measures the horizontal velocity of the wrist
sphere's bottom contact point. It includes rigid-body angular motion:
`v_contact = v_com + omega x (p_contact - p_com)`. Its bounded cost is
`1 - exp(-speed_xy^2 / 0.20^2)`. Wrist lift and vertical pushing speed alone
are not penalized as horizontal slip. This term covers wrist-sphere support,
not arbitrary contact points on the forearm box.

Both terms use the same contact and phase gates:

- Separate one-body sensors filter only contacts with
  `/World/ground/terrain/GroundPlane/CollisionPlane`. Self-contact cannot
  trigger support. This follows the installed Isaac Lab 2.1 contact sensor
  implementation's single-body filtered contact contract; see also the
  [official contact sensor documentation](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sensors.html).
- Upward wrist-floor force fades from zero at 5 N to full weight at 30 N;
  additional impact force does not increase the score. This bounds impact
  reward magnitude, but does not impose a minimum contact duration.
- Native reference phase `(0.08, 0.12, 0.30, 0.38)` smoothly fades in at
  0.60--0.91 s and fades out at 2.27--2.87 s for the 379-frame, 50 Hz clip.
  Phase uses reference frames, so mixed-reset episodes receive the correct
  gate regardless of elapsed episode time. This envelope permits support
  adjustment; it is not a ground-truth contact annotation of the reference.
- Forearm-box sensors remain available for diagnosis. Box contact without
  wrist contact cannot activate either reward.

Original C tracking rewards, motion, robot physics, observation/action layouts
and reset curriculum are retained. The continuation uses curriculum offset
18501 so resumed training does not restart C's initial reset mixture.

## Launch and validation

Launcher: `scripts/slurm/finetune_t800_getup_forearm_4x3090.sbatch`.
Four exclusive Slurm steps each receive one RTX 3090. Each experiment uses
4096 environments, seed 42, a fixed learning rate of 1e-5, 5000 additional
updates, and saves every 250 updates. The planned final iteration is 23500.

Before formal training, all four variants must pass a 32-environment preflight
with two PPO updates and a full 379-frame rollout. The checks must verify
checkpoint restoration, unchanged actor/critic widths (134/284), actual
nonzero ground forces and new rewards, finite values, and uninterrupted
reference playback. Preflight proves that the training path runs and the
rewards activate; it does not prove successful MuJoCo transfer.

Each run records source hashes, the configuration delta, restored state,
curriculum offset, Slurm/GPU allocation and W&B URL in `finetune_manifest.json`.
Source checkpoints and existing SDK exports are not replaced by this training.

## Submitted run: 2026-09-07

Job **9768** started at 13:26:45 Asia/Shanghai on `epyc1`; the cluster routed
the four-RTX-3090 request to `rtx3090-large`. Formal training started at
13:27:29 after all four preflights passed. Slurm steps `.4`, `.5`, `.6`, `.7`
each reserve one GPU, eight CPUs and 24 GiB of memory exclusively. Each step
sees its own allocated GPU as local CUDA device 0.

Run directories are under `logs/rsl_rl/t800_flat/`:

| Variant | Run directory | W&B |
| --- | --- | --- |
| F1 | `2026-09-07_13-27-29_t800_getup_forearm_f1_finetune_3090_j9768` | [F1](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/getup-forearm-9768-f1) |
| F2 | `2026-09-07_13-27-29_t800_getup_forearm_f2_finetune_3090_j9768` | [F2](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/getup-forearm-9768-f2) |
| F3 | `2026-09-07_13-27-29_t800_getup_forearm_f3_finetune_3090_j9768` | [F3](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/getup-forearm-9768-f3) |
| F4 | `2026-09-07_13-27-29_t800_getup_forearm_f4_finetune_3090_j9768` | [F4](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/getup-forearm-9768-f4) |

Validation evidence:

- All four 32-environment preflights completed two PPO updates and all
  379 reference frames without interruption. Actual wrist-floor force,
  vertical guidance and raw slip costs were nonzero. Reports are in
  `outputs/diagnostics/getup_forearm_preflight_9768/{f1,f2,f3,f4}/`.
- The targeted forearm, aggressive reward/command, fine-tune and scratch
  regression suite passed **68 tests**. Python compilation, launcher
  `bash -n`, and `git diff --check` also passed.
- Formal logs confirmed progress through at least iterations F1=18564,
  F2=18561, F3=18538, F4=18535. Vertical episode rewards were nonzero in
  all four runs; slip penalties were nonzero in F3/F4 and intentionally
  zero-weighted in F1/F2. This verifies activation, not policy quality.
- Logs: `slurm_logs/getup-forearm-9768-train-f{1,2,3,4}.out` and `.err`;
  allocation log: `slurm_logs/t800_getup_forearm_finetune-9768.out`.
- An earlier submission, **9767**, failed during preflight because the
  finite-loss logging check lacked `import math`. The import was restored,
  a regression test added, and 9768 passed the complete preflight gate.
  Failed-job diagnostics are retained; it did not launch formal training.

Training is ongoing. No Play success or MuJoCo sim2sim improvement is claimed
for these new runs yet.
