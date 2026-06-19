# G1 1307 Isaac-Only Paper-Only Runbook

This runbook covers the local Isaac-only G1 PaperFastSAC paper-only path. It does not add MuJoCo evaluation,
real-robot evaluation, or deployment steps.

## Scope

- Motion asset: `data/g1/1307.npz`
- Recovery pool: `data/g1/grsi_states.pth`
- Contract source: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/paper_contract.py`
- Training entrypoint: `scripts/fast_sac/train.py`
- Eval entrypoint: `scripts/fast_sac/eval.py`

## Task IDs

- `Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0`
- `Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0`
- `Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0`

## Contract

The canonical `paper_equivalence_contract` is versioned in `paper_contract.py` and is written into:

- `logs/fast_sac/.../params/run.json`
- `logs/fast_sac/.../eval_summary.json`
- `logs/fast_sac/.../accepted_artifact.json` when local staged acceptance passes

Current paper-only posture:

- Isaac-only evaluation boundary is enforced.
- Only `arXiv:2602.13656` and its FastSAC reference `arXiv:2512.01996` are normative.
- The paper-only runtime uses a three-hidden-layer actor, a four-hidden-layer critic per Q-network, LayerNorm,
  observation normalization, mean twin-Q, and a twin C51 critic.
- `undesired_contacts` and `action_rate_before_stand` are explicit implemented paper-side reward terms.
- The active staged training profile is local 30k acceptance: each stage trains to 30,000 steps with a
  10,000-step warmup, so about 20,000 steps are update-eligible.
- No sim-to-real claim.

## GRSI Recovery Pool

Generate a schema-smoke file:

```bash
srun python scripts/g1/generate_grsi_states.py \
  --mode schema-smoke \
  --num_states 64 \
  --output data/g1/grsi_states_schema_smoke.pth
```

Generate the production Isaac recovery pool:

```bash
srun python scripts/g1/generate_grsi_states.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0 \
  --motion_file data/g1/1307.npz \
  --num_states 512 \
  --num_envs 32 \
  --max_attempts 256
```

Acceptance requirements:

- `version` is an exact integer and equals `GRSI_SCHEMA_VERSION`
- `source == "isaaclab-grsi:gravity-release"`
- `generation_config.mode == "gravity_release"`
- `generation_config.minimum_required_states == 512`
- `generation_config.accepted == validated accepted`
- `validated accepted >= 512`
- `validated rejected rows == 0`
- `generation_config.candidate_rejected` is present and non-negative
- `generation_config.candidate_rejection_reasons` is present for auditability
- `generation_config.friction_randomization` records sampled batch friction evidence
- `generation_config.rotational_recombination` records augmentation evidence
- `generation_config.source_contract_id` matches the canonical paper contract
- `contact_summary` is non-empty
- the file is not a trivial 2-state placeholder

## Stage Train / Resume / Eval

Run Isaac/GPU commands through `srun` in this environment.

The default staged profile is 30k local acceptance, not paper long-run parity. Slurm defaults are:

- `FINAL_STEP=30000`
- `MAX_STEPS=30000`
- `TRAIN_STEPS=30000`
- `CHECKPOINT_INTERVAL=5000`
- `LOGGER=wandb`
- `WANDB_MODE=offline`

Offline W&B data is written under each run directory and can be synced later with standard W&B tooling. The
accepted artifact remains a local file in this phase; `train.py` logs training curves and train/eval summary only.
For resumed Stage II/III runs, `TRAIN_STEPS=30000` means 30,000 new steps are collected after the resume checkpoint;
the final checkpoint therefore uses the cumulative step number, for example `model_60000.pt` for Stage II after
resuming from Stage I `model_30000.pt`.

### Stage I

```bash
srun python scripts/fast_sac/train.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0 \
  --motion_file data/g1/1307.npz \
  --num_envs 4096 \
  --logger wandb \
  --wandb_mode offline \
  --train_steps 30000 \
  --eval_episodes 16
```

Resume Stage I from the latest checkpoint:

```bash
srun python scripts/fast_sac/train.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0 \
  --motion_file data/g1/1307.npz \
  --num_envs 4096 \
  --logger wandb \
  --wandb_mode offline \
  --train_steps 30000 \
  --resume_checkpoint logs/fast_sac/g1_paper_fast_sac/<stage1-run>/model_<step>.pt
```

Evaluate a Stage I checkpoint without further training:

```bash
srun python scripts/fast_sac/eval.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0 \
  --motion_file data/g1/1307.npz \
  --checkpoint logs/fast_sac/g1_paper_fast_sac/<stage1-run>/model_<step>.pt \
  --episodes 32 \
  --num_envs 32
```

### Stage II

```bash
srun python scripts/fast_sac/train.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0 \
  --motion_file data/g1/1307.npz \
  --num_envs 4096 \
  --train_steps 30000 \
  --resume_checkpoint logs/fast_sac/g1_paper_fast_sac/<stage1-run>/model_<final-step>.pt \
  --eval_episodes 16
```

### Stage III

```bash
srun python scripts/fast_sac/train.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0 \
  --motion_file data/g1/1307.npz \
  --num_envs 4096 \
  --train_steps 30000 \
  --resume_checkpoint logs/fast_sac/g1_paper_fast_sac/<stage2-run>/model_<final-step>.pt \
  --eval_episodes 16
```

## Artifacts Per Stage

Each stage run under `logs/fast_sac/<experiment>/<run>/` should contain:

- `params/agent.json`
- `params/run.json`
- `metrics.json`
- `eval_summary.json`
- `model_<step>.pt`
- `training_curve.csv`
- `training_curve.jsonl`
- W&B offline run data when `LOGGER=wandb` and `WANDB_MODE=offline`
- `accepted_artifact.json` or `acceptance_blocker.json` after `scripts/fast_sac/accept_run.py`

`run.json` records the full paper and recipe contracts, seed, task, motion file, and resume source.
`eval_summary.json` records:

- `success_rate`
- `orientation_error_mean`
- `smoothness_mean`
- `episode_return_mean`
- `episode_length_mean`
- termination summary counts

Accept a 30k local staged run:

```bash
python scripts/fast_sac/accept_run.py \
  --stage stage-i \
  --log_dir logs/fast_sac/g1_paper_fast_sac/<stage-run> \
  --expected_task_id Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0 \
  --expected_contract_id <paper-contract-id> \
  --expected_recipe_contract_id <recipe-contract-id> \
  --expected_grsi_path data/g1/grsi_states.pth \
  --expected_grsi_hash <canonical-grsi-hash>
```

## Verification

Static verification:

```bash
pytest \
  tests/test_fast_sac_core.py \
  tests/test_fast_sac_acceptance.py \
  tests/test_fast_sac_wandb_logging.py \
  tests/test_g1_paper_full_config.py \
  tests/test_grsi_state_schema.py \
  tests/test_g1_kungfu_1307_config.py
```

Isaac smoke verification:

1. Generate the GRSI schema-smoke file.
2. Run a short Stage I train with `--max_steps` and `--eval_episodes`.
3. Confirm the run directory contains `metrics.json`, `eval_summary.json`, and a checkpoint.
4. Run `scripts/fast_sac/eval.py` directly against that checkpoint.

## Explicit Non-Goals

- No MuJoCo evaluation
- No real-robot evaluation
- No deployment script
