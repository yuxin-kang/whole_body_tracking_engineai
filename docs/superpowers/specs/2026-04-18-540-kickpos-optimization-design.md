# 540 KickPos Optimization Design

## Goal

Design four focused follow-up experiments for the 540 kick task, all starting from the current best visual policy family `Tracking-Flat-T800-540Huixuanti1-KickPos-v0`, to solve one concrete failure mode: the right thigh lifts and rotates correctly, but the right knee does not open and the lower leg does not fully kick out.

## Problem Statement

Current visual evaluation in [exp/4.17.md](../../../exp/4.17.md) shows:

- `KickPos-v0` is the current best result by motion fidelity.
- The policy already captures the spin, airborne phase, side body shape, and left-leg behavior reasonably well.
- The main remaining defect is distal right-leg extension during the kick window.

The current reward design explains this gap:

- `KickPos-v0` adds a right-foot position reward only.
- Existing phase rewards operate on body position or body velocity, not on joint-specific kick articulation.
- The policy can therefore satisfy much of the shaping signal by moving the hip and ankle endpoint without fully opening the knee.

## Success Criteria

Primary evaluation is video quality, not aggregate reward.

An experiment is considered better if it improves the following in order:

1. Right knee visibly opens during the kick window.
2. Right lower leg clearly extends outward instead of following the torso as a small jump.
3. Existing strengths of `KickPos-v0` remain intact:
   side-body posture, airborne quality, left-leg support shape, and spin continuity.

Scalar metrics are guardrails, not the ranking target:

- kick-specific reward should improve for the modified target
- `Episode_Reward/huixuanti_end_effector_pos` should not regress materially
- `Episode_Termination/time_out` should remain close to current `KickPos-v0`
- `Metrics/motion/error_body_pos` and `Metrics/motion/error_joint_pos` should not collapse

## Shared Experimental Rules

- All four experiments use `KickPos-v0` as the parent design.
- All four experiments resume from the current best `KickPos` checkpoint instead of training from scratch.
- All four experiments should first run as short continuation jobs of roughly `10k-15k` additional steps for screening.
- This round intentionally excludes `kick_vel` additions because they are more likely to reward a sharp swing or hop than true knee opening.
- Each experiment should modify one dominant idea only, so video differences remain interpretable.

## Experiment Matrix

### Experiment A: Kick-Leg Phase Position Reward

Hypothesis:
The current shaping target is too distal. Rewarding the kick leg chain inside the kick window will encourage true leg extension rather than only ankle placement.

Change:

- keep all `KickPos-v0` settings
- add a new phase-limited body position reward for:
  - `LINK_KNEE_PITCH_R`
  - `LINK_ANKLE_ROLL_R`
- apply it only inside `T800_540_KICK_PHASE`

Suggested starting range:

- `weight`: `0.4-0.6`
- `std`: `0.14-0.18`

Expected outcome:

- highest chance of directly improving visible knee opening
- moderate risk of over-constraining the kick if the weight is too high

### Experiment B: Light Retract Assistance

Hypothesis:
The policy may be avoiding full extension because it anticipates the need to recover balance and retract the leg. A light retract cue may allow a fuller kick without losing the later motion.

Change:

- keep all `KickPos-v0` settings
- add a light retract-phase reward on the right kick leg chain

Suggested starting range:

- `weight`: `0.3-0.5`
- `std`: `0.16-0.20`

Expected outcome:

- may improve overall kick completeness and motion continuity
- risk: can reinforce conservative behavior if it dominates the true extension target

### Experiment C: Phase Curriculum on Top of KickPos

Hypothesis:
The policy already knows the pre-kick motion, but training still spends too much effort on early spin and jump segments. Sampling and reset should focus more often on the kick window.

Change:

- keep all `KickPos-v0` settings
- increase kick-window and retract-window phase sampling weights
- add a small `reset_preroll_frames` so resets land shortly before the kick rather than far earlier in the clip

Suggested starting range:

- kick sampling weight: `5.0-6.0`
- retract sampling weight: `2.5-3.5`
- `reset_preroll_frames`: `8-12`

Expected outcome:

- should preserve the existing global motion style
- best candidate for a low-risk refinement pass over the current good policy

### Experiment D: Retimed Kick / Retract Windows

Hypothesis:
The current phase landmarks may not align with the moment where the knee should actually open. If the supervision window is early, the right target is being optimized at the wrong time.

Change:

- keep all `KickPos-v0` settings
- do not introduce a new reward family
- shift `T800_540_KICK_PHASE` later by roughly `0.03-0.05` in normalized phase
- shift `T800_540_RETRACT_PHASE` accordingly to preserve ordering

Expected outcome:

- if the current phase labels are off, this can fix timing without changing the reward shape
- highest uncertainty because it depends on the clip alignment being the true root cause

## Recommended Priority

Recommended ranking for expected value:

1. Experiment A
2. Experiment C
3. Experiment B
4. Experiment D

Reasoning:

- Experiment A most directly targets the observed failure mode.
- Experiment C is the safest way to refine the current best policy without changing the motion objective too much.
- Experiment B is plausible but less directly tied to the visual defect.
- Experiment D may be important, but only if the clip timing is indeed misaligned.

## Implementation Shape

To keep the diff small and reversible:

1. add new helper constructors in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py` only if existing phase body reward helpers are insufficient
2. add four new `EnvCfg` variants derived from `T800Flat540Huixuanti1KickPosEnvCfg`
3. register four new gym task ids in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py`
4. avoid editing generic tracking reward logic unless a helper is missing

## Risks

- Overweighting a kick-specific reward can destroy the currently good torso and support-leg motion.
- Retract shaping can accidentally reward caution instead of stronger extension.
- Phase curriculum can overfit the kick moment and weaken transitions if the window is too narrow.
- Retimed phase windows can silently make things worse if the current timing is already correct.

## Verification Plan

For each run:

1. export one comparable replay clip at the same checkpoint cadence
2. score the clip against the three primary visual criteria
3. record the following scalars at the chosen checkpoint:
   - kick-specific reward
   - `Episode_Reward/huixuanti_end_effector_pos`
   - `Episode_Termination/time_out`
   - `Metrics/motion/error_body_pos`
   - `Metrics/motion/error_joint_pos`
4. select the winner by visual quality first, scalar stability second

## Out of Scope

- adding generic velocity shaping to all experiments
- reworking the full reward stack
- changing the robot model or global tracking formulation
- launching a wide hyperparameter sweep in this round
