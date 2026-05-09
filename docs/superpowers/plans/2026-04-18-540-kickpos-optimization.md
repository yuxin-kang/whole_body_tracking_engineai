# 540 KickPos Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four `KickPos`-derived 540 experiment variants, register them, cover them with regression tests, and document the exact continuation commands used to launch the four follow-up runs.

**Architecture:** Keep the implementation inside the existing T800 tracking config surface. Extend `flat_env_cfg.py` with a tiny helper layer plus four `KickPos` subclasses, register them in `config/t800/__init__.py`, verify them with import-light pytest regression tests, and publish a run sheet in `exp/` so the four continuation jobs can be launched consistently from the current best `KickPos` checkpoint.

**Tech Stack:** Python, Isaac Lab `configclass`, `gymnasium` registry, pytest text/config regression tests, markdown experiment notes

---

## File Structure

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py`
  Adds late-phase constants, one new kick-leg phase helper, and four new `KickPos`-derived `EnvCfg` classes.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py`
  Registers the four new task ids so training and play scripts can resolve them from the gym registry.
- `tests/test_540_kickpos_followups.py`
  Lightweight regression coverage for helper constants, env config class definitions, registry entries, and the run sheet.
- `exp/4.18-kickpos-followups.md`
  Exact resume commands for the four continuation experiments, all based on the current best `KickPos` checkpoint.

### Task 1: Add Follow-Up Helper Tests and Helper Plumbing

**Files:**
- Create: `tests/test_540_kickpos_followups.py`
- Modify: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py`
- Test: `tests/test_540_kickpos_followups.py`

- [ ] **Step 1: Write the failing regression test file**

```python
from pathlib import Path


def test_540_kickpos_followup_phase_helpers_exist():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()

    assert "T800_540_KICK_PHASE_LATE = (0.4323, 0.5131)" in config
    assert "T800_540_RETRACT_PHASE_LATE = (0.5131, 0.5823)" in config
    assert "def _set_540_phase_sampling_focus(" in config
    assert "kick_phase: tuple[float, float] = T800_540_KICK_PHASE" in config
    assert "retract_phase: tuple[float, float] = T800_540_RETRACT_PHASE" in config
    assert "def _add_540_kick_leg_phase_position_reward(" in config
    assert 'reward_name: str = "kick_right_leg_phase_pos"' in config
    assert "body_names=T800_540_KICK_LEG_BODY_NAMES" in config
```

- [ ] **Step 2: Run the targeted helper test and confirm it fails**

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_phase_helpers_exist -v
```

Expected:

```text
FAIL ... assert "T800_540_KICK_PHASE_LATE = (0.4323, 0.5131)" in config
```

- [ ] **Step 3: Add the late-phase constants and helper plumbing in `flat_env_cfg.py`**

```python
T800_540_KICK_PHASE = (0.3923, 0.4731)
T800_540_RETRACT_PHASE = (0.4731, 0.5423)
T800_540_KICK_PHASE_LATE = (0.4323, 0.5131)
T800_540_RETRACT_PHASE_LATE = (0.5131, 0.5823)


def _set_540_phase_sampling_focus(
    env_cfg,
    kick_weight: float = 4.0,
    retract_weight: float = 2.5,
    kick_phase: tuple[float, float] = T800_540_KICK_PHASE,
    retract_phase: tuple[float, float] = T800_540_RETRACT_PHASE,
):
    env_cfg.commands.motion.phase_sampling_windows = [
        (kick_phase[0], kick_phase[1], kick_weight),
        (retract_phase[0], retract_phase[1], retract_weight),
    ]


def _add_540_kick_leg_phase_position_reward(
    env_cfg,
    weight: float,
    std: float,
    reward_name: str = "kick_right_leg_phase_pos",
    kick_phase: tuple[float, float] = T800_540_KICK_PHASE,
):
    setattr(
        env_cfg.rewards,
        reward_name,
        _make_phase_body_position_reward(
            weight=weight,
            std=std,
            body_names=T800_540_KICK_LEG_BODY_NAMES,
            phase_start=kick_phase[0],
            phase_end=kick_phase[1],
        ),
    )
```

- [ ] **Step 4: Re-run the targeted helper test and confirm it passes**

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_phase_helpers_exist -v
```

Expected:

```text
PASS tests/test_540_kickpos_followups.py::test_540_kickpos_followup_phase_helpers_exist
```

- [ ] **Step 5: Commit the helper layer and regression test skeleton**

```bash
git add tests/test_540_kickpos_followups.py source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py
git commit -m "Expose helper surfaces for 540 KickPos follow-ups" \
  -m "The follow-up experiments need one reusable kick-leg phase reward helper and one retimed phase window surface so later env variants can stay small and isolated." \
  -m "Constraint: Keep the implementation inside the existing T800 flat config file instead of introducing a new helper module
Rejected: Rework generic tracking reward plumbing | wider scope than this experiment lane needs
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep future 540 follow-up shaping helpers specific and local unless another motion family reuses them
Tested: pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_phase_helpers_exist -v
Not-tested: Task registration and run sheet are implemented in later tasks"
```

### Task 2: Add the Four `KickPos` Follow-Up Env Config Variants

**Files:**
- Modify: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py`
- Test: `tests/test_540_kickpos_followups.py`

- [ ] **Step 1: Run the env-config regression test and confirm it fails**

Append this test to `tests/test_540_kickpos_followups.py`:

```python
def test_540_kickpos_followup_env_cfgs_are_defined():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()

    leg_phase_block = config.split("class T800Flat540Huixuanti1KickPosLegPhaseEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):", 1)[1].split(
        "@configclass", 1
    )[0]
    retract_block = config.split(
        "class T800Flat540Huixuanti1KickPosRetractAssistEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):", 1
    )[1].split("@configclass", 1)[0]
    curriculum_block = config.split(
        "class T800Flat540Huixuanti1KickPosPhaseCurriculumEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):", 1
    )[1].split("@configclass", 1)[0]
    retimed_block = config.split(
        "class T800Flat540Huixuanti1KickPosRetimedPhaseCurriculumEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):", 1
    )[1].split("@configclass", 1)[0]

    assert "_add_540_kick_leg_phase_position_reward(self, weight=0.5, std=0.16)" in leg_phase_block
    assert "_add_540_retract_reward(self, weight=0.4, std=0.18)" in retract_block
    assert "_set_540_phase_sampling_focus(self, kick_weight=5.5, retract_weight=3.0)" in curriculum_block
    assert "self.commands.motion.reset_preroll_frames = 10" in curriculum_block
    assert "kick_phase=T800_540_KICK_PHASE_LATE" in retimed_block
    assert "retract_phase=T800_540_RETRACT_PHASE_LATE" in retimed_block
    assert "self.commands.motion.reset_preroll_frames = 10" in retimed_block
```

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_env_cfgs_are_defined -v
```

Expected:

```text
FAIL ... IndexError: list index out of range
```

- [ ] **Step 2: Add the four `KickPos`-derived env config classes**

```python
@configclass
class T800Flat540Huixuanti1KickPosLegPhaseEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_540_kick_leg_phase_position_reward(self, weight=0.5, std=0.16)


@configclass
class T800Flat540Huixuanti1KickPosRetractAssistEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_540_retract_reward(self, weight=0.4, std=0.18)


@configclass
class T800Flat540Huixuanti1KickPosPhaseCurriculumEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_540_phase_sampling_focus(self, kick_weight=5.5, retract_weight=3.0)
        self.commands.motion.reset_preroll_frames = 10


@configclass
class T800Flat540Huixuanti1KickPosRetimedPhaseCurriculumEnvCfg(T800Flat540Huixuanti1KickPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_540_phase_sampling_focus(
            self,
            kick_weight=5.5,
            retract_weight=3.0,
            kick_phase=T800_540_KICK_PHASE_LATE,
            retract_phase=T800_540_RETRACT_PHASE_LATE,
        )
        self.commands.motion.reset_preroll_frames = 10
```

- [ ] **Step 3: Keep the experiment ideas isolated**

```python
# Do not add kick velocity shaping in this round.
# Experiment meanings:
# - KickPosLegPhase: add kick-leg phase reward only
# - KickPosRetractAssist: add retract reward only
# - KickPosPhaseCurriculum: add sampling/preroll only
# - KickPosRetimedPhaseCurriculum: same as curriculum, but use later phase windows
```

- [ ] **Step 4: Re-run the env-config regression test and confirm it passes**

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_env_cfgs_are_defined -v
```

Expected:

```text
PASS tests/test_540_kickpos_followups.py::test_540_kickpos_followup_env_cfgs_are_defined
```

- [ ] **Step 5: Commit the new env config variants**

```bash
git add source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py tests/test_540_kickpos_followups.py
git commit -m "Add four focused KickPos follow-up env configs" \
  -m "Each new env config isolates one hypothesis: direct kick-leg phase shaping, light retract assistance, default phase curriculum, and retimed phase curriculum. This keeps the four experiment lanes comparable." \
  -m "Constraint: The user wants four non-baseline experiments, not a reproduction slot
Rejected: Combine reward and curriculum changes into one mega-config | would make the four runs uninterpretable
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep these follow-up configs inheriting from KickPos so the visual baseline stays fixed
Tested: pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_env_cfgs_are_defined -v
Not-tested: Gym registry wiring is added in the next task"
```

### Task 3: Register the New Task IDs

**Files:**
- Modify: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py`
- Test: `tests/test_540_kickpos_followups.py`

- [ ] **Step 1: Run the registry regression test and confirm it fails**

Append this test to `tests/test_540_kickpos_followups.py`:

```python
def test_540_kickpos_followup_tasks_are_registered():
    registry = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py").read_text()

    assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPosLegPhase-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosLegPhaseEnvCfg' in registry
    assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPosRetractAssist-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosRetractAssistEnvCfg' in registry
    assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPosPhaseCurriculum-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosPhaseCurriculumEnvCfg' in registry
    assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPosRetimedPhaseCurriculum-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosRetimedPhaseCurriculumEnvCfg' in registry
```

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_tasks_are_registered -v
```

Expected:

```text
FAIL ... assert 'id="Tracking-Flat-T800-540Huixuanti1-KickPosLegPhase-v0"' in registry
```

- [ ] **Step 2: Register the four gym task ids in `config/t800/__init__.py`**

```python
gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-KickPosLegPhase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosLegPhaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-KickPosRetractAssist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosRetractAssistEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-KickPosPhaseCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosPhaseCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-540Huixuanti1-KickPosRetimedPhaseCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1KickPosRetimedPhaseCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)
```

- [ ] **Step 3: Re-run the registry regression test and confirm it passes**

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_tasks_are_registered -v
```

Expected:

```text
PASS tests/test_540_kickpos_followups.py::test_540_kickpos_followup_tasks_are_registered
```

- [ ] **Step 4: Sanity-check the modified Python files compile**

Run:

```bash
python -m py_compile \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py \
  tests/test_540_kickpos_followups.py
```

Expected:

```text
# no output
```

- [ ] **Step 5: Commit the registry wiring**

```bash
git add source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py \
  tests/test_540_kickpos_followups.py
git commit -m "Register the 540 KickPos follow-up tasks" \
  -m "The four new env configs need stable gym ids so train/play workflows can launch them without editing scripts or Hydra configuration." \
  -m "Constraint: Launch surface must stay compatible with scripts/rsl_rl/train.py and scripts/rsl_rl/play.py
Rejected: Drive the follow-up experiments through ad-hoc config patching at runtime | harder to review and reproduce
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep task ids descriptive and aligned with the single hypothesis each experiment tests
Tested: pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_tasks_are_registered -v; python -m py_compile source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py tests/test_540_kickpos_followups.py
Not-tested: End-to-end training launch uses the final run sheet from the next task"
```

### Task 4: Add the Four-Run Continuation Sheet and Final Verification

**Files:**
- Create: `exp/4.18-kickpos-followups.md`
- Test: `tests/test_540_kickpos_followups.py`

- [ ] **Step 1: Run the run-sheet regression test and confirm it fails**

Append this test to `tests/test_540_kickpos_followups.py`:

```python
def test_540_kickpos_followup_run_sheet_lists_all_resume_commands():
    run_sheet = Path("exp/4.18-kickpos-followups.md").read_text()

    assert "--resume True" in run_sheet
    assert "--load_run 2026-04-16_20-56-57_540cut-kick-pos" in run_sheet
    assert "--checkpoint model_29999.pt" in run_sheet
    assert "--task=Tracking-Flat-T800-540Huixuanti1-KickPosLegPhase-v0" in run_sheet
    assert "--task=Tracking-Flat-T800-540Huixuanti1-KickPosRetractAssist-v0" in run_sheet
    assert "--task=Tracking-Flat-T800-540Huixuanti1-KickPosPhaseCurriculum-v0" in run_sheet
    assert "--task=Tracking-Flat-T800-540Huixuanti1-KickPosRetimedPhaseCurriculum-v0" in run_sheet
```

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_run_sheet_lists_all_resume_commands -v
```

Expected:

```text
FAIL ... FileNotFoundError: [Errno 2] No such file or directory: 'exp/4.18-kickpos-followups.md'
```

- [ ] **Step 2: Write the four continuation commands in `exp/4.18-kickpos-followups.md`**

````markdown
# 540 KickPos Follow-Up Runs

Base checkpoint:

- task: `Tracking-Flat-T800-540Huixuanti1-KickPos-v0`
- run: `2026-04-16_20-56-57_540cut-kick-pos`
- checkpoint: `model_29999.pt`
- motion file: `data/npz/540/cut/540huixuanti1.npz`

## Experiment A: KickPosLegPhase

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-540Huixuanti1-KickPosLegPhase-v0 \
  --motion_file data/npz/540/cut/540huixuanti1.npz \
  --resume True \
  --load_run 2026-04-16_20-56-57_540cut-kick-pos \
  --checkpoint model_29999.pt \
  --run_name 540-kickpos-leg-phase \
  --max_iterations 15000 \
  --num_envs 4096 \
  --headless
```

## Experiment B: KickPosRetractAssist

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-540Huixuanti1-KickPosRetractAssist-v0 \
  --motion_file data/npz/540/cut/540huixuanti1.npz \
  --resume True \
  --load_run 2026-04-16_20-56-57_540cut-kick-pos \
  --checkpoint model_29999.pt \
  --run_name 540-kickpos-retract-assist \
  --max_iterations 15000 \
  --num_envs 4096 \
  --headless
```

## Experiment C: KickPosPhaseCurriculum

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-540Huixuanti1-KickPosPhaseCurriculum-v0 \
  --motion_file data/npz/540/cut/540huixuanti1.npz \
  --resume True \
  --load_run 2026-04-16_20-56-57_540cut-kick-pos \
  --checkpoint model_29999.pt \
  --run_name 540-kickpos-phase-curriculum \
  --max_iterations 15000 \
  --num_envs 4096 \
  --headless
```

## Experiment D: KickPosRetimedPhaseCurriculum

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-540Huixuanti1-KickPosRetimedPhaseCurriculum-v0 \
  --motion_file data/npz/540/cut/540huixuanti1.npz \
  --resume True \
  --load_run 2026-04-16_20-56-57_540cut-kick-pos \
  --checkpoint model_29999.pt \
  --run_name 540-kickpos-retimed-phase-curriculum \
  --max_iterations 15000 \
  --num_envs 4096 \
  --headless
```
````

- [ ] **Step 3: Re-run the run-sheet regression test and then the full targeted suite**

Run:

```bash
pytest tests/test_540_kickpos_followups.py::test_540_kickpos_followup_run_sheet_lists_all_resume_commands -v
pytest tests/test_540_kickpos_followups.py \
  tests/test_540_kick_phase_landmarks.py \
  tests/test_540_preroll_reset.py \
  tests/test_540_support_com_reward.py \
  tests/test_zhiquan_config_tuning.py -v
```

Expected:

```text
PASS tests/test_540_kickpos_followups.py::test_540_kickpos_followup_run_sheet_lists_all_resume_commands
PASS tests/test_540_kickpos_followups.py
PASS tests/test_540_kick_phase_landmarks.py
PASS tests/test_540_preroll_reset.py
PASS tests/test_540_support_com_reward.py
PASS tests/test_zhiquan_config_tuning.py
```

- [ ] **Step 4: Spot-check the new task names and run sheet by grep**

Run:

```bash
rg -n "KickPos(LegPhase|RetractAssist|PhaseCurriculum|RetimedPhaseCurriculum)" \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800 \
  tests/test_540_kickpos_followups.py \
  exp/4.18-kickpos-followups.md
```

Expected:

```text
# matches in flat_env_cfg.py, __init__.py, tests/test_540_kickpos_followups.py, and exp/4.18-kickpos-followups.md
```

- [ ] **Step 5: Commit the run sheet and final verification state**

```bash
git add exp/4.18-kickpos-followups.md \
  tests/test_540_kickpos_followups.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py
git commit -m "Document and verify the 540 KickPos follow-up run matrix" \
  -m "The follow-up task ids are only useful if the operator can launch the four continuation runs consistently from the same KickPos checkpoint. The run sheet closes that loop and the targeted suite verifies the full surface." \
  -m "Constraint: Keep the operator workflow aligned with scripts/rsl_rl/train.py CLI flags and the local motion-file workflow documented in README.md
Rejected: Leave the four launch commands in chat only | not durable enough for repeated experiment runs
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Update this run sheet if checkpoint names, task ids, or screening budgets change
Tested: pytest tests/test_540_kickpos_followups.py tests/test_540_kick_phase_landmarks.py tests/test_540_preroll_reset.py tests/test_540_support_com_reward.py tests/test_zhiquan_config_tuning.py -v; rg -n \"KickPos(LegPhase|RetractAssist|PhaseCurriculum|RetimedPhaseCurriculum)\" source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800 tests/test_540_kickpos_followups.py exp/4.18-kickpos-followups.md
Not-tested: Live Isaac Sim training runs"
```
