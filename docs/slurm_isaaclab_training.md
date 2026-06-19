# Slurm IsaacLab Training Runbook

This note records the working flow for launching T800 whole-body tracking
training jobs on the `rtx3090` Slurm partition.

## Known Working Context

- Repo: `/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai`
- Conda env: `isaaclab_vid`
- Python used on shared node:
  `/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid/bin/python`
- IsaacLab on shared node:
  `/srv/shared/home/kyx/Workspace/IsaacLab`
- GPU partition: `rtx3090`
- One experiment should request one RTX 3090.
- Use `PYTHONUNBUFFERED=1` in Slurm jobs so `Learning iteration` appears in
  `slurm_logs/*.out` without waiting for stdout buffering.
- Interactive terminals may use `zsh`. The `sbatch --wrap` command below is
  written in plain shell/bash-compatible syntax because Slurm runs it as a
  non-interactive batch script. Do not rely on interactive `zsh` startup state
  inside the job.

## Cluster Guide Check Before Launching

Before starting the next training batch, quickly re-read the cluster usage
notes because the cluster policy or recommended commands may change:

```bash
sed -n '1,220p' /srv/shared/docs/CLUSTER_USER_QUICKSTART.md
sed -n '1,260p' /srv/shared/docs/CLUSTER_USER_GUIDE.md
```

Current interpretation from the May 9, 2026 run:

- Long GPU training should not run directly on the login node `venus`.
- GPU training should be submitted through Slurm with `srun` or `sbatch`.
- Stable multi-hour training should use `sbatch`; interactive `srun` is better
  for short debugging.
- A 1-GPU RTX 3090 experiment should use `--partition=rtx3090` or `dev`, not
  `rtx3090-large`.
- Each T800 experiment should request one RTX 3090:
  `--gres=gpu:rtx3090:1`.
- Keep explicit CPU, memory, time, stdout, and stderr settings.
- Clean temporary per-job cache directories after jobs finish if they are not
  removed automatically.

The four May 9 training jobs complied with this: they were submitted with
`sbatch`, used `rtx3090`, requested one 3090 each, used `8 CPU / 32G / 24h`,
wrote logs under `slurm_logs/`, and ran on compute node `epyc2` instead of the
login node.

TensorBoard is separate from training. It does not use GPU, but if it is left
running for a long time prefer binding it to `127.0.0.1` and viewing through
SSH forwarding. If temporarily using `--host 0.0.0.0` for browser access,
remember to stop it after use.

## One-off Checks

From repo root:

```bash
pwd
which python
python -c "import sys; print(sys.executable); print(sys.prefix)"
python -c "import isaacsim, gymnasium, os; print('isaacsim', os.path.dirname(isaacsim.__file__)); print('gymnasium', gymnasium.__version__)"
python -c "import isaaclab, os; print(os.path.dirname(isaaclab.__file__))"
git -C /srv/shared/home/kyx/Workspace/IsaacLab rev-parse HEAD
```

If running through Slurm, prefer setting `CONDA_PREFIX`, `PATH`, and
`PYTHONPATH` explicitly inside the batch command instead of relying on shell
startup files.

## Required Environment Variables

Use these in the Slurm job:

```bash
export PYTHONUNBUFFERED=1

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

export HOME=/tmp/wbt_omni_home_$SLURM_JOB_ID
export XDG_CACHE_HOME=$HOME/.cache
export XDG_DATA_HOME=$HOME/.local/share
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$HOME/Documents" "$HOME/.nvidia-omniverse/logs"

export CONDA_PREFIX=/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid
export PATH=/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid/bin:$PATH
export PYTHONPATH=/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_assets:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_rl:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_tasks:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_mimic:/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/source/whole_body_tracking
```

The isolated `HOME` avoids shared Omniverse/Kit cache and lock contention when
multiple IsaacSim jobs start on the same node.

## Training Command Template

Use these Hydra overrides on the shared IsaacLab setup:

```bash
python scripts/rsl_rl/train.py \
  --task <TASK_ID> \
  --motion_file <MOTION_NPZ> \
  --num_envs 4096 \
  --run_name <RUN_NAME> \
  --headless \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
```

Why these overrides matter:

- `env.scene.terrain.visual_material=null` avoids headless visual material
  setup problems on the shared IsaacLab / IsaacSim 5.1 stack.
- Do not set `env.scene.terrain.physics_material=null` for baseline-comparable
  runs. The local IsaacLab ground-plane spawner has been patched to create a
  fallback collision plane when the default ground USD exposes no collision
  child, so the default terrain physics material can be bound normally.
- `env.commands.motion.debug_vis=false` avoids remote/debug USD reference
  problems such as `frame_prim.usd`.
- `env.scene.contact_forces.debug_vis=false` keeps contact-force visualization
  off for headless training.

## sbatch Template

Change only `TASK`, `MOTION`, `RUN`, and `JOB` for new experiments.

```bash
mkdir -p slurm_logs

TASK="Tracking-Flat-T800-540Huixuanti1-v0"
MOTION="data/npz/540/cut/540huixuantitui1.npz"
RUN="t800-540huixuanti1-24h"
JOB="t800_540_huixuanti1"

sbatch --parsable \
  --job-name="$JOB" \
  --partition=rtx3090 \
  --gres=gpu:rtx3090:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=24:00:00 \
  --output="slurm_logs/${JOB}-%j.out" \
  --error="slurm_logs/${JOB}-%j.err" \
  --wrap="export PYTHONUNBUFFERED=1; \
export http_proxy=http://127.0.0.1:7890; export https_proxy=http://127.0.0.1:7890; export HTTP_PROXY=http://127.0.0.1:7890; export HTTPS_PROXY=http://127.0.0.1:7890; \
export HOME=/tmp/wbt_omni_home_\$SLURM_JOB_ID; export XDG_CACHE_HOME=\$HOME/.cache; export XDG_DATA_HOME=\$HOME/.local/share; export CUDA_CACHE_PATH=\$HOME/.nv/ComputeCache; mkdir -p \$XDG_CACHE_HOME \$XDG_DATA_HOME \$CUDA_CACHE_PATH \$HOME/Documents \$HOME/.nvidia-omniverse/logs; \
export CONDA_PREFIX=/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid; export PATH=/srv/shared/home/kyx/miniconda3/envs/isaaclab_vid/bin:\$PATH; \
export PYTHONPATH=/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_assets:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_rl:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_tasks:/srv/shared/home/kyx/Workspace/IsaacLab/source/isaaclab_mimic:/srv/shared/home/kyx/Workspace/whole_body_tracking_engineai/source/whole_body_tracking; \
python scripts/rsl_rl/train.py --task $TASK --motion_file $MOTION --num_envs 4096 --run_name $RUN --headless env.scene.terrain.visual_material=null env.commands.motion.debug_vis=false env.scene.contact_forces.debug_vis=false"
```

## Four T800 Jobs Used Successfully

```text
Tracking-Flat-T800-540Huixuanti1-v0
  motion: data/npz/540/cut/540huixuantitui1.npz
  run:    t800-540huixuanti1-24h

Tracking-Flat-T800-540Huixuanti1-KickVelKickPos-v0
  motion: data/npz/540/cut/540huixuantitui1.npz
  run:    t800-540huixuanti1-kick-vel-pos-24h

Tracking-Flat-T800-540Huixuanti1-KickJointLate-v0
  motion: data/npz/540/cut/540huixuantitui1.npz
  run:    t800-540huixuanti1-kick-joint-late-24h

Tracking-Flat-T800-Zhiquan-v0
  motion: data/npz/zhiquan/cut/zhiquan_quanji1.npz
  run:    t800-zhiquan-24h
```

## Monitor and Verify

Check Slurm state:

```bash
squeue -j <JOB_IDS> -o "%.18i %.35j %.9T %.10M %.25R %.8b"
sacct -j <JOB_IDS> --format=JobID,JobName%35,State,ExitCode,Elapsed,NodeList%20 --noheader
```

Check that training actually started:

```bash
rg -n "Completed setting up|Learning iteration|Actor MLP|Critic MLP|Error executing job|Traceback|Stage.GetPrimAtPath|No contact sensors|ValueError" slurm_logs/<JOB_LOG>.out slurm_logs/<JOB_LOG>.err
```

The strongest evidence that a run is healthy:

- `squeue` shows `RUNNING`.
- `scontrol show job <id>` shows `TimeLimit=1-00:00:00` and
  `TresPerNode=gres/gpu:rtx3090:1`.
- stdout contains `Completed setting up the environment...`.
- stdout contains `Learning iteration N/30000`.
- log dir under `logs/rsl_rl/t800_flat/<timestamp>_<run_name>/` contains
  `params/env.yaml`, `params/agent.yaml`, `model_0.pt`, and a growing
  `events.out.tfevents...` file.
- `params/env.yaml` confirms the exact `motion_file` and `num_envs: 4096`.

Example verification:

```bash
rg -n "motion_file:|num_envs:|run_name:|max_iterations:" \
  logs/rsl_rl/t800_flat/<timestamp>_<run_name>/params/env.yaml \
  logs/rsl_rl/t800_flat/<timestamp>_<run_name>/params/agent.yaml
```

## TensorBoard Remote Viewing

TensorBoard can be run on the server and viewed from a local browser.

Use the environment Python directly. The `tensorboard` script in this conda env
may have an old shebang pointing at `/home/kyx/miniconda3/...`, so this form is
more robust:

```bash
mkdir -p tensorboard_logs

setsid -f /srv/shared/home/kyx/miniconda3/envs/isaaclab_vid/bin/python \
  -m tensorboard.main \
  --logdir logs/rsl_rl/t800_flat \
  --host 0.0.0.0 \
  --port 16007 \
  > tensorboard_logs/$(date +%Y-%m-%d_%H%M)_tensorboard_all_t800_flat_16007.log 2>&1
```

Then open this from a local browser when the machine is reachable on the
cluster network:

```text
http://10.12.120.237:16007/
```

If direct access is blocked, use SSH port forwarding instead:

```bash
ssh -p 11222 -N -L 16007:127.0.0.1:16007 kyx@10.12.120.237
```

Then open:

```text
http://127.0.0.1:16007/
```

Useful checks on the server:

```bash
ss -ltnp | rg ':16007|tensorboard'
NO_PROXY=127.0.0.1,localhost,10.12.120.237 \
  curl -I http://10.12.120.237:16007/
ps -ef | rg '[p]ython -m tensorboard.main'
```

Use `--logdir logs/rsl_rl/t800_flat` when you want TensorBoard to show all
experiments under that directory with their real run directory names.
Use `--logdir_spec name:path,name2:path2` only when you intentionally want to
select a subset of runs and override their display names.

## Notes from the Debug Session

- The original fatal error was:
  `Stage.GetPrimAtPath(Stage, NoneType)` during ground-plane material setup.
  The current fix is in the local IsaacLab ground-plane spawner: it finds an
  existing collision child or creates `/World/ground/terrain/CollisionPlane`
  before binding the terrain physics material. Do not null the terrain physics
  material for baseline-comparable runs.
- The `RuntimeError: Accessed <pxr.Usd.Property object ...>` stderr message came
  from Isaac's ground-plane color `ChangePropertyCommand`; the local spawner now
  writes the USD shader attribute directly.
- Slurm stdout may buffer output. Use `PYTHONUNBUFFERED=1` so iteration logs are
  visible while the job is still running.
- Do not treat a quiet stdout as a hang by itself. Also check:
  `model_0.pt`, TensorBoard event growth, process/GPU usage, and latest
  `Learning iteration`.
- If many IsaacSim jobs start at once, isolated `HOME` is important to reduce
  Kit/Omniverse cache and lock contention.
- If a job needs internet access for registry/W&B/artifact operations, export
  the proxy variables inside the Slurm job. Do not rely on interactive shell
  state.
