# T800 短 Episode 策略 Play 命令

以下六个策略使用动作原始时长训练，没有把轨迹扩展到 10 秒。Play 时使用训练对应的
`data/npz/traj_eng_50hz/` 数据。

## 1. 左直拳

```bash
python scripts/rsl_rl/play.py \
--task Tracking-Flat-T800-v0 \
--motion_file data/npz/traj_eng_50hz/straight_punch_L.npz \
--load_run 2026-08-10_13-15-26_straight_punch_L_base4090_j7236 \
--checkpoint model_27500.pt \
--num_envs 1 \
--play_from_start true
```

## 2. 右直拳

```bash
python scripts/rsl_rl/play.py \
--task Tracking-Flat-T800-v0 \
--motion_file data/npz/traj_eng_50hz/straight_punch_R.npz \
--load_run 2026-08-10_13-15-26_straight_punch_R_base4090_j7236 \
--checkpoint model_27000.pt \
--num_envs 1 \
--play_from_start true
```

## 3. 左摆拳

```bash
python scripts/rsl_rl/play.py \
--task Tracking-Flat-T800-v0 \
--motion_file data/npz/traj_eng_50hz/left_hook_001.npz \
--load_run 2026-08-10_13-15-26_left_hook_001_base4090_j7236 \
--checkpoint model_27500.pt \
--num_envs 1 \
--play_from_start true
```

## 4. 后手摆拳

```bash
python scripts/rsl_rl/play.py \
--task Tracking-Flat-T800-v0 \
--motion_file data/npz/traj_eng_50hz/rear_hook_001.npz \
--load_run 2026-08-10_13-15-26_rear_hook_001_base4090_j7236 \
--checkpoint model_27500.pt \
--num_envs 1 \
--play_from_start true
```

## 5. 正蹬

```bash
python scripts/rsl_rl/play.py \
--task Tracking-Flat-T800-v0 \
--motion_file data/npz/traj_eng_50hz/left_front_kick_002.npz \
--load_run 2026-08-10_13-15-26_left_front_kick_002_base4090_j7236 \
--checkpoint model_25500.pt \
--num_envs 1 \
--play_from_start true
```

## 6. 回旋踢

```bash
python scripts/rsl_rl/play.py \
--task Tracking-Flat-T800-v0 \
--motion_file data/npz/traj_eng_50hz/roundhouse_kick_001.npz \
--load_run 2026-08-10_13-15-26_roundhouse_kick_001_base4090_j7236 \
--checkpoint model_26500.pt \
--num_envs 1 \
--play_from_start true
```
