# T800 起身 tracking 任务

推荐任务 ID：`Tracking-Flat-T800-GetUp-Smooth-v0`

旧版对照任务：`Tracking-Flat-T800-GetUp-v0`

任务针对躺倒后起身的完整轨迹，默认从第 0 帧开始播放到 clip 末尾。由于参考动作从地面姿态开始，配置关闭了普通 tracking 的 anchor、末端、身高和姿态摔倒终止，只保留 episode timeout；同时关闭 reset 时的位姿/速度/关节扰动和 interval push，先固定住 tracking 基线。

## 当前结论

旧版 `getup1_t800` 的 Slurm 9373 / `model_29999.pt` 已完成数值训练，但用户在客户端 Play 时反馈：“没学会跟踪，站起来那一下没有学会”。该 run 的训练曲线显示 episode 长度一直为 166 帧，末尾 body position error 约 `0.94 m`、joint position error 约 `2.17 rad`；因此它只能作为失败对照，不作为新的客户端策略。

推荐改用 `Tracking-Flat-T800-GetUp-Smooth-v0`：使用 379 帧、7.56 秒、连续性更好的 `faint_prone_getup_03_t800`，并在归一化相位 `0.24–0.76` 对关节、关键身体位置和 base 高度增加起身过渡 shaping；当前高度项覆盖 `0.20–0.90`、权重 `1.25`、`std=0.18`，垂直速度项权重 `0.35`、`std=0.75`，减少“直接跳起”的解。actor 输入维度不变，便于部署接口保持兼容。

## 1. 生成 tracking NPZ

源数据目录：

```text
/srv/shared/home/xjh/ws_kyx/legged_lab/source/legged_lab/legged_lab/data/MotionData/t800/amp/get_up/
```

从源目录的 joblib `.pkl` 生成完整 34-body、50 Hz NPZ：

```bash
cd /srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai

python scripts/convert_t800_getup_to_npz.py \
  --source-dir /srv/shared/home/xjh/ws_kyx/legged_lab/source/legged_lab/legged_lab/data/MotionData/t800/amp/get_up \
  --output-dir data/npz/t800_get_up \
  --urdf source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf \
  --fps 50 \
  --overwrite
```

如果当前 Python 没有 `joblib`/`scipy`，使用已经安装 legged_lab 依赖的解释器：

```bash
/srv/shared/home/xjh/app/miniconda3/envs/legged_lab_g1/bin/python scripts/convert_t800_getup_to_npz.py \
  --source-dir /srv/shared/home/xjh/ws_kyx/legged_lab/source/legged_lab/legged_lab/data/MotionData/t800/amp/get_up \
  --output-dir data/npz/t800_get_up \
  --urdf source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf \
  --fps 50 \
  --overwrite
```

输出文件：

| 文件 | 帧数 | 训练时长（50 Hz） |
| --- | ---: | ---: |
| `getup1_t800.npz` | 166 | 3.30 s |
| `getup4_t800.npz` | 333 | 6.64 s |
| `prone_getup_02_t800.npz` | 875 | 17.48 s |
| `faint_prone_getup_03_t800.npz` | 379 | 7.56 s |

转换器会用仓库内同一份 T800 URDF 复算 source key bodies，并把误差写入 metadata；`.pkl` 不能直接传给 tracking loader。

## 2. 训练

### 推荐：RTX4000 正式训练

脚本默认已经切换到平滑起身任务和数据：

```bash
cd /srv/shared/home/xjh/ws_kyx/whole_body_tracking_engineai
sbatch scripts/slurm/train_t800_getup_rtx4000.sbatch
```

提交前先做小规模 smoke：

```bash
sbatch --export=ALL,TASK=Tracking-Flat-T800-GetUp-Smooth-v0,MOTION_FILE=data/npz/t800_get_up/faint_prone_getup_03_t800.npz,NUM_ENVS=32,MAX_ITERATIONS=2,WANDB_MODE=offline,RUN_NAME=t800_getup_smooth_smoke \
  scripts/slurm/train_t800_getup_rtx4000.sbatch
```

### 旧版对照

旧版使用 `getup1_t800.npz`：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-GetUp-v0 \
  --num_envs 4096 \
  --run_name t800_getup1 \
  --headless \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
```

直接训练推荐任务时使用：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-GetUp-Smooth-v0 \
  --motion_file data/npz/t800_get_up/faint_prone_getup_03_t800.npz \
  --num_envs 4096 \
  --run_name t800_getup_smooth \
  --headless \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
```

训练其他起身 clip 时保持同一个 task，只替换 `--motion_file`：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-GetUp-v0 \
  --motion_file data/npz/t800_get_up/prone_getup_02_t800.npz \
  --num_envs 4096 \
  --run_name t800_prone_getup_02 \
  --headless \
  env.scene.terrain.visual_material=null \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
```

可用的 `--motion_file` 是：

```text
data/npz/t800_get_up/getup1_t800.npz
data/npz/t800_get_up/getup4_t800.npz
data/npz/t800_get_up/prone_getup_02_t800.npz
data/npz/t800_get_up/faint_prone_getup_03_t800.npz
```

## 3. 播放 checkpoint

推荐 checkpoint 的播放命令：

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-T800-GetUp-Smooth-v0 \
  --motion_file data/npz/t800_get_up/faint_prone_getup_03_t800.npz \
  --load_run <新的训练 run 目录名> \
  --checkpoint model_29999.pt \
  --num_envs 1 \
  --play_from_start true
```

客户端同步新的 checkpoint 后，任务名、motion 文件和 `--load_run` 必须使用这一组对应值；不能继续用旧的 `Tracking-Flat-T800-GetUp-v0` 配置播放新策略。

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-T800-GetUp-v0 \
  --motion_file data/npz/t800_get_up/getup1_t800.npz \
  --load_run <训练 run 目录名> \
  --num_envs 1 \
  --play_from_start true
```

验证转换结果（CPU-only）：

```bash
/srv/shared/home/xjh/app/miniconda3/envs/legged_lab_g1/bin/python \
  scripts/validate_t800_motion_fk.py data/npz/t800_get_up/*.npz
```
