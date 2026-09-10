# EXP-20260820：target 软靶击打训练变体

## 目标

在不改变原有纯 tracking 任务的前提下，为代表性动作增加独立的外部软靶训练环境。策略仍以动作参考跟踪为主，靶子只在击打窗口提供轻量接触奖励，episode 结束前不因水平位移或动作姿态瞬变提前重置；重置只保留超时和实际摔倒。

## 动作规格

所有帧编号从 0 开始，motion 采样率为 50 Hz。击打帧取末端第一次主导峰值附近的 5 帧，靶心放在峰值末端沿基座方向退 0.06 m。

| 任务 | 使用数据 | 击打末端 | 击打帧 | 靶心（motion/env 原点坐标） | 击打后恢复帧 |
| --- | --- | --- | --- | --- | --- |
| 左直拳 | `traj_eng_50hz_improved/straight_punch_L_terminal_hold_0p5s.npz` | `LINK_WRIST_END_L` | 9–13，峰值 11 | `(0.680, 0.105, 1.295)` | 14 起，剩余 71 帧 |
| 右直拳 | `traj_eng_50hz_improved/straight_punch_R_terminal_hold_0p5s.npz` | `LINK_WRIST_END_R` | 15–19，峰值 17 | `(0.810, -0.182, 1.297)` | 20 起，剩余 55 帧 |
| 左摆拳 | `traj_eng_50hz_improved/left_hook_001_terminal_hold_0p5s.npz` | `LINK_WRIST_END_L` | 34–38，峰值 36 | `(0.535, -0.372, 1.205)` | 39 起，剩余 116 帧 |
| 正蹬 | `traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz` | `LINK_ANKLE_ROLL_L` | 44–48，峰值 46 | `(1.821, 0.034, 1.135)` | 49 起，剩余 130 帧 |
| 回旋踢 | `traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz` | `LINK_ANKLE_ROLL_R` | 57–61，峰值 59 | `(1.330, -0.200, 1.570)` | 62 起，剩余 156 帧 |

## 靶子与奖励

- 靶子是每个动作独立朝向的 `0.18 × 1.40 × 1.60 m` 固定墙式 padded rigid proxy，不与地面共用硬接触参数；薄的法向厚度避免机器人在击打前被墙体过早拦截，宽高面用于覆盖末端轨迹误差。
- 靶子使用 `kinematic_enabled=True` 固定在世界坐标中，因此不会被踢走；它仍参与 PhysX 碰撞，动态机器人受到的接触冲量由腕端/脚踝 `ContactSensor` 的 `net_forces_w` 读取，反馈力不是手工伪造的。
- 靶材使用低摩擦、零反弹的 compliant contact：stiffness `3500 N/m`、damping `160 N·s/m`。`mass=12 kg` 和线/角阻尼保留作为刚体代理参数，但固定墙模式下不用于允许靶子移动。
- 五个墙面法向分别按动作末端的击打方向设置：左右直拳/正蹬沿 `+x`，左摆拳约 `-144°`，回旋踢约 `-76°`。
- 接触传感器挂在机器人已经启用 PhysX contact reporter 的对应腕端或脚踝末端，并只过滤对应靶子；接触力经 `tanh` 压缩并限制在击打窗口内，权重为 `0.25`。动作 tracking reward 保持主导。
- 击打窗口之后加入低权重恢复稳定奖励（基座高度、竖直姿态、角速度），权重为 `0.20`，帮助策略学会击打后的收腿/回位。
- actor 观测维度不变；靶心相对位置只加入 privileged critic，便于学习接触后的价值变化。

## 靶心二次校准与旧策略诊断

- 初版靶心按 link 原点估计，左直拳、左摆拳和回旋踢在可视化中出现明显偏置。二次校准改用 URDF 中腕端球碰撞体、踝端盒碰撞体的实际几何中心，再沿末端伸展方向向机器人内侧退 `0.06 m`；正蹬保持原位置，因为其可视化位置已经合理。
- 校准后的三次确定性旧 checkpoint 回放（关闭训练用材质、关节、质心和推力随机化）均显示 `target_displacement_max=0`，说明固定墙和接触反馈链路正常；旧策略的末端仍未完全到达参考击打点，不能据此把靶子再迁移到旧策略的错误轨迹上。
- 诊断结果：左直拳最近距离约 `0.17 m`；左摆拳在新靶心下轨迹发生明显偏离；回旋踢最近距离约 `0.90 m` 且无有效靶面接触。它们是旧纯 tracking 策略与参考末端的误差，后续应通过 target task 训练解决，而不是继续改变参考靶心。

## 训练边界

- 原有 `Tracking-Flat-T800-*`、`Baseline-*`、`Improved-*` 任务不改动。
- 新增五个 task ID：
  - `Tracking-Flat-T800-Target-StraightPunchL-v0`
  - `Tracking-Flat-T800-Target-StraightPunchR-v0`
  - `Tracking-Flat-T800-Target-LeftHook-v0`
  - `Tracking-Flat-T800-Target-LeftFrontKick-v0`
  - `Tracking-Flat-T800-Target-RoundhouseKick-v0`
- 首版靶训将动作起始 pose/velocity 固定为零扰动，使靶心和零相位 motion 严格对齐；确认训练有效后再增加相位感知的靶心随机化。
- 正蹬沿用完整 episode 且仅高度摔倒终止；其他三个沿用当前确认较好的改进数据和仅高度摔倒设置。

## 验证状态

- 已完成 Isaac Lab 2.1.0 对 `RigidObjectCfg`、compliant material、`ContactSensorCfg` 和 reset API 的源码核对。
- 已通过 Python 语法编译和 `git diff --check`。
- GPU smoke 使用 `scripts/slurm/smoke_target_pad.sbatch`，先验证 reset/step，再提交正式训练；脚本支持通过 `SMOKE_NUM_ENVS` 调整复制规模。
- Slurm job `8136` 在 `tr1` 的 RTX A4000 上顺序通过四个 task（array 0–3，均 `COMPLETED 0:0`）：每个任务完成场景构造、contact sensor 初始化、reset/step 和 1 次 PPO iteration；actor 输入保持 134 维，critic 输入为 287 维。
- Slurm job `8140` 在同一张 RTX A4000 上用 `SMOKE_NUM_ENVS=32` 顺序通过四个 task（均 `COMPLETED 0:0`），每个任务完成 768 个总采样步，确认目标和传感器可以在多环境复制下工作。
- Slurm job `8151` 在 `tr1` 的 RTX A4000 上重新通过四个固定墙 task（array 0–3，均 `COMPLETED 0:0`），确认运动学靶体、墙面尺寸/朝向、reset/step 和接触传感器初始化正常。
- Slurm jobs `8180`、`8181`、`8182` 在 `tr1` 的 RTX A4000 上完成左直拳、回旋踢、左摆拳的确定性诊断回放（均 `COMPLETED 0:0`）；固定墙位移均为 `0`。
- smoke 的 `DriverShaderCacheManager::init() called without a shutdown()` 是 Isaac Sim headless 退出时的既有插件告警，未导致任务失败；四个数组任务退出码均为 `0:0`。
- Slurm job `8208` 在 `tr1` 的 RTX A4000 上顺序通过五个 task（array 0–4，均 `COMPLETED 0:0`）；右直拳 target task 的 actor 输入为 134 维、critic 输入为 287 维，`target_contact`/`target_recovery` 均成功注册。
- Slurm job `8214` 用现有右直拳 tracking checkpoint `model_29999.pt` 做了 80 步 headless target Play（`COMPLETED 0:0`）：normalizer 兼容加载，固定靶位移为 `0`，旧策略最近腕端距离约 `0.1522 m`，峰值过滤接触力约 `741.82 N`。这证明现有策略可直接加载到 target task 做可视化，但尚未经过靶子专项 PPO 训练。
- `DriverShaderCacheManager::init() called without a shutdown()` 仍是 Isaac Sim headless 退出时的既有插件告警，未导致 8208/8214 失败。

## 正式靶训启动

- 训练脚本：`scripts/slurm/train_target_right_punch_front_kick_a4000_array.sbatch`。
- 右直拳：Slurm `8216_0`，`Tracking-Flat-T800-Target-StraightPunchR-v0`，W&B [run o8ipxb66](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/o8ipxb66)。
- 正蹬：首次与右直拳同时冷启动的数组元素 `8216_1` 在 Isaac Sim 场景创建阶段触发底层 allocator race；未改动右直拳任务，正蹬随后以独立任务 `8218_1` 重启并正常进入 PPO，W&B [run glupt6at](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/glupt6at)。
- 两项均使用 `4096` 个并行环境、`30000` 次迭代、RTX A4000、固定软靶、native motion episode；W&B 仅记录曲线和配置，checkpoint 保存在训练服务器本地。
- 正式 PPO 已完成：右直拳使用 Slurm `8216_0`，正蹬因首次冷启动 allocator race 改由 Slurm `8218_1` 独立重启；两项均训练至 `model_29999.pt`。训练末尾 mean episode length 分别为 `75/75` 与 `179/179`，checkpoint 及 W&B 证据见 `docs/experiments/2026-08-24.md`；后续 Play 结果也按该日期单独记录。
