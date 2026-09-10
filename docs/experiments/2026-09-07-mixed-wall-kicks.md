# 2026-09-07：正蹬 / 回旋踢，50% 有墙 + 50% 无墙，从零训练

## 用户要求与当前状态

两项动作均加入固定软墙，一半训练环境有墙、一半无墙，机器人—地面静/动摩擦范围均为 1.0–1.7。两面墙都须在动作末端可踢到，避免提前或过度阻挡抬腿、横扫、收腿。使用 2×4090，每张卡运行一个独立任务。

Slurm **9770** 已在 **epyc4** 启动：两项 GPU preflight 均通过，两个 4096-env PPO 进程均持续迭代并保存 `model_0.pt`。当前是训练启动记录，尚无新策略的用户 Play / sim2sim 成功结果。

## 墙位的计算依据

采用当前 `T800_CFG` 实际加载的 `serial_t800.urdf`，不是可选的 MuJoCo 碰撞配置。两只踝部碰撞盒尺寸均为 `(0.26, 0.10, 0.02)` m；局部偏移分别为 `(0.02, ±0.00166, -0.054)` m。逐帧用参考 link 姿态把碰撞盒八角点转换到世界坐标，并区分墙中心和迎击面（墙厚 0.18 m，迎击面比中心靠近机器人 0.09 m）。

| 项目 | 正蹬 | 回旋踢 |
| --- | --- | --- |
| 旧墙中心 xyz，m | `(1.821, 0.034, 1.135)` | `(1.330, -0.200, 1.570)` |
| 新墙中心 xyz，m | `(2.1500544, 0.034, 1.135)` | `(1.5392255, -0.500, 1.570)` |
| 新墙 yaw | `0` | `0`（旧值 `-1.33`） |
| 新迎击面 x，m | `2.0600544` | `1.4492255` |
| 参考脚碰撞体最大 x，m | `2.08505438`，第 48 帧 | `1.47422551`，第 61 帧 |
| 新墙最大参考压入深度 | `0.025 m` | `0.025 m` |
| 全动作几何相交帧（0 起始） | `48–49` | `61–63` |
| 接触奖励窗口 | `47–50` | `60–64` |
| 靶训恢复奖励开始帧 | `51` | `65` |

旧正蹬迎击面 x=1.731，参考脚最大会越过该面约 0.354 m，因此即使当时观看感觉距离合适，也与后续动作有明显冲突。旧回旋踢倾斜墙截断横扫路径，后续脚位置会越过它的无限延伸迎击面约 1.078 m；这是轨迹越面距离，不是脚持续嵌在有限厚墙体中的深度。

新位置对全部原始 URDF 碰撞体做保守包围盒检查（球/圆柱用外包盒，补齐参考中省略的固定子 link）：两项都只有击打脚在上述几帧相交，没有其他身体、手臂、支撑脚的候选墙碰撞。墙仍为固定 compliant pad：尺寸 `(0.18, 1.40, 1.60)` m，材质静/动摩擦 `0.30/0.22`，恢复系数 `0`，刚度 `3500 N/m`，阻尼 `160 N·s/m`。参考的 2.5 cm 几何重叠不代表学习策略的实际最大接触形变或力。

## 训练配置

- Task：`Tracking-Flat-T800-Target-MixedWall-LeftFrontKick-v0` / `Tracking-Flat-T800-Target-MixedWall-RoundhouseKick-v0`。
- 各 4096 个环境；偶数 env ID 有墙（2048），奇数无墙（2048）。类别固定，不随局部重置抽样，避免两类失败率不同造成数量漂移；单环境 Play 默认有墙，两个环境可同时查看两类。
- 无墙环境将墙停放到本环境地面下 z=-10 m，击墙奖励强制为 0，critic 的 target position 置 0；actor ABI 保持 134 维，critic 287 维。每个环境均保留动作跟踪和恢复目标。
- 地面平面材质静/动摩擦保持 `1.0/1.0`、`multiply`，通过机器人刚体材质的 startup 随机化实现各环境不同的有效地面接触摩擦。两项范围均 `1.0–1.7`，64 个材质桶，`make_consistent=True`（动摩擦不超过静摩擦），材质恢复系数随机范围 `0–0.1`。这不是每个 episode 改写共享地面材质。
- 从零训练：`resume=false`，无 checkpoint 加载；30000 iterations，save interval 500，24 steps/env，PPO adaptive LR 起始 `1e-3`，网络 `[512,256,128]`，种子 42。
- 正蹬沿用 Target tracking 奖励；回旋踢加入既有 StableRecovery 的落脚/恢复/终段站立奖励，包括恢复脚滑 `-2`、支撑 COM `1.0`、global anchor pos/ori `1.0`，终段附加项沿用已有 helper。
- 使用 native clip，从起点播放参考，末帧 hold，真实高度跌倒和 episode 超时终止。
- Slurm `rtx4090`，account `users`，显式 QoS `csc101_gpu_unlimited`，总 2×`rtx4090d`、16 CPU、48 GB RAM。独立 exclusive steps `9770.2` / `9770.3` 各占 1 GPU、8 CPU、24 GB RAM。
- 每个进程独立 USD/缓存临时目录，不修改 HOME；W&B 记录曲线/配置，权重保存在服务器本地。

## 验证证据

- CPU 测试：`tests/test_mixed_wall.py`、`tests/test_t800_improved_actions.py`、`tests/test_t800_roundhouse_recovery.py` 共 **18 passed**。包含非零场景原点、非对称子集反复重置仍保持半数、隐藏墙观测及奖励隔离。
- 全参考几何审计：`scripts/diagnostics/analyze_mixed_wall_geometry.py`，输出 `logs/preflight/t800_mixedwall50_mu1_1p7_scratch_4090_j9770/geometry.json`，记录参考、URDF、配置 SHA256。
- Isaac GPU preflight：各 32 个环境、16/16 分组，目标实际位置/姿态、重置、观测维度、材质读回均通过；连续运行超过一个原生 episode 长度且观测/奖励有限值。证据分别为同目录 `left_front_kick.json` / `roundhouse_kick.json`。
- 实际读回材质范围：静摩擦约 `[1.01134,1.68679]`，动摩擦约 `[1.01134,1.63256]`，所有采样均满足动≤静。
- 参考姿态短时物理探测中两面墙均可接触，正蹬窗口记录非零接触，回旋踢窗口记录非零接触；无墙组过滤墙接触力始终为 0。此探测会设置参考姿态，不等于新策略已经学会踢墙，也不作为真实击打力测量。
- 两项正式运行保存的 `params/env.yaml` / `params/agent.yaml` 与新任务、墙参数、摩擦、4096 环境、从零训练设置一致。
- preflight 关闭时出现既有 `DriverShaderCacheManager::init() called without a shutdown()` 告警；两项显式 passed marker 均生成，随后 PPO 正常启动。

## Run / 曲线 / 复现

正蹬 run：`2026-09-07_17-32-14_t800_left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770`。

[正蹬 W&B](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/t800_left_front_kick_mixedwall50_mu1_1p7_scratch_4090_j9770)

回旋踢 run：`2026-09-07_17-32-14_t800_roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770`。

[回旋踢 W&B](https://wandb.ai/gkmbgm-northeastern-university/urkl-t800/runs/t800_roundhouse_kick_mixedwall50_mu1_1p7_scratch_4090_j9770)

两项输出均位于 `logs/rsl_rl/t800_flat/<run>/`；完整日志 `slurm_logs/t800_mixedwall_2x4090-9770-{left_front_kick,roundhouse_kick}.runtime.log`。

提交脚本：`scripts/slurm/train_t800_mixed_wall_kicks_2x4090.sbatch`，内置两项顺序 preflight，然后双 GPU 并行 PPO；结束时检查正式最终 checkpoint `model_29999.pt`，避免仅凭 Isaac 退出码判定训练完成。

后续应分别在有墙、无墙环境进行完整起点 Play，检查是否击中、收腿和站立，不能仅凭平均 target_contact 奖励判定成功（该均值含 50% 无墙环境）。MuJoCo 还需配置同样的墙和地面条件才能做有意义的对应比较。
