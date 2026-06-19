# 论文对齐改造计划 — arXiv:2602.13656（Kung Fu Athlete Bot, FastSAC + G1 29-DoF）

> 目标：把本工程改成**和论文一致**，使训练有复现意义。本文件取代已删除的
> `fastsac_debug_plan.md` / `wave2_followups.md`（那是调参排查思路，已废）。
> 依据是论文逐维交叉核对得到的四层差异 + 一组疑似实现 bug（见 Part A）。
>
> **范式纠正（最重要）**：论文是**单阶段统一训练**——一个策略端到端，tracking 与 fall-recovery
> 在同一次训练里用 `z∼Bernoulli(p)` 混合（靠 LKES + GRSI + 双初始化），**没有 Stage I/II/III 课程**。
> 本工程现有的三阶段串链是自造的，要塌成单跑。
>
> **算法纠正**：论文明确"FastSAC averages outputs from both critics"，所以代码的 **mean-Q 是对的，
> 不要改 min-Q**。执行 session 之前在 `/tmp/wbt-minq-pkg` 做的 min-Q 变体对论文复现无意义，弃用。

---

## 执行者须知（HANDOFF）

- 本文件是**改造计划**，规划 session 产出，执行交由其它 session。审核人已确认"四层全要 + 修 bug"。
- 改完要能"和论文一样地单跑训练"，再验证。不是调参炼丹，是对齐论文。
- **已实证（来自之前预检，勿重复踩坑）**：
  - C51 `agent.` override 必须用**浮点字面量**（`agent.critic_v_min=-300.0`，写 `-300` 会因 configclass 类型检查报错）。
  - **真实折扣回报量级 ≈ -240**（v_min=-600 时 mean_q≈-240、target_q≈-244）→ 支撑要覆盖到这个量级。
  - 多 job 并发起 Isaac 会 USD 资产竞态（`No contact sensors / no rigid bodies`）→ **错峰启动 ~50s**；
    每 job 用隔离 omniverse HOME（见 `claude/run_dbg.sh`）。
  - zsh 无引号不分词 → 用 `${=VAR}` 或内联。
  - `rl/`、`data/g1/` 未被 git 跟踪 → 要隔离改动就**整包复制到 /tmp 前置 PYTHONPATH**，git worktree 不可行。
  - mean_q/target_q_mean 日志补丁**已落地主工作区**（`algorithm.py` `update()` + `train.py` CurveLogger），保留。
- **不许偏离**：保持 mean-Q（论文要求）；单阶段统一训练（不要再串链）；Part A 第 3 层"勿动"清单别去"修"。

---

## Part A — 差异总表（改造依据，按层）

### 第 1 层 · 高影响（必改，否则训练无意义）

| # | 项 | 论文 | 现状（file:line） | 改法 |
|---|---|---|---|---|
| A1 | 训练范式 | 单阶段统一，Bernoulli(p) 混合 tracking/recovery | 三阶段 30k 串链，阶段间改环境（`submit_g1_paper_fastsac_chain.sh`；`paper_full_env_cfg.py:281-301`） | 塌成单跑（见 Task A1） |
| A2 | 超参遵循 FastSAC 默认 | "Other hyperparameters follow the default settings in FastSAC" | 本地猜值（`config.py`），`recipe_contract.py:55` 自认 underspecified | 对齐参考默认（见 Task A2） |

A2 明细（你的值 → 目标）：

| 超参 | 现 | 目标（FastSAC/FastTD3 参考；以 repo `fast_td3/hyperparams.py` 为准确认） | file:line |
|---|---|---|---|
| batch_size | 256 | **≥8192（参考 32768）** | `config.py:18` |
| C51 v_min/v_max | [-50,50] | **覆盖实测回报：约 [-300, +50]**（实测 floor≈-240，留余量；或先测回报分布取 p1/p99） | `config.py:38-39` |
| C51 num_atoms | 51 | **101** | `config.py:37` |
| tau | 0.005 | **0.1** | `config.py:20` |
| updates/step (UTD) | 4 | **2** | `config.py:16` |
| warmup | 10000 | **~10（learning_starts 量级）** | `config.py:15` |
| replay_size | 1e6 固定 | num_envs × N（远小，按参考） | `config.py:17` |

> 注：mean-Q（`algorithm.py:113,141`）、actor 3 层 / critic 2×4 层（`config.py:29-30`）、auto-alpha、
> obs 归一化、layer norm —— **这些已对，别动**。

### 第 2 层 · 公式字面保真（建议改，不阻塞训练）

| # | 论文公式 | 现状（file:line） | 改法 |
|---|---|---|---|
| L2-3 | CoM 核 `exp(−‖·‖/σ²)`（线性范数） | `exp(−‖·‖²/σ²)` 平方 `rewards.py:200-221` | 指数里改用 `sqrt(error_sq)` |
| L2-4 | close_feet 阈值 **0.16 m** | 0.18 `paper_full_env_cfg.py:147` | 改 0.16 |
| L2-5 | feet_slip `Σ√‖v‖`（sqrt） | `Σ v²` `rewards.py:236-251` | 改 sqrt |
| L2-6 | LKES 能量 `E=Σ|q̇|`（绝对值） | `Σ q̇²` `lke.py:6-9` | 改 `torch.abs` |
| L2-7 | 终止位置/体位指标：全 3D L2 over 被跟踪体集 ℬ | 只看 z、只在 4 个末端体 `terminations.py:81-142` | 改为全 3D L2、覆盖 `G1_TRACKING_BODY_NAMES` |
| L2-8 | rollout push 间隔 5–10 s | 1–3 s `tracking_env_cfg.py:186-191` | 改 (5.0,10.0) |

### 第 3 层 · 论文未给数值（**勿动**，列出防止误"修"）

论文根本没给这些；现值合理且 `range_source:"local"` 已诚实标注。**不要改、不要当 bug**：
- 全部 DR 范围（mass 0.9-1.1、friction、CoM、PD 0.75-1.25、action delay 0-3 步、joint bias…）
- 奖励 σ（body-pos 0.3 / ori 0.4 / ang-vel 3.14 / σ_COM 0.1，源自 BeyondMimic）
- LKES α=0.2 / w_min=1 / w_max=8；双初始化 p=0.5；GRSI 512 状态
- 观测空间成员（policy/critic）；action scale 公式；50Hz；target_entropy=−0.5|A|

### 第 4 层 · 代码加了论文没有的（移除或设开关，做"论文忠实"跑时关掉）

| # | 项 | 现状 | 改法 |
|---|---|---|---|
| L4-a | rough terrain | Stage III `paper_rough_terrain=True` `paper_full_env_cfg.py:296` | 统一跑用 flat（`paper_rough_terrain=False`） |
| L4-b | torque disturbance | ±1 N·m `paper_full_env_cfg.py:209-218` | 移除该 event |

### 疑似实现 bug（论文没规定，但影响训练，独立修）

| # | bug | file:line | 改法 / 验证 |
|---|---|---|---|
| B1 | 手腕 4010 执行器刚度≈0.027、action scale≈46.6 rad（后两齿轮级转子惯量写成 0） | `g1.py:65` | 给后两级填合理转子惯量，或显式设 wrist 刚度/阻尼同量级，并把 `G1_ACTION_SCALE` 手腕封顶 ~0.5；验证：打印 `G1_ACTION_SCALE` 手腕应 0.2-0.5；`replay_npz.py` 看手腕不乱飞 |
| B2 | GRSI 摩擦随机化疑似只记 metadata、未写进 sim material | `generate_grsi_states.py:178-185` | 确认 friction 在 drop 前写入物理材质；否则随机化名存实亡 |
| B3 | LKES 权重 w_min=1 单调增不衰减 → 采样持续偏向失败锚点 | `commands.py:510` | 论文是 `clip(w+α, wmin, wmax)` 本就单调增，属设计；但确认 w_max=8 不会让分布塌到少数锚点，必要时加衰减（超出论文，慎） |
| B4 | `robot_init_states_8192.pth` 躺地坏池 | `flat_env_cfg.py:21` | **只影响非论文 StageX/PPO lane**；统一跑用 grsi 不受影响。做 PPO 对照时才需换成 grsi |

---

## Part B — 改造任务（按执行顺序，可勾选）

### Task A1 · 塌成单阶段统一训练
两种实现，选其一：
- **快路（最小改动）**：直接把 `Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0` 当成统一配置**单跑一次**
  （它已含全奖励 + 全 DR + GRSI + LKES + 双初始化），但先做 L4（关 terrain、去 torque）与 L2-7（终止改全3D L2）。
  **不走 `submit_g1_paper_fastsac_chain.sh`**。
- **正路（推荐）**：新增 `G1Flat1307PaperFastSACUnifiedEnvCfg`（在 `paper_full_env_cfg.py`）：
  以 `G1FlatStandingEnvCfg` 为基 + `_apply_paper_full_common()`（全奖励/DR/GRSI/LKES/双初始化）；
  `paper_rough_terrain=False`；删 `torque_disturbance`；终止用 L2-7 的全 3D L2；
  在 `config/g1/__init__.py` 注册任务 id `Tracking-Flat-G1-1307-PaperFastSAC-Unified-v0`。
验证：`gym.make` 能建，obs/action 维度与原 Stage 一致；单跑不串链。

### Task A2 · 对齐 FastSAC 默认超参
1. 读 `github.com/younggyoseo/FastTD3` 的 `fast_td3/hyperparams.py` 取**确切** FastSAC 默认，回填上表"目标"列。
2. 改 `config.py` / `fast_sac_cfg.py` 默认：batch_size、tau、updates_per_step、warmup、replay_size、num_atoms。
3. **价值支撑**：先用一次短跑测折扣回报分布（在 `algorithm.update` 临时打印 `backup.quantile([.01,.99])`），
   据此设 `v_min/v_max`（实测 floor≈-240 → 起点 [-300.0, 50.0]，浮点）。
4. 验证：`params/agent.json` 反映新值；短跑 `mean_q` 不贴任一边界。

### Task L2 · 公式保真（L2-3..L2-8 六条，按上表逐条改 + 单测/抽查）
- 改完各项后，写一个 reward/termination 抽查脚本：参考姿态零残差下打印各 reward 分量与终止指标，确认
  CoM/feet_slip 数值随公式改变、close_feet 在 0.16 触发、终止用全 3D。

### Task L4 · 移除论文外附加（terrain、torque disturbance）—— 已并入 Task A1 的统一配置

### Task B · 修 bug（B1 必修；B2/B3 跑前确认；B4 仅 PPO 对照时）

---

## Part C — 训练与验证计划（对齐后）

### C.1 单次统一训练（替代三阶段链）
一次训练就是论文范式。多卡用于**并行多 seed / 确认超参**，不是串阶段。
```bash
# 错峰启动多 seed（论文是单策略；多 seed 仅为复现稳健性与方差）
for SEED in 1 2 3; do
  $SRUN $PY scripts/fast_sac/train.py \
    --task Tracking-Flat-G1-1307-PaperFastSAC-Unified-v0 \
    --motion_file data/g1/1307.npz --num_envs 4096 \
    --max_steps <N> --train_steps <N> \
    --eval_episodes 32 --logger none --headless \
    --run_name paper-unified-seed$SEED --seed $SEED \
    agent.critic_v_min=-300.0 agent.critic_v_max=50.0 agent.critic_num_atoms=101 \
    agent.batch_size=8192 agent.tau=0.1 agent.updates_per_step=2 agent.warmup_steps=10
  sleep 50   # 错峰，避 USD 竞态
done
```
（环境前导 `$SRUN/$PY/PYTHONPATH/CONDA_PREFIX` 同 `claude/run_dbg.sh`；`agent.` 值全用浮点。）

### C.2 训练步数预算
论文未给总步数（A100-80G 大规模）。先定 `N`，按学习信号决定是否加长；**不要再用 30k×3 串链的语义**。
单跑总量应远大于旧的"20k 有效更新"（旧 batch 256 + warmup 1/3 是欠训根因）。

### C.3 成功判据（论文一致）
- eval `success_rate > 0`（joint<0.5rad、ori<0.8rad、不摔；`paper_contract.py`）。先求 >0 证明方向对，再求逼近论文。
- 训练曲线：`mean_q` 离开支撑下界、`reward_mean` 上升、`terminated_rate` 下降、`episode_length` 变长。
- 摔倒恢复：从 GRSI 初始化的 recovery envs 能站回参考。

---

## Part D — 执行清单（勾选）

- [ ] A1 塌成单阶段统一配置（新 Unified task 或 Stage-III 单跑 + L4），不串链
- [ ] A2 对齐 FastSAC 默认超参（先读 repo 取确切值；价值支撑按实测回报设）
- [ ] L2-3 CoM 线性范数  / L2-4 close_feet 0.16 / L2-5 feet_slip sqrt
- [ ] L2-6 LKES `Σ|q̇|` / L2-7 终止全 3D L2 over ℬ / L2-8 push 5–10s
- [ ] L4 关 rough terrain + 去 torque disturbance（并入 A1）
- [ ] B1 修手腕执行器（必）；B2 确认 GRSI 摩擦写入；B3 看 LKES 权重分布；B4 PPO 对照才换 init
- [ ] 第 3 层"勿动"清单：确认没人去"修"它们
- [ ] reward/termination 抽查脚本通过（公式生效）
- [ ] 单次统一训练跑通，`success_rate>0`，曲线学习信号正向
- [ ] 结果记 `claude/paper_unified_results.md` 回传规划 session

---

## 执行状态（执行 session 回填 2026-06-19）

代码改动全部落地并 `py_compile` 通过：
- **A2 超参**：`config.py` + `fast_sac_cfg.py`：batch 256→8192、tau 0.005→0.1、UTD 4→2、warmup 10000→10、
  atoms 51→101、v_min −50→−300（v_max 50 不变）。mean-Q 保留未动。
- **L2-3 CoM 线性范数**：`rewards.py:220-221` → `exp(-norm/sigma_com)`。
- **L2-4 close_feet 0.16**：`paper_full_env_cfg.py:147`。
- **L2-5 feet_slip sqrt**：`rewards.py:251` → 去 square，惩罚线性速度。
- **L2-6 LKES Σ|q̇|**：`lke.py:9` → `sum(abs(q̇))`。
- **L2-7 终止全 3D L2**：`tracking_env_cfg.py` base `anchor_pos`/`ee_body_pos` 改 3D 函数（thr 0.5）；
  `flat_env_cfg.py` 容忍式 `_make_tolerant_tracking_failure` 两项改 3D（项名保留）；
  Unified cfg 追加 `bad_motion_body_pos` 全 3D over `G1_TRACKING_BODY_NAMES`（thr 0.5）。
- **L2-8 push 5–10s**：`tracking_env_cfg.py:189`。
- **L4-a 关 rough terrain**：`paper_full_env_cfg.py` StageIII `paper_rough_terrain=False`；Unified 默认 False。
- **L4-b 去 torque disturbance**：`_apply_paper_full_randomization` 内置 None。
- **A1 单阶段统一**：新增 `G1Flat1307PaperFastSACUnifiedEnvCfg`（StageII 基 + 全 paper common + StageIII 重置事件 +
  base VELOCITY_RANGE + plane + 3D 体位终止），注册任务 `Tracking-Flat-G1-1307-PaperFastSAC-Unified-v0`。
- **B1 手腕**：`g1.py:65` `ROTOR_INERTIAS_4010=(0.068e-4,0.010e-4,0.020e-4)`。
  验证：旧尺度 46.56 rad → 新 0.247 rad（5020 同侪 0.363），stiffness 0.027→5.06。
- **B2/B3/B4**：B2/B3 为确认类（GRSI 摩擦元数据、LKES 权重），未盲改；B4 仅 PPO lane，FastSAC 用 grsi 不受影响。

已验证：
- [✅] Unified 任务预检（512 env/300 步/warmup 10，run `paper-preflight`）：env 正常构建（全 paper 奖励表打印正常）、
  无 traceback、rc=0；`agent.json` 确认 batch=8192/warmup=10/UTD=2/tau=0.1/atoms=101/v_min=-300/v_max=50；
  曲线含 `mean_q`，末值 ≈ −34（**不贴 −300 地板，落在支撑内**），terminated_rate≈0.07。

待训练（需用户确认是否开跑——属长/贵任务）：
- [ ] 短程 3000 步（4096 env）看学习信号（mean_q 抬升 / reward 升 / terminated 降）。
- [ ] 满预算单跑（≥60k，多 seed 错峰）验 success_rate>0。命令见 Part C.1（任务改 Unified-v0；
  超参已是新默认，agent. override 可省）。

### 复核纠正轮（2026-06-19，审核反馈 8 条，全部落地+预检2 通过）
预检2（1024 env/300 步，run `paper-preflight2`）rc=0、无 OOM；agent.json 确认全部新值；mean_q≈-11 不贴边界、
reward_mean≈-0.37（较预检1 -0.85 改善，因去掉两个 -10 惩罚+三个多余跟踪项）。
1. **close_feet 去平方**：`rewards.py:234` `square(clamp(...))` → `clamp(0.16-d, min=0)`（线性 hinge，论文式）。
2. **feet_slip 外层 sqrt**：`rewards.py` → `Σ sqrt(norm(v_xy))`（即 (vx²+vy²)^0.25），上一轮"去平方"不够。
3. **CoM 分母 σ²**：`rewards.py:222` `exp(-error/sigma_com)` → `exp(-error/sigma_com**2)`（与其余 exp 项一致）。
4. **清零 3 个非论文跟踪项**：`motion_global_anchor_pos`(0.5)/`motion_global_anchor_ori`(0.5)/`motion_body_lin_vel`(1.0)
   → 在 `_apply_paper_full_reward_weights` 置 0（论文 tracking 只有 body pos/ori/ang-vel/CoM）。
5. **清零非论文惩罚**：`self_collisions`(-10)/`electrical_power_cost`(-10) → 置 0（论文只留 undesired_contacts -0.5）。
6. **CoM 用全身质心**：`rewards.py:212` `root_com_pos_w`(pelvis 单体) → `Σ(mᵢ·body_com_posᵢ)/Σmᵢ`（`default_mass`×`body_com_pos_w`）。
7. **replay_size**：1e6 → **204800**（≈50 步×4096 env，FastTD3 短缓冲设计；buffer.capacity 是总 transition 数）。
8. **其余超参对齐参考**：batch 8192→**32768**；C51 支撑 -300/+50 → **对称 ±250**
   （兼修 v_max 过低——好策略跟踪回报可达 +数百，原 +50 会截顶）；网络宽度
   actor [512,256,128]→**[512,512,512]**、critic [512,256,256,128]→**[1024,1024,1024,1024]**（参考统一宽度，层数不变）。

> 仍存判断点（可后续调）：replay 按 50 步×4096env=204800 端口化（用户参考是 1024env→51200，按 env 数缩放）；
> v_max=250 若实测好策略回报超 250 需再放宽；终止 3D 阈值 0.5 若过严可调。

### 容忍门控 + 终止阈值纠正轮（2026-06-19，审核 A/B 两条，落地+预检3 通过 rc=0 无 traceback）
- **A. 恢复容忍触发键改逐帧 `I_recovering`**（`terminations.py` TolerantTermination.__call__）：
  原 `command.is_standing_task`（整局 Bernoulli 初始化分支）→ 改为逐帧肩高偏差
  `‖shoulder_z_ref − shoulder_z_robot‖ > recovering_shoulder_threshold(=1.0)`（与 before_stand 同信号）。
  效果：tracking 回合被推倒也获恢复容忍（fall-resilient），recovery 回合站起后失去容忍（严格跟踪）。
  `flat_env_cfg.py` 容忍式 cfg 显式加 `recovering_shoulder_threshold=1.0`。
- **B. Unified 终止对齐论文 3 指标**（`paper_full_env_cfg.py` Unified `__post_init__`）：
  `anchor_ori` 0.6→**0.8**(τ_ori)、`anchor_pos` 1.0→**0.5**(τ_pos)、**删 hip_dof**(非论文指标)、
  保留 body_pos 0.5 over ℬ(τ_body)。最终容忍项 = {anchor_pos0.5, anchor_ori0.8, body_pos0.5} 恰为论文 3 指标。
  预检3：rc=0、无 is_standing_task 报错、mean_q≈-2、reward≈-0.19、critic_loss 降至 1.53。

## 附 · 关键文件
- 超参/算法：`rl/fast_sac/config.py`、`fast_sac_cfg.py`、`algorithm.py`（mean-Q 保留）、`networks.py`
- 环境/奖励/终止：`tasks/tracking/tracking_env_cfg.py`、`mdp/rewards.py`、`mdp/terminations.py`、
  `config/g1/paper_full_env_cfg.py`、`config/g1/flat_env_cfg.py`
- 采样/初始化：`mdp/lke.py`、`config/g1/lke.py`、`config/g1/grsi.py`、`scripts/g1/generate_grsi_states.py`、`mdp/commands.py`
- 机器人：`robots/g1.py`（B1 手腕）
- 训练/启动：`scripts/fast_sac/train.py`（已含 mean_q 日志）、`claude/run_dbg.sh`（环境前导/错峰/隔离 HOME 模板）
- 既往实测：`claude/wave1_results.md`（回报≈-240、override 浮点、USD 竞态等坑）
