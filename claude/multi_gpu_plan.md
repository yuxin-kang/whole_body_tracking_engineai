# 多卡并行 FastSAC 训练 — 实施计划（单节点 8×RTX3090，一个策略）

> 目标：把单卡 `scripts/fast_sac/train.py` 改成**真正的多卡数据并行**——8 个进程各驱动一张卡、
> 各跑 ~1024 环境（有效 8192 环境）、梯度 NCCL all-reduce 同步，**训练同一个策略**。
> 不是 8 个独立 job。配套论文对齐计划 `claude/paper_alignment_plan.md`（超参已对齐：batch/UTD/支撑）。
>
> 结论先行：**采用同步数据并行 + 手动 NCCL all-reduce**（照搬 IsaacLab/rsl_rl 既有范式），
> 用 `torchrun --standalone --nproc_per_node=8` 启动，每进程一卡一 SimulationApp。
> **不用 torch DDP 包装器**（FastSAC 每步 3 次 backward + target 网 + 裸 log_alpha，DDP 不适配）。

---

## ⚠️ 审查修订（v2，子代理对抗审查后采纳，**这些覆盖下文相应处**）

架构（手动 all-reduce、无 DDP、critic→actor 顺序、target 免同步、alpha reduce、gate 不死锁）经审查**确认正确**。
但以下 9 处必须按此修订，否则会静默污染训练或 hang：

1. **[严重] normalizer 改用"只在 observe() 更新"方案**：现状 normalizer 每 env step 被更新 **5 次**
   （`observe()` 1 次 + `update()` 内 4 次 × `updates_per_step`，`algorithm.py:105-108`），"每 100 次同步"会让各 rank 在
   **不一致的归一化输入**上算梯度再 all-reduce → 静默发散。**改法**：把 `update()` 内 4 处 `update_stats=True` 改成 `False`，
   normalizer 只在 `observe()`（`train.py:374`，rollout 数据）更新——这也更符合标准 SAC——然后**每 env step 同步一次**
   normalizer（2 个小 all-reduce，廉价）。
2. **[严重] gate 与 sample 都要用 per_rank_batch**：`train.py:395` 的 `replay_buffer.size >= agent_cfg.batch_size`
   和 `:397` 的 `sample(agent_cfg.batch_size)` **两处**都改 `per_rank_batch`，否则填充阈值比采样 batch 大 8×、UTD 漂移。
3. **[严重] K-shard 方差合并用精确公式**：`M2_total = ΣM2_i + Σ count_i·(mean_i−mean_global)²`；
   丢弃"AVG 近似"（漏了 mean 间方差项 → 低估方差 → 过归一化）；处理 `count` 初值=1.0（`normalization.py:14`，
   合并后 count 设为**全局值**，勿跨同步重复累加）；加单元测试：8 shard 合并 == 单进程对拼接数据。
4. **[严重→修] scope/顺序**：`AppLauncher` 在**模块层**构造，`agent_cfg/env_cfg` 只在 `main()` 内（`@hydra_task_config`，`train.py:271`）。
   → 模块层只做 `set_device(local_rank)` + 读 ranks；`agent_cfg.device=...`、`init_process_group`、broadcast 全部放 **`main()` 内**、
   在第一次 collective 之前。
5. **[严重→修] 单卡回退别崩**：`app_launcher.local_rank` 仅在 `--distributed` 时才赋值（`app_launcher.py:670`），
   非分布式访问会 AttributeError。→ `local_rank = getattr(app_launcher, "local_rank", 0)`，整块 dist 逻辑 `if is_distributed` 门控。
6. **[高] 启动断言防死锁**：启动时 all-reduce `num_envs` 与 `replay_capacity`，断言 `min==max` 否则 fail-fast；
   `assert batch_size % world_size == 0` 和 `replay_size % world_size == 0`。前 N 步的参数/ size 校验和断言**设为必跑**（之后关）。
7. **[中] eval/收尾要两道 barrier**：训练循环结束 → `barrier()`（全 rank 汇合）→ rank0 存盘+eval+日志 → `barrier()` → `destroy_process_group()` → `close()`。
   单一退出 barrier 不够（其它 rank 会在 rank0 eval 期间冲进 teardown 拆 NCCL）。eval 本身只调 `act()` 无 collective，不会死锁。
8. **[中] resume 要 broadcast checkpoint 真正的 target_critic**（`algorithm.py:219` 存了独立 target），
   勿用"从 critic 重建 target"（会丢 soft-update 滞后量、造成重启不连续）。Adam optimizer state 也 broadcast（一次性，慢但可接受）。
9. **[确认正确] 无需处理**：梯度平均对 SAC 等价（critic/actor loss 是 mean）、critic.step 先于 actor 前向、
   target 稳态免同步（NCCL all-reduce 跨 rank 逐位一致 → critic 同步 → target 同步）、alpha reduce、
   gate 不死锁（每 rank 每步恒加 `num_envs` 行，`size` 逐步一致）；代码无 grad clip / 无 amp，无需处理。
   保留周期参数校验和断言作为"漏 reduce"的唯一守卫。

---

## 0. 架构决策与理由

| 决策 | 选择 | 理由 |
|---|---|---|
| 并行范式 | 同步数据并行（每卡独立 sim+replay，梯度 all-reduce 平均，参数全程一致） | 算法代码几乎不动（3 优化器仍是 3 优化器）；风险最低；与 IsaacLab 已验证路径一致 |
| 梯度同步 | **手动 all-reduce**（照抄 rsl_rl `reduce_parameters`），每个优化器 step 前一次 | DDP 假设每图一次 backward；FastSAC 有 3 次 backward + actor 复用 critic + log_alpha 是裸 leaf，DDP 不适配（需 find_unused_parameters/no_sync 等坑） |
| 启动 | `torchrun --standalone --nnodes=1 --nproc_per_node=8`，外层 `srun -N1 -n1` | torchrun 自动设 MASTER_ADDR/PORT/RANK/LOCAL_RANK/WORLD_SIZE，正是 AppLauncher 与 init_process_group 读取的；IsaacLab 官方多卡路径 |
| 设备/rank | IsaacLab `AppLauncher(--distributed)` 按 LOCAL_RANK 绑 `cuda:{local_rank}` | `IsaacLab/.../app/app_launcher.py:668-690` 已实现：读 LOCAL_RANK/RANK，设 device_id、physics_gpu、限 CPU 线程 |
| init_process_group | 训练脚本自己调（backend=nccl），照 rsl_rl | `rsl_rl/runners/on_policy_runner.py:351-393`：init_process_group + `torch.cuda.set_device(local_rank)` |
| 参数初值一致 | 启动时从 rank0 broadcast 全部网络/log_alpha/normalizer | 每 rank 种子不同（环境多样性），不能靠种子保证初值一致，必须 broadcast |
| target 网 | **不做任何同步** | soft update 是 (target,critic,tau) 的确定函数；只要 critic 同步，target 自动同步。多同步反而双重计数 |
| batch 语义 | config `batch_size` 视为**有效值**，每 rank 取 `batch_size // world_size` | 保持有效梯度 batch 与单卡基线一致；SAC 对 batch/UTD 比敏感，不能悄悄 ×8 |
| obs normalizer | 启动 broadcast + 每 N 次更新做精确 K-shard 合并 | 各 rank 见不同数据 → 统计漂移；归一化在产生梯度的前向里（`algorithm.py:105-108`），漂移会致静默参数发散 |
| replay buffer | 每 rank 独立 CPU buffer，按时间horizon定容量 | 不共享（共享=被否决的 Ape-X 路径）；聚合 replay≈各 rank 之和 |

**有效配方**：8×1024=8192 环境、每 env step 收 8192 新样本、每 step 做 `updates_per_step` 次同步更新、
每次有效 batch = `per_rank_batch × 8` = config `batch_size`。等价于"单卡 8192 环境 + config batch"的论文级配方。

---

## 1. IsaacLab 分布式机制（参考实现，照抄）

- **设备/rank 解析**：`IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py:668-690`
  读 `LOCAL_RANK`/`RANK`，设 `device="cuda:{local_rank}"`、`physics_gpu=active_gpu=local_rank`，
  并按 `os.cpu_count()//WORLD_SIZE` 限制 PhysX/BLAS 线程。`--distributed` 由**训练脚本自己的 argparse** 加，AppLauncher 只读取。
- **进程组初始化**：`rsl_rl/runners/on_policy_runner.py:351-393`
  `init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)` + `torch.cuda.set_device(LOCAL_RANK)`。
- **初值 broadcast**：`rsl_rl/algorithms/ppo.py:419-430`（`broadcast_object_list([state_dict], src=0)` 后 load）。
- **梯度 all-reduce**：`rsl_rl/algorithms/ppo.py:432-460`
  （flatten 所有 `param.grad` → `all_reduce(SUM)` → `/= world_size` → 切片写回），在 `backward()` 后 `step()` 前调。
- **启动**：`IsaacLab/docs/.../multi_gpu.rst:106` `python -m torch.distributed.run --nnodes=1 --nproc_per_node=N train.py ... --distributed`。
- **日志门控**：`on_policy_runner.py:52` `disable_logs = is_distributed and global_rank != 0`。

---

## 2. 代码改动清单（逐文件）

### 2.1 `scripts/fast_sac/train.py`
1. **加 `--distributed`** 到 `parse_args`（~line 46），使 AppLauncher 能读到。
2. **进程组与设备/种子**（`app_launcher = AppLauncher(args_cli)` 之后，~line 53）：
   - 读 `world_size=int(os.getenv("WORLD_SIZE","1"))`、`global_rank=int(os.getenv("RANK","0"))`、`local_rank=app_launcher.local_rank`。
   - `is_distributed = world_size > 1`。
   - 若分布式：`agent_cfg.device = env_cfg.sim.device = f"cuda:{local_rank}"`；`agent_cfg.seed += local_rank`（环境多样性）；`env_cfg.seed = agent_cfg.seed`。
   - `torch.cuda.set_device(local_rank)`；`init_process_group(backend="nccl", rank=global_rank, world_size=world_size)`
     **在 SimulationApp 起来之后、第一次 collective 之前**调（CUDA context 已在正确设备上）。
3. **batch/replay 切分**：`per_rank_batch = agent_cfg.batch_size // world_size`（**gate `:395` 与 sample `:397` 两处都用**，见修订#2）；
   `per_rank_replay = agent_cfg.replay_size // world_size`（默认保聚合不变；见 §3 horizon 说明）。
   **注意**：每 rank 的 `--num_envs` 是单卡环境数（如 1024），不是总数。
4. **I/O 门控到 rank0**：`_make_log_dir`(95)、`CurveLogger`(138)、wandb init/log(300-321,420)、
   `_save_checkpoint`(104)、`evaluate_policy`(458) 全部 `if global_rank==0`。非 rank0 不建目录/不起 wandb。
5. **更新分支保持确定一致**（避免死锁）：更新条件用**全局 step + 固定 warmup**判定，不要用可能不一致的浮点。
   各 rank 每步加同样 `per_rank_num_envs` 到 buffer，故 `replay_buffer.size` 在各 rank**逐步相同**，
   `if size>=per_rank_batch and step>warmup` 的分支天然一致 → **无需每步 collective**。
   （debug 期可选：每 K 步 all-reduce 一个 `size` 校验断言一致。）
6. **barrier 与收尾**：启动 broadcast 后 `barrier()`；rank0 存 checkpoint 前 `barrier()`；
   退出前 `barrier()` → `destroy_process_group()` → 再 `env.close()`/`simulation_app.close()`（`train.py:486`），
   避免一 rank 拆 NCCL 时另一 rank 还在 collective。
7. **resume**：仅 rank0 读 `.pt`，然后 broadcast 网络/optimizer/normalizer 到各 rank（见 2.2 broadcast）。

### 2.2 `source/.../rl/fast_sac/algorithm.py`
1. **构造接受分布式信息**：`__init__(... , world_size=1, is_distributed=False)`，存 `self.world_size/self.is_distributed`。
2. **`broadcast_parameters()`**（启动时、resume 后各调一次）：
   `broadcast_object_list([actor.state_dict(), critic.state_dict(), log_alpha.detach(), actor_obs_normalizer.state_dict(), critic_obs_normalizer.state_dict(), 各 optimizer.state_dict()], src=0)`，
   各 rank load；然后本地 `target_critic.load_state_dict(critic.state_dict())`。
3. **`_reduce_grads(params)`** 助手（照 ppo.py:432-460）：flatten 非 None 的 `param.grad` → `all_reduce(SUM)` → `/= world_size` → 写回。
4. **在 `update` 里插 3 次**（均 `if self.is_distributed`）：
   - critic：`critic_loss.backward()`(:136) 后、`step()`(:137) 前 → `_reduce_grads(self.critic.parameters())`
   - actor：(:143) 后、(:144) 前 → `_reduce_grads(self.actor.parameters())`
   - alpha：(:148) 后、(:149) 前 → `_reduce_grads([self.log_alpha])`
   **顺序保持**：critic.step() 必须先于 actor 前向（actor 复用更新后的 critic，:140），与单卡一致。
   （可选优化：actor+alpha 在同一 `actor.sample` 后，可 flatten 成一次 all-reduce 省 collective；critic 不可并。）
5. **normalizer（见修订#1，覆盖此处）**：把 `update()` 内 4 处 `update_stats=True`→`False`，
   normalizer 仅在 `observe()`（`train.py:374`）更新；然后**每 env step** `reduce_normalizers()` 同步一次（廉价），
   不再用"每 100 次"。启动时 broadcast normalizer 统一初值。

### 2.3 `source/.../rl/fast_sac/normalization.py`
- 加 `merge_from_ranks(world_size)`：把本 rank 的 `(count, mean, var)` 用 `all_gather` 收齐 K 份，
  按并行方差公式（把 `normalization.py:24-33` 的两两合并推广到 K 份：逐对 Chan 合并或一次性加权）
  算出全局 `mean/var/count`，各 rank `copy_` 写回。**精确、与 rank 顺序无关**。
  （退而求其次：`all_reduce(mean,AVG)`、`all_reduce(var,AVG)`、`count=sum(counts)`——近似，N 小时够用。）

### 2.4 `scripts/slurm/`（新增 `run_g1_paper_fastsac_multigpu.sbatch`）
```bash
#SBATCH -N1 --gres=gpu:8 --ntasks=1 --cpus-per-task=64 --time=...
# 环境前导同 run_dbg.sh（CONDA_PREFIX/PYTHONPATH）。务必让 8 张卡全可见（不要预先 narrow CUDA_VISIBLE_DEVICES）。
# 先 USD 预热：单进程单环境跑一次 import + env build 填 Omniverse/USD 缓存，避免 8 进程并发竞态。
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/fast_sac/train.py \
  --task Tracking-Flat-G1-1307-PaperFastSAC-Unified-v0 \
  --motion_file data/g1/1307.npz --num_envs 1024 \
  --max_steps <N> --train_steps <N> --eval_episodes 32 \
  --logger none --headless --distributed \
  agent.critic_v_min=-300.0 agent.critic_v_max=50.0 agent.critic_num_atoms=101 \
  agent.batch_size=32768 agent.tau=0.1 agent.updates_per_step=2
```

---

## 3. 关键数值/语义（执行者照此设定）

- **batch**：config `batch_size`=有效值，`per_rank=batch_size//8`。例：32768→每 rank 4096。
- **replay horizon**：单卡 `replay_size=204800`≈"50 步×4096 环境"。多卡每 rank 1024 环境，
  要保同样 50 步 horizon 则 `per_rank_replay=50×1024=51200`；若直接 `204800//8=25600` 则 horizon 减半。
  **建议** `per_rank_replay = max(replay_size//world_size, target_steps × per_rank_num_envs)`，target_steps≈50。
- **updates_per_step**：每 rank 保持 config 值（如 2），各 rank 同步执行 → 有效 UTD 不变。
- **normalizer_sync_interval**：100 次 update（可调）；启动必 broadcast。
- **种子**：`net 初值靠 broadcast 统一`；`env 种子 = base+rank`（多样性）。

---

## 4. 风险与缓解

| 风险 | 成因 | 缓解 |
|---|---|---|
| NCCL 启动 hang | init_process_group 在 set_device 前，或某 rank SimApp 启动慢致 rendezvous 超时 | 先 `set_device(local_rank)` 再任何 collective；init 放在 agent 构造后、各 rank 同时到达；加大 init 超时；`NCCL_DEBUG=INFO` 首跑验证 |
| Isaac USD 资产竞态 | 8 进程同时打开/缓存同一 robot/motion USD | **先单进程预热填缓存**；或按 `local_rank*Δ` 错峰；必要时每 rank 独立 Kit 缓存目录（呼应 wave1 实测的并发竞态坑） |
| 静默参数漂移 | 某优化器 step 前漏 all-reduce，或各 rank grad flatten 顺序不一致 | 3 次 reduce 全部无条件（分布式时）；同构构造+启动 broadcast 保证参数顺序一致；debug 期周期 all-reduce 参数校验和断言一致 |
| normalizer 失同步 | 各 rank 不同数据喂 normalizer，归一化进梯度图 | 启动 broadcast + 周期精确合并（§2.3） |
| 更新分支跨 rank 不一致致死锁 | 某 rank 因 buffer 未达 batch 跳过 collective，其它 rank 在等 | 用固定 warmup + 确定性 fill 使分支逐步一致；不要让更新分支按 rank 分叉 |
| 有效 batch 漂移 | 把 config batch 当每 rank 用 → 有效 ×8 | `per_rank=batch//world_size`（§3），写注释 |
| 多日志目录/wandb/checkpoint 竞写 | 所有 rank 跑 I/O | 全 I/O 门控 rank0；存盘前 barrier |
| 收尾拆解竞态 | 一 rank `simulation_app.close()` 时另一 rank 仍在 collective | barrier → destroy_process_group → 再 close |
| CPU 超订/ sim 变慢 | 8 sim 抢线程 | 依赖 AppLauncher 线程上限；`--cpus-per-task` 设好 |

---

## 5. 验证计划（必须做，按序）

1. **冒烟（2 卡）**：`--nproc_per_node=2`，num_envs 256、200 步。判据：两进程都进训练循环、无 NCCL hang、
   rank0 出 1 个 log 目录、agent.json 反映 per_rank batch。
2. **参数一致性**：训练中每 K 步在各 rank 算 `sum(p.sum() for p in actor.parameters())` 并 all-reduce 比较，
   断言各 rank 完全相等（容差 0）。验证 broadcast + grad all-reduce 正确。
3. **单卡 vs 多卡等价性 sanity**：固定有效 batch 与有效环境数，跑短程，比较 reward/critic_loss 趋势量级一致
   （不要求 bit 级，因数据不同；要求量级与学习方向一致）。
4. **吞吐扩展**：记录 steps/s，2/4/8 卡近线性（sim 为主，collective 小）。若不扩展 → 查 collective/CPU 瓶颈。
5. **数值正确**：8 卡满预算单跑，曲线 `mean_q` 离地板、`reward_mean` 升、`terminated_rate` 降；
   eval `success_rate>0`。归一化未漂移（周期打印各 rank normalizer mean 范数应一致）。
6. **回退保证**：不加 `--distributed` 时 world_size=1，所有 collective 跳过 → 与现单卡路径逐行等价（feature-flag）。

---

## 6. 实施顺序（分阶段，可独立验收）

- **P0 脚手架**：train.py 加 `--distributed`、dist init、device/seed、rank0 I/O 门控、barrier/teardown。验收=冒烟(1)。
- **P1 梯度并行**：algorithm.py 加 `broadcast_parameters` + `_reduce_grads`×3。验收=参数一致性(2)。
- **P2 normalizer 同步**：normalization.py `merge_from_ranks` + 周期调用。验收=各 rank normalizer 一致。
- **P3 启动与规模化**：sbatch + USD 预热 + batch/replay 切分。验收=吞吐扩展(4)。
- **P4 满预算训练**：8 卡跑通，success_rate>0（验证(5)）。
- 全程保 **feature-flag 回退**（验证(6)），不破坏现单卡路径。

---

## 7. 开放问题（实施前定）
1. `per_rank_replay` 取"保 horizon (51200)" 还是"保聚合 (25600)"？建议保 horizon。
2. normalizer 合并：精确 K-shard 还是近似 AVG？建议精确（代价小）。
3. 是否融合 actor+alpha 的 all-reduce 省一次 collective？P1 先不融，profile 后再说。
4. 启动用 torchrun（推荐）还是 srun 手动映射 SLURM_PROCID/LOCALID？单节点用 torchrun；将来多节点再切 srun 映射。

---

## 附 · 关键文件
- 改动：`scripts/fast_sac/train.py`、`source/.../rl/fast_sac/algorithm.py`、`normalization.py`、新增 `scripts/slurm/run_g1_paper_fastsac_multigpu.sbatch`
- 不动：`buffer.py`（每 rank 独立用，无需改）、`networks.py`、`config.py`（仅消费侧切分）
- 参考：`IsaacLab/.../app/app_launcher.py:668-690`、`rsl_rl/runners/on_policy_runner.py:351-393`、`rsl_rl/algorithms/ppo.py:419-460`、`IsaacLab/docs/.../multi_gpu.rst`
- 实测坑：`claude/wave1_results.md`（Isaac 并发 USD 竞态、override 浮点）
