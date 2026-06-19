# FastSAC 排查 — 第 1 波结果

> 执行 session 回填。配套 `claude/fastsac_debug_plan.md`（第 4-6 节）。
> 短程配置：Stage-I，4096 envs，3000 步，warmup 500，curve_interval 50，--skip_eval。
> 全部跑在 epyc3（rtx3090）。日志：`claude/dbg_logs/<run>.log`；曲线：`logs/fast_sac/g1_paper_fast_sac/<ts>_<run>/training_curve.csv`。

## 0. 前置完成情况
- **0.1 日志补丁**：已落地（主工作区）。`algorithm.py` 的 `update()` 返回新增 `mean_q`/`target_q_mean`；
  `train.py` 的 `CurveLogger._fieldnames` 与 `curve_row` 各加两列。`py_compile` 通过。
- **0.5 预检**：PASS。256 envs/200 步，三判据全过：无 traceback；csv 含 `mean_q`/`target_q_mean` 两列共 20 行；
  `params/agent.json` 的 `critic_v_min == -600.0`（override 生效）。
  - 关键发现：**override 需用 float 字面量**（`-600.0` 而非 `-600`），否则
    `configclass` 抛 `Incorrect type ... Expected float, Received int`。Wave1/2 全部据此用浮点。
  - 预检 mean_q 在 v_min=-600 下落到 ≈-240（未贴地板），target_q_mean≈-244 → 真实回报量级 ≈-240，
    远低于默认支撑 [-50,50]，初步支持根因 A。

## 1. 基建注意事项（执行中实证）
- **zsh 不做无引号变量分词**：`$SRUN`/`$COMMON` 直接展开会被当成单个词。改用 `${=COMMON}` 或内联 srun。
- **并发启动会触发 Isaac USD 资产竞态**：6 job 同时起，J0/J3/J4 报
  `ValueError: No contact sensors added to the prim ... no rigid bodies present`。
  J1/J2/J5 存活。**解法：错峰启动**（间隔 ~50s）后全部正常。
- 每个 job 用隔离的 omniverse HOME（`/tmp/wbt_omni_home_<jobid>_<run>`），见 `claude/run_dbg.sh`。
- J5（min-Q）隔离方式：因 `rl/`、`data/g1/` 均未被 git 跟踪，git worktree 不可行。
  改为把 `whole_body_tracking` 包整体复制到 `/tmp/wbt-minq-pkg` 并前置到 PYTHONPATH
  （`claude/run_dbg_minq.sh`），data 共享只读。min-Q 改动：`algorithm.py` 113/141 `mean_q`→`torch.minimum`，
  外加分布式 target 按两路期望值取较保守分布。

## 2. 各 job 数值（首批更新 vs 末值）

<!-- 待回填：用 `python claude/analyze_curves.py dbg-` 生成 -->

| run | rows | mq首 | mq末 | ΔMq | rew首 | rew末 | tq末 | critic_loss末 | term率末 | alpha末 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|

## 3. 决策树结论

<!-- 待回填 -->

## 4. 进入第 2 波的分支

<!-- 待回填 -->
