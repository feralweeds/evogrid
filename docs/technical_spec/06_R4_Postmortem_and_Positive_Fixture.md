# R4 Postmortem 与 Positive Fixture 诊断记录

版本：1.0

记录日期：2026-07-21

状态：R4 DeepSeek revision loop 已触发 stop rule；已新增 handcrafted positive fixture 验证晋升通道。

## 1. R4 目标

R4 的目标不是证明 Agent 已经完成自进化，而是验证一条受控的 Skill 生命周期：

1. LLM 只能提出 Candidate Skill；
2. Candidate 必须经过独立 verifier；
3. 只有 verifier 产出的通过报告可以把 Candidate 晋升为 Verified；
4. revision 只能使用聚合 verification feedback，不能读取逐 seed 真值或 test 结果；
5. Agent 能力必须由 Verified Skill Set 和固定 benchmark 表示，而不是单次 reward 或人工解释。

本轮聚焦的是道路修建 Skill：在已知运输路线、粗糙地形、预计复用次数足以回本时修路，并避免在负例或低收益情形过度修路。

## 2. Candidate 迭代记录

| 版本 | 结果 | 主要失败原因 | 归因 |
| --- | --- | --- | --- |
| 1.0.3 | rejected / 近零激活 | memory predicate 过严，几乎不触发；同时暴露 SkillAgent 没有把 fallback route/memory hints 完整桥接到 SkillContext | Candidate + 平台 |
| 1.0.4 | rejected | 激活恢复，但出现负收益和 runtime failure | Candidate |
| 1.0.5 | rejected | tile guard 修复部分 runtime 问题，但仍过度修路；同时暴露 continuous terrain 下 route rough diagnostics 统计错误 | Candidate + 平台 |
| 1.0.6 | invalid / no-op | 形式上 accepted，但 executable contract 没有真实行为变化 | Candidate |
| 1.0.7 | invalid | 使用 `route.remaining_length_bucket gte 2`，对 enum bucket 使用了非法有序比较 | Candidate + schema 缺口 |
| 1.0.8 | invalid / no-op | 被 no-op revision guard 正确拒绝 | Candidate |
| 1.0.9 | rejected | 合法且可执行，使用 `route.remaining_length_bucket in ["medium", "long"]`，但 verifier 指标未过：`paired_delta_mean = -0.0072`，`success_rate = 0.2`，`activation_rate = 0.3`，`runtime_failure_rate = 0.0`，`false_trigger_rate = 0.1` | Candidate / DSL 表达不足 |

## 3. 已修复的平台问题

R4 过程中已修复或加固的平台问题如下：

- SkillAgent route/memory context bridge：Skill runtime 现在可以读取 fallback agent 产生的 route plan 和 memory summary。
- continuous terrain route rough diagnostics：修复了正例分层里 route rough opportunity 统计不连续的问题。
- no-op revision guard：只改描述、rationale 或 metadata 的 revision 不再被当作有效候选。
- applicability schema validation：对 boolean、numeric、enum feature 做类型和 operator 约束，特别是拒绝 enum bucket 的 `gte/lte` 等非法有序比较。
- revision prompt contract：prompt 明确列出 allowed features、bucket 类型和可执行修改要求，减少 DeepSeek 产生不可执行 JSON 的概率。
- runtime episode controls：新增 `budget.max_uses_per_episode` 和 `budget.stop_after_success`，并由 `SkillEpisodeState` 在 runtime / SkillAgent 层执行。

## 4. 尚未解决的瓶颈

当前主要瓶颈不是 verifier 完全不可用，而是 Candidate 能表达的行为仍太弱。

最关键的缺口：

- DeepSeek revision 倾向于在 predicate 上反复收窄或写出非法 bucket 条件，没有稳定学会“少量、有序、高收益地修路”。
- DSL 缺少 route-local target selection，比如“当前路线上的第一个高收益 rough tile”。
- DSL 还缺少候选排序语义，比如 `rank_candidates_by`。
- 当前 rollout verifier 对真实环境收益很敏感，Candidate 一旦过度触发就会被负收益压倒。
- R4 仍没有从真实 DeepSeek 候选中产生 Verified Skill。

## 5. 为什么触发 stop rule

继续 v10 的信息增量已经很低：

1. 多轮 revision 已覆盖零激活、负收益、runtime failure、no-op、非法 schema、合法但无效等主要失败类型；
2. 最新 v1.0.9 已经通过 schema 和 runtime，但真实 paired verification 仍显著不过关；
3. 继续要求 DeepSeek 在同一 DSL 下微调 predicate，大概率只会在过度收窄和过度触发之间摆动；
4. 需要先回答更基础的问题：如果给一个手工已知有效的 Skill，verifier 是否真的能把它晋升为 Verified。

因此 R4 DeepSeek loop 应停止，转入 positive fixture 和 DSL 能力增强。

## 6. Handcrafted Positive Fixture

新增脚本：

```powershell
python scripts\run_handcrafted_positive_skill_fixture.py --out outputs\r4_handcrafted_positive_fixture
```

该 fixture 构造一个 handcrafted candidate：

- 只在 positive transport context 中触发；
- 当前 tile 必须是可修路地块；
- route 必须存在且为 known transport route；
- route length bucket 必须为 `medium` 或 `long`；
- `future_route_uses >= road_break_even_uses` 才执行 `BUILD_ROAD`；
- `max_uses_per_episode = 1`；
- `stop_after_success = true`。

本地运行结果：

| 指标 | 结果 |
| --- | ---: |
| decision | `verified` |
| promoted_status | `verified` |
| sample_size | 60 |
| paired_delta_mean | 0.5 |
| success_rate | 0.5 |
| activation_rate | 0.5 |
| false_trigger_rate | 0.0 |
| runtime_failure_rate | 0.0 |

结论：在受控 positive/negative transport-context fixture 中，现有 verifier + registry 可以完成 Candidate -> Verifier -> Verified 晋升通道。也就是说，R4 当前失败不能简单归因于“verifier 不能晋升任何 Skill”；更可能的瓶颈是 DeepSeek Candidate 质量和 DSL 表达能力。

边界：该 fixture 是合成诊断，不等价于真实 rollout formal acceptance；不能据此宣称环境中已经学得有效道路 Skill。

## 7. 本次 DSL 增强

已实现：

- `budget.max_uses_per_episode`：限制同一 Skill 在单个 episode 内成功产生环境动作的次数。
- `budget.stop_after_success`：Skill 一旦成功产生环境动作，本 episode 后续调用直接停止。
- `SkillEpisodeState`：由 runtime 记录每个 Skill 的 episode use count 和 stop 状态。
- `SkillAgent.reset()`：每个 episode 重置 `SkillEpisodeState`。

尚未实现，建议作为下一阶段：

- `first_matching_tile_on_route`
- `rank_candidates_by`
- 显式 route tile list / route-local target selector
- episode-state predicate feature，例如 `episode_state.skill_use_count`
- 多候选 target 的收益估计和排序 trace

## 8. 下一阶段建议

最合理的下一步是 `R5-DSL-ROUTE-TARGET`：

1. 在 route_plan observable context 中加入当前 route 的有限、可观察 tile 摘要；
2. 实现 `SELECT_ROUTE_TILE` 或等价的 `first_matching_tile_on_route` procedure op；
3. 实现 `rank_candidates_by`，只允许白名单 estimator 输出排序；
4. 用真实 rollout 重新构造 handcrafted road Skill，而不是 synthetic fixture；
5. 再让 DeepSeek 基于 R4 postmortem 生成新 Candidate。

在 R5 前，不建议继续 v10。继续 v10 之前应先让 DSL 能表达“只修当前路线第一个高收益 rough tile”这类受控行为。
