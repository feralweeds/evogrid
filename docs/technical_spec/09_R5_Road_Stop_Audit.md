# R5 Road Stop Audit

版本：1.0

日期：2026-07-22

状态：开发分区停止审计；道路主线不再继续规则精调。

## 1. 审计问题

本审计不再尝试增加第二个道路规则条件，也不尝试把人工道路 Skill 推成正式 Verified Skill。

唯一问题是：

> 在完全相同的 `5300-5319` seeds 上，`cargo.has_ore == true` return gate 是否真的优于原始 route-target Skill，并且不伤害整体任务表现？

## 2. 冻结边界

本轮审计冻结：

- Skill Runtime；
- Skill DSL；
- 环境配置；
- Verifier protocol；
- route-target selection procedure；
- paired seeds：`5300-5319`。

三组对照：

| 组别 | 内容 |
| --- | --- |
| `no_skill` | 只运行 `RouteOnlyAgent` fallback |
| `ungated_skill` | 原始 route-target road Skill |
| `return_gated_skill` | 仅增加 `cargo.has_ore == true` |

脚本：

```powershell
python scripts\run_r5_road_stop_audit.py --out outputs\r5_road_stop_audit_20260722
```

输出：

- `outputs/r5_road_stop_audit_20260722/run_manifest.json`
- `outputs/r5_road_stop_audit_20260722/audit_report.md`
- `outputs/r5_road_stop_audit_20260722/episodes/metrics.csv`
- `outputs/r5_road_stop_audit_20260722/comparisons/pairwise_summary.csv`

## 3. 同种子三组结果

| group | builds | positive build episodes | nonpositive build episodes | positive ratio | road net sum | road usage | reward mean | ore mean | steps mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_skill` | 0 | 0 | 0 | 0.000 | 0.000 | 0 | 30.142 | 3.300 | 220.000 |
| `ungated_skill` | 8 | 7 | 1 | 0.875 | 4.050 | 97 | 30.334 | 3.300 | 220.000 |
| `return_gated_skill` | 7 | 7 | 0 | 1.000 | 4.050 | 95 | 25.994 | 2.900 | 220.000 |

`return_gated_skill` 的正面信号：

- 过滤掉了 `ungated_skill` 的 1 次非正收益建设；
- 修路 episode 全部为正收益；
- false-trigger 风险下降。

但同种子因果审计不支持继续宣称它优于 ungated：

- road net sum 与 ungated 相同：`4.05 vs 4.05`；
- road usage 略低：`95 vs 97`；
- episode reward mean 低于 ungated：`25.994 vs 30.3345`；
- ore delivered mean 低于 ungated：`2.9 vs 3.3`；
- invalid actions 比 ungated 多 `203` 次，全部来自 seed `5312`。

## 4. Gated vs Ungated

`return_gated_skill - ungated_skill`：

| 指标 | mean delta | 说明 |
| --- | ---: | --- |
| `road_net_payoff` | `~0.0000` | 未优于原始 Skill |
| `episode_reward` | `-4.3405` | 被 seed `5312` 强烈拉低 |
| `ore_delivered` | `-0.4000` | seed `5312` 少交付 8 ore |
| `num_build_road` | `-0.0500` | 少建 1 条路 |
| `road_total_usage_count` | `-0.1000` | 总复用略低 |
| `invalid_actions` | `+10.1500` | 总计多 203 次非法动作 |

任务无伤害检查失败。

## 5. 关键反例：Seed 5312

seed `5312`：

| group | road net | reward | ore delivered | builds | road usage | invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_skill` | 0.00 | 76.36 | 8 | 0 | 0 | 0 |
| `ungated_skill` | 0.70 | 77.07 | 8 | 1 | 16 | 0 |
| `return_gated_skill` | 0.00 | -10.48 | 0 | 0 | 0 | 203 |

trace 显示，return-gated Skill 在携矿阶段选择了距离较远的 rough route target，随后持续执行 `FOLLOW_ROUTE` 产生的 `MOVE_UP`，但该动作没有推进状态，episode 中反复失败。

这暴露的是通用 runtime/skill 交互风险，而不是继续给道路 Skill 加条件的理由：

- `FOLLOW_ROUTE` 产生的动作缺少执行前合法性保护；
- `max_uses_per_episode` 当前只计入 `BUILD_ROAD`，无法限制没有建路但反复移动的 Skill 介入；
- target lock 会在未成功建设时保持同一目标，可能放大局部路线错误。

这些问题应作为后续通用 Runtime 安全审计输入，而不是在 R5 road 主线中继续人工调参。

## 6. 决策

当前停止审计结论：

```text
freeze_handcrafted_return_gated_road_baseline = false
task_no_harm_passed = false
```

因此：

- 不冻结为 `Handcrafted Return-Gated Road Skill Baseline`；
- 不标记为 `development_verified`；
- 不进行受限 DeepSeek return-gate 回归；
- 不使用剩余一轮额度继续增加道路规则条件；
- 停止道路专用优化，转向 DIG 或另一种 Skill。

这并不否定上一轮 pilot 的价值。更准确的结论是：

> `cargo.has_ore == true` 在独立 pilot 中是一个真实正信号，能减少非正收益修路；但同种子 STOP audit 不能证明它优于 ungated route-target Skill，也不能证明它对整体任务无伤害。

## 7. 后续工作

下一工作包应转向：

```text
WP-R5-DIG-01 — DIG Skill Learnability Pilot
```

目标是验证当前通用基础设施是否能服务第二种 Skill：

- `SELECT_TARGET`
- `filters`
- `rank_by`
- episode target lock
- Verifier
- Registry

进入 DIG 前建议只做通用安全审计，不做道路专用修补：

- 检查 runtime 是否应在 `FOLLOW_ROUTE` 输出动作前验证动作合法性；
- 检查 `max_uses_per_episode` 是否应能覆盖非 `ACT` 的环境动作；
- 检查 target lock 是否需要在连续未推进时释放。

这些属于通用 Skill Runtime 安全问题，可服务 DIG、BUILD_ROAD 和后续多步 Skill。

## 8. Runtime Safety 后复核

`WP-RUNTIME-SAFETY-01` 完成后，已复跑 seed `5312` 与 `5300-5319` stop audit。

seed `5312` 的非法动作循环已消除：

| group | road net | reward | ore delivered | builds | road usage | invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_skill` | 0.00 | 76.36 | 8 | 0 | 0 | 0 |
| `ungated_skill` | 0.70 | 77.07 | 8 | 1 | 16 | 0 |
| `return_gated_skill` | 0.00 | 76.04 | 8 | 0 | 0 | 0 |

完整 `5300-5319` 复跑后：

| group | builds | positive build episodes | nonpositive build episodes | road net sum | road usage | reward mean | ore mean | invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_skill` | 0 | 0 | 0 | 0.00 | 0 | 30.142 | 3.3 | 0 |
| `ungated_skill` | 8 | 7 | 1 | 4.05 | 97 | 30.3325 | 3.3 | 0 |
| `return_gated_skill` | 7 | 7 | 0 | 4.05 | 95 | 30.318 | 3.3 | 0 |

因此，停止结论保持，但理由更精确：

> return gate 不再造成任务安全反例，但它在同种子对照中仍没有优于 ungated road Skill，不能冻结为 development_verified baseline。

下一工作包仍为 `WP-R5-DIG-01`。
