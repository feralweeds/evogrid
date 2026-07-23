# Runtime Safety and R5 Seed 5312 Regression

版本：1.0

日期：2026-07-22

状态：WP-RUNTIME-SAFETY-01 已完成；进入 DIG 前的通用安全补丁。

## 1. 目标

R5 road stop audit 暴露了一个不应作为道路规则继续调参的问题：

> Skill 可以在未成功执行核心动作时持续介入 fallback，并反复输出无法推进的移动动作。

本工作包只修通用 Runtime 安全边界，不修改道路 Skill 条件，不新增道路专用 DSL。

## 2. 修改范围

实现位置：

- `evogrid/skills/runtime.py`
- `evogrid/skills/schemas.py`
- `tests/test_skill_runtime.py`
- `tests/test_skill_schema.py`

新增通用安全能力：

| 能力 | 作用 |
| --- | --- |
| 可观察动作合法性 | `FOLLOW_ROUTE` 和 `ACT` 输出动作前，检查移动、DIG、BUILD_ROAD、MINE、DROPOFF 是否在当前可观察状态下合法 |
| 无进展检测 | 若上一轮 Skill 输出移动动作后 agent 位置未变化，下一轮停止该 Skill 介入 |
| 目标失败解锁 | 对导致 illegal/no-progress 的 episode target 做 episode 内屏蔽，避免继续锁定同一失败目标 |
| 连续介入限制 | 新增 `budget.max_consecutive_interventions`，默认限制非计数动作持续挤占 fallback |

这些机制不读取隐藏地图，只使用 `SkillContext` 中已有的 observation、route_plan 和 episode state。

## 3. Seed 5312 安全回归

脚本：

```powershell
python scripts\run_r5_road_stop_audit.py --out outputs\runtime_safety_seed5312_20260722 --seed-start 5312 --seed-count 1
```

安全补丁前，`return_gated_skill` 在 seed `5312`：

| road net | reward | ore delivered | builds | road usage | invalid actions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | -10.48 | 0 | 0 | 0 | 203 |

安全补丁后，seed `5312`：

| group | road net | reward | ore delivered | builds | road usage | invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_skill` | 0.00 | 76.36 | 8 | 0 | 0 | 0 |
| `ungated_skill` | 0.70 | 77.07 | 8 | 1 | 16 | 0 |
| `return_gated_skill` | 0.00 | 76.04 | 8 | 0 | 0 | 0 |

结论：

> Runtime safety patch 消除了 seed `5312` 的任务灾难和非法动作循环；return-gated Skill 不再把 episode 拉到 ore=0。

trace 中可见安全终止：

- `illegal_action`
- `episode_intervention_limit_reached`

这些是安全层阻断，不是道路 Skill 成功。

## 4. R5 Stop Audit 复跑

脚本：

```powershell
python scripts\run_r5_road_stop_audit.py --out outputs\runtime_safety_stop_audit_20260722
```

`5300-5319` 同种子三组结果：

| group | builds | positive build episodes | nonpositive build episodes | road net sum | road usage | reward mean | ore mean | invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_skill` | 0 | 0 | 0 | 0.00 | 0 | 30.142 | 3.3 | 0 |
| `ungated_skill` | 8 | 7 | 1 | 4.05 | 97 | 30.3325 | 3.3 | 0 |
| `return_gated_skill` | 7 | 7 | 0 | 4.05 | 95 | 30.318 | 3.3 | 0 |

`return_gated_skill - ungated_skill`：

| 指标 | mean delta |
| --- | ---: |
| `road_net_payoff` | `~0.0000` |
| `episode_reward` | `-0.0145` |
| `ore_delivered` | `0.0000` |
| `invalid_actions` | `0.0000` |

安全补丁后，任务无伤害问题基本解除；但 return gate 仍未在同种子上优于 ungated。

## 5. 决策

当前决策保持：

```text
freeze_handcrafted_return_gated_road_baseline = false
```

原因已经从“安全反例阻断”收敛为：

> return gate 能过滤非正收益建设，但在同种子对照中没有提升 road_net，也没有提升整体任务表现；因此不能把改进归因给该门控。

所以：

- 不继续增加道路规则条件；
- 不执行 DeepSeek return-gate 回归；
- 不把 return-gated road Skill 标为 development_verified；
- 下一步进入 `WP-R5-DIG-01`。

## 6. DIG 前置影响

DIG pilot 可以复用这次安全层：

- `FOLLOW_ROUTE` 输出动作前会检查可观察合法性；
- target 失败后可解锁并屏蔽；
- 非核心动作连续介入会被限制；
- Runtime 不再允许一个 Skill 长时间挤占 fallback。

这使 DIG 实验更适合检验通用基础设施：

- `SELECT_TARGET`
- `filters`
- `rank_by`
- episode target lock
- Verifier
- Registry
