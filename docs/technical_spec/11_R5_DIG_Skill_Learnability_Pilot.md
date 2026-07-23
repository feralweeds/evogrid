# R5 DIG Skill Learnability Pilot

版本：1.0

日期：2026-07-22

状态：WP-R5-DIG-01 开发 fixture 已完成；尚未进入真实 rollout。

## 1. 目标

道路主线停止后，本工作包验证：

> R5 的 `SELECT_TARGET`、filters、rank_by、episode target lock、Runtime safety、Verifier 和 Registry 是否能支持第二种环境改造 Skill：DIG。

本阶段不证明真实环境中 DIG 策略有效，只证明通用 Skill 基础设施不是 road 专用。

## 2. 通用 DSL 补充

为支持 DIG，不新增 oracle，只补充一个通用可观察特征：

```text
candidate.distance_from_agent
```

当 `SELECT_TARGET` 从 `visible_tiles` 或其他含 `pos` 的候选源读取目标时，Runtime 会根据当前 agent 位置自动填充该距离。该特征已存在于 schema 的候选特征集合中，本次只是让缺省候选也能获得这个可观察派生值。

该能力可复用于 road、DIG、frontier target 和其他局部目标选择。

## 3. Handcrafted DIG Candidate

脚本：

```powershell
python scripts\run_handcrafted_dig_skill_fixture.py --out outputs\r5_handcrafted_dig_fixture_20260722
```

Candidate：

```text
handcrafted_adjacent_obstacle_dig@1.0.0
```

核心 procedure：

```text
SELECT_TARGET source=visible_tiles
  filters:
    candidate.tile_type == OBSTACLE
    candidate.distance_from_agent <= 1
  rank_by:
    candidate.distance_from_agent asc
  episode_store_as=dig_target

IF target exists:
  ACT DIG
ELSE:
  RETURN no_adjacent_obstacle
```

budget：

```text
max_uses_per_episode = 1
episode_use_actions = ["DIG"]
stop_after_success = true
max_consecutive_interventions = 2
```

## 4. Fixture Protocol

正负场景：

| stratum | context | expected |
| --- | --- | --- |
| `adjacent_obstacle_context` | agent 旁边有可观察 obstacle | DIG 一次 |
| `no_adjacent_obstacle_context` | agent 旁边没有 obstacle | 不触发 |

该 fixture 使用 `SkillVerifier` 和 `SkillRegistry`，但属于 development fixture，不是 formal DIG Skill claim。

## 5. 结果

输出：

- `outputs/r5_handcrafted_dig_fixture_20260722/run_manifest.json`
- `outputs/r5_handcrafted_dig_fixture_20260722/candidate.json`
- `outputs/r5_handcrafted_dig_fixture_20260722/registry/`

结果：

| metric | value |
| --- | ---: |
| decision | `verified` |
| promoted_status | `verified` |
| sample_size | 60 |
| paired_delta_mean | 0.5 |
| success_rate | 0.5 |
| activation_rate | 0.5 |
| false_trigger_rate | 0.0 |
| runtime_failure_rate | 0.0 |

解释：

> 同一套 Skill infrastructure 能够表达并验证一个非 road 的 DIG fixture。R5 SELECT_TARGET 已不再只是道路专用扩展。

边界：

- 这是合成 fixture；
- 没有证明真实 rollout 中 DIG 能提高 reward 或 ore delivery；
- 没有证明 LLM 能自主归纳 DIG Skill；
- 没有新增 DIG oracle。

## 6. 下一步

下一允许工作包：

```text
WP-R5-DIG-02 — DIG Real Rollout Diagnosis
```

建议顺序：

1. 定义真实 DIG positive/negative 场景；
2. 先跑 no-skill vs handcrafted DIG 同种子对照；
3. 检查 DIG 是否非冗余于 fallback 的内置 route planner；
4. 若 handcrafted DIG 在真实 rollout 中仍无增益，停止 DIG 优化并记录负结果；
5. 若有增益，再考虑一次受限 LLM DIG 回归。
