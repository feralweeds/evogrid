# R5-04A Road Skill Learnability Diagnosis

版本：1.0

日期：2026-07-22

状态：开发分区诊断；不是 final/test 结论。

## 1. 目标

本工作包不以“继续优化道路 Skill 到通过”为目标，而是判断：

> 当前道路建设失败案例与成功案例，在建设发生前是否存在简单、通用、可观察的分界信号。

若存在分界，最多允许两轮最小 handcrafted pilot；若不存在，则停止道路专用优化并转向第二类 Skill。

## 2. 冻结边界

本轮诊断冻结：

- Skill Runtime；
- route target DSL；
- 环境配置；
- Verifier protocol；
- paired verification 逻辑。

唯一允许修改的 candidate 条件：

```json
{"feature": "cargo.has_ore", "op": "eq", "value": true}
```

解释：该条件检验“只在返航阶段建设道路是否减少过早投资”。它不是道路收益 oracle，也不读取未来复用次数。

## 3. 规则发现样本

来源：`outputs/r5_handcrafted_route_target_rollout`

seeds：`5200-5219`

该 seed 段用于发现假设，不用于检验假设。

| seed | 结果 | 携矿 | 目标访问数 | 已观察路线访问和 | 剩余步数 | 路线长 | 目标距离 | route_order | 后续真实使用 | road_net |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5203 | 非正 | false | 2 | 2 | 198 | 6 | 2 | 2 | 0 | -0.10 |
| 5204 | 正 | true | 4 | 7 | 205 | 9 | 6 | 6 | 20 | 0.90 |
| 5205 | 正 | true | 2 | 11 | 192 | 10 | 5 | 5 | 12 | 0.50 |
| 5206 | 非正 | false | 2 | 2 | 205 | 5 | 0 | 0 | 0 | -0.10 |
| 5218 | 非正 | false | 2 | 2 | 216 | 9 | 0 | 0 | 0 | -0.10 |
| 5219 | 非正 | false | 2 | 2 | 213 | 5 | 0 | 0 | 1 | -0.05 |

简单条件枚举结果：

| 条件 | 保留正例 | 保留非正例 | 判断 |
| --- | ---: | ---: | --- |
| `cargo.has_ore == true` | 2 | 0 | 可检验 |
| `route_observed_visit_sum >= 7` | 2 | 0 | 可检验，但需要先暴露 aggregate feature |
| `target_route_order >= 5` | 2 | 0 | 可检验，但可能更接近路线位置启发式 |
| `target_visit_count >= 3` | 1 | 0 | 过严，会漏掉一个正例 |
| `remaining_steps` 阈值 | 不稳定 | 不稳定 | 不能分开 |
| `route_length` 阈值 | 不稳定 | 不稳定 | 不能干净分开 |

结论：当前样本存在简单可观察分界，优先检验 `cargo.has_ore == true`，因为该 feature 已在 DSL 中存在，不需要新增字段。

## 4. 新 Seed 假设检验

脚本：

```powershell
python scripts\run_handcrafted_return_gate_rollout.py --out outputs\r5_handcrafted_return_gate_rollout
```

seeds：`5300-5319`

该 seed 段未参与上面的规则发现。

唯一 candidate 改动：

```text
Added applicability leaf: cargo.has_ore eq true.
```

结果：

| 指标 | 无返航门控发现样本 | 返航门控新 seed pilot |
| --- | ---: | ---: |
| decision | rejected | verified |
| paired_delta_mean | 0.0525 | 0.2025 |
| success_rate | 0.10 | 0.35 |
| activation_rate | 0.40 | 0.40 |
| false_trigger_rate | 0.20 | 0.00 |
| runtime_failure_rate | 0.00 | 0.00 |
| enabled road builds | 6 | 7 |
| positive road-net builds | 2 | 7 |
| nonpositive road-net builds | 4 | 0 |
| enabled road net sum | 1.05 | 4.05 |
| enabled road total usage | 33 | 95 |

边界说明：两列使用不同 seed 段，不能作为严格 A/B 因果估计。它们的作用是：先用 `5200-5219` 发现假设，再用 `5300-5319` 检查该假设是否立即崩溃。

## 5. 失败类型复核

上一版失败主要属于：

- A. 目标选择错误：会在去矿阶段和路线前段过早建设；
- B. 复用证据不足：亏损建路后续真实使用次数为 0 或 1；
- C. episode 剩余时间不足：当前证据不支持，亏损案例剩余步数不低；
- D. 道路经济参数不支持回本：不成立，正例可产生 0.5-0.9 净收益；
- E. 观测信息不足：不作为当前主因，因为 `cargo.has_ore` 能分开发现样本；
- F. verifier 门槛与任务目标不匹配：不作为当前主因，上一版 verifier 拒绝合理。

## 6. 当前决策

返航阶段门控假设通过了一个新的开发 seed pilot，但仍不能宣称：

- DeepSeek 自主学会道路 Skill；
- 道路 Skill 已经通过 formal/test；
- 环境-Skill 同步演化已经成立。

允许结论：

> 在受控 corridor 真实 rollout pilot 中，`cargo.has_ore == true` 作为返航阶段门控显著减少了过早、低复用道路建设，并在新开发 seed 上保留了正收益道路机会。

下一步不应继续无上限优化 road。若继续 road，只允许第二轮且必须只改一个核心因素；否则应进入 R5 road stop audit 或 DIG 预备任务。

## 7. 后续停止规则

道路主线剩余额度：

- handcrafted pilot 剩余最多 1 轮；
- 最多新增 2 个通用可观察特征，目前已新增 0 个；
- 不允许新增道路收益 oracle；
- 不允许使用 final/test 结果调参；
- 不允许继续生成 road v10/v11/v12。

立即停止条件仍然有效：

- 下一轮不能接近或保持当前 pilot 改善；
- 只能依赖未来真实复用次数区分正负案例；
- activation 通过收窄条件接近 0；
- 人工 Skill 仍不能稳定通过；
- 新增规则无法复用于 DIG 或其他 Skill。
