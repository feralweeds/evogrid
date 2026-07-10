# Controlled Random Road-Learning Curriculum

This note documents the controlled random curriculum used after the first fully random map smoke tests.

The goal is not to recreate the old fixed rough corridor. The generator should provide enough positive road-learning examples while keeping negative and mixed cases so the agent must learn when road building is worthwhile.

## Config

```text
configs/env_controlled_corridor_curriculum.yaml
```

The map mode is:

```yaml
env:
  map_mode: controlled_corridor_curriculum
```

## Map Types

The generator samples three hidden map types. The type is not exposed to the agent.

1. Positive maps
   Transport paths are more likely to include rough terrain. Road building can produce positive payoff after repeated use.

2. Mixed maps
   Some rough terrain is on useful routes and some is off-route. The agent must distinguish useful road opportunities from distractors.

3. Negative maps
   Rough terrain is mostly off-route or low-reuse. Blind road building should often lose payoff.

Default weights:

```text
positive: 0.60
mixed:    0.20
negative: 0.20
```

## Randomization

Each seed randomizes:

- base position
- ore position
- route shape
- rough transport band position and thickness
- rough tiles on the route
- off-route rough distractors
- obstacles outside protected paths

This should encourage learning a transferable condition: road building is valuable when a high-cost tile lies on a repeated transport route.

## Diagnostics

The environment now reports these analysis-only metrics:

```text
rough_tile_count
buildable_tile_count
shaping_opportunity_count
transport_corridor_length
route_rough_tile_count
off_route_rough_tile_count
positive_road_opportunity_count
```

These fields are not included in the current LLM compact metrics prompt.

## Smoke Commands

Baseline:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:3 --test-seeds 1000:1003 --episodes-per-seed 1 --max-steps 120 --groups route_only random_road rough_rule_road exploration_road --out outputs/runs/controlled_corridor_baseline_smoke
```

Mock LLM:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:3 --test-seeds 1000:1003 --episodes-per-seed 1 --max-steps 120 --groups route_only exploration_road llm_no_road_learning llm_with_road_learning --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --out outputs/runs/controlled_corridor_llm_mock_smoke
```

Use `--max-learned-builds-per-episode` to prevent a positive tile-level estimate from turning into unlimited road construction in one episode. This keeps learned exploitation visible while making overbuilding measurable.

## Contextual Road Learning

Road payoff records are now enriched with the agent-visible context from the `BUILD_ROAD` trace:

```text
route_on_build
build_mode
route_remaining_length
known_as_transport_corridor
observed_visit_count_on_build
build_decision_source
```

`RoadLearningModule` still supports old tile-only records, but context records are preferred. The estimate order is:

1. exact contextual match
2. transport/off-route contextual aggregate
3. legacy tile-specific fallback only when no context evidence exists for that tile

This prevents one positive `ROUGH` example from becoming a blanket instruction to pave every rough tile. In the controlled mock run, contextual learning reduced learned overbuilding and made `llm_with_road_learning` behave like the capped exploration baseline instead of building extra roads from an overgeneral tile estimate.

The current limitation is important: context learning has reduced the false-positive surface, but it has not yet shown that `llm_with_road_learning` outperforms `llm_no_road_learning`. It mainly changes why the same road is built: learned evidence can replace exploration as the decision source.

## Evidence Thresholds

Learned road builds can be gated by:

```text
min_contextual_evidence_count
positive_rate_threshold
learned_value_threshold
confidence_threshold
require_contextual_evidence
require_on_route_learned_build
```

The generalization runner includes learned-threshold profiles:

```text
llm_with_road_learning_no_threshold
llm_with_road_learning_loose_threshold
llm_with_road_learning_balanced_threshold
llm_with_road_learning_medium_threshold
llm_with_road_learning_strict_threshold
```

The short aliases `no_threshold`, `loose_threshold`, `balanced_threshold`, `medium_threshold`, and `strict_threshold` also run the LLM-with-road-learning policy with the corresponding gate.

Threshold ablation:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --max-steps 200 --groups llm_with_road_learning_no_threshold llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold llm_with_road_learning_medium_threshold llm_with_road_learning_strict_threshold --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --out outputs/runs/controlled_corridor_threshold_ablation_20x2
```

New metrics include:

```text
strong_learned_evidence_count
weak_learned_evidence_count
build_when_strong_learned_evidence_count
skip_when_strong_learned_evidence_count
llm_strong_learned_evidence_count
llm_weak_learned_evidence_count
llm_build_given_strong_evidence_count
llm_skip_given_strong_evidence_count
p_build_given_strong_learned_evidence
p_build_given_weak_learned_evidence
p_llm_build_given_strong_learned_evidence
p_llm_build_given_weak_learned_evidence
```

## Learned-Only Test

Use `--test-exploration-budget 0` to allow exploration during training but disable exploration during test. In test, `BUILD_ROAD` can then come only from strong learned evidence.

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --max-steps 200 --groups route_only llm_no_road_learning llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold llm_with_road_learning_medium_threshold llm_with_road_learning_strict_threshold --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --test-exploration-budget 0 --out outputs/runs/controlled_corridor_learned_only_balanced_20x2
```

This is the cleanest test of whether training experience transfers into test-time environment shaping.

Use `--train-episodes-per-seed` and `--test-episodes-per-seed` when increasing training evidence density without also increasing test cost.

## Early Smoke Result

On the small smoke run, the curriculum produced both positive and negative road examples:

- `random_road` frequently lost road payoff.
- `rough_rule_road` found positive payoff on some maps but was not universally safe.
- mock `llm_with_road_learning` produced learned road builds and positive test road payoff on the aggregate smoke.

This is enough to justify a larger mock run before spending real LLM calls.

## Latest Mock Result

Contextual mock run:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --max-steps 200 --groups route_only exploration_road llm_no_road_learning llm_with_road_learning --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --out outputs/runs/controlled_corridor_contextual_20x2
```

Observed aggregate:

- `llm_with_road_learning` no longer overbuilds relative to `llm_no_road_learning`.
- test mean `llm_build_road_count` is `1.125` for both LLM groups.
- test mean `road_net_payoff` is `0.1005` for both LLM groups.
- test mean reward is `23.99175` for both LLM groups.
- `llm_with_road_learning` sources those builds from contextual learned estimates, while `llm_no_road_learning` sources them from exploration.

Interpretation: the context fix is a stability improvement, not a final success criterion. The next useful test is to make learned decisions stricter than exploration, for example by requiring stronger contextual evidence or by adding a decision rule that skips weak learned contexts instead of reproducing the exploration baseline.

## Threshold Ablation Result

Run:

```text
outputs/runs/controlled_corridor_threshold_ablation_20x2
```

Test means:

```text
group                                      reward    roads   learned_roads   positive_ratio   avg_payoff   road_net
llm_with_road_learning_no_threshold        23.9918   1.125   1.125           0.1917           0.0335       0.1005
llm_with_road_learning_loose_threshold     24.0625   1.900   1.025           0.2025           0.0355       0.1685
llm_with_road_learning_medium_threshold    24.0790   1.275   0.150           0.2042           0.0469       0.1863
llm_with_road_learning_strict_threshold    24.0790   1.275   0.150           0.2042           0.0469       0.1863
```

Interpretation: medium/strict thresholds reduce learned over-reuse while improving average payoff per road and total road net payoff. The reward gain is small but in the desired direction. This is closer to the intended behavior: fewer learned builds, stronger evidence, better road quality.

## Learned-Only Result

Run:

```text
outputs/runs/controlled_corridor_learned_only_20x2
```

Test totals:

```text
group                                      roads   llm_explore   llm_learned   strong_evidence   weak_evidence   road_net   reward_mean
route_only                                 0       0             0             0                 0               0.00       25.685
llm_no_road_learning                       0       0             0             0                 0               0.00       25.685
llm_with_road_learning_loose_threshold     45      0             45            1074              906             6.49       24.053
llm_with_road_learning_balanced_threshold  19      0             19            44                2228            9.55       24.872
llm_with_road_learning_medium_threshold    0       0             0             0                 2511            0.00       25.685
llm_with_road_learning_strict_threshold    0       0             0             0                 2511            0.00       25.685
```

Interpretation:

- The learned-only setup works: test exploration builds are zero.
- `llm_no_road_learning` becomes a clean no-road control in test.
- loose thresholds transfer learned evidence into test-time road building and produce positive road net payoff, but reward is below route-only.
- balanced thresholds reduce learned builds from 45 to 19, keep test exploration at zero, improve road net payoff from 6.49 to 9.55, and bring reward closer to route-only.
- medium/strict thresholds produce no strong evidence in test, so they do not build roads. This means the current training run does not generate enough high-confidence contextual evidence for those gates.

The balanced profile is the current best mock learned-only candidate. The next research step is either evidence-density scaling or a small real LLM learned-only pilot using `llm_with_road_learning_balanced_threshold`.

## Evidence-Density Scaling Result

Run:

```text
outputs/runs/controlled_corridor_evidence_density_20x4_train_20x1_test
```

Command:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --train-episodes-per-seed 4 --test-episodes-per-seed 1 --max-steps 200 --groups route_only llm_no_road_learning llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold llm_with_road_learning_medium_threshold llm_with_road_learning_strict_threshold --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --test-exploration-budget 0 --out outputs/runs/controlled_corridor_evidence_density_20x4_train_20x1_test
```

Test totals:

```text
group                                      roads   llm_explore   llm_learned   strong_evidence   weak_evidence   road_net   reward_mean
route_only                                 0       0             0             0                 0               0.00       22.188
llm_no_road_learning                       0       0             0             0                 0               0.00       22.188
llm_with_road_learning_loose_threshold     21      0             21            725               232             2.05       21.245
llm_with_road_learning_balanced_threshold  10      0             10            26                977             4.00       21.835
llm_with_road_learning_medium_threshold    0       0             0             0                 1108            0.00       22.188
llm_with_road_learning_strict_threshold    0       0             0             0                 1108            0.00       22.188
```

Road-quality totals:

```text
group                                      roads   positive_roads   total_positive_ratio
llm_with_road_learning_loose_threshold     21      8                0.381
llm_with_road_learning_balanced_threshold  10      8                0.800
```

Interpretation:

- Test exploration is still zero, so road builds are learned-only.
- Balanced becomes more selective than loose: 10 learned builds instead of 21, higher road net payoff, and much better positive-road ratio.
- Medium and strict still produce no strong evidence after doubling train episodes from 2 to 4 per seed. The bottleneck is therefore not just sample count; the current medium/strict gates are too hard for the present context aggregation and train/test map variation.

Next useful step: split test maps into positive on-route, off-route negative, and mixed contexts to verify whether balanced is skipping the right roads, not merely building fewer roads.

## Positive/Negative Context Split Result

The generalization runner supports test-only context splits:

```text
--test-context-scenarios positive negative mixed
```

Training still uses the normal controlled curriculum mixture. Test is repeated once per forced context, with exploration disabled:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --train-episodes-per-seed 4 --test-episodes-per-seed 1 --max-steps 200 --groups route_only llm_no_road_learning llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold --test-context-scenarios positive negative mixed --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --test-exploration-budget 0 --out outputs/runs/controlled_corridor_context_split_20x4_train_20x1_test
```

The run writes `context_comparison.csv` in addition to the aggregate comparison files.

Test totals:

```text
context   profile    roads   learned   explore   strong   weak   on   off   pos   neg   road_net   pos_ratio   avg_payoff
positive  loose      21      21        0         742      231    21   0     7     14    2.61       0.333       0.124
positive  balanced   18      18        0         39       858    18   0     15    3     8.30       0.833       0.461
mixed     loose      21      21        0         746      213    21   0     8     13    2.70       0.381       0.129
mixed     balanced   11      11        0         43       932    11   0     8     3     5.10       0.727       0.464
negative  loose      21      21        0         754      235    21   0     6     15   -0.38       0.286      -0.018
negative  balanced   1       1         0         1        1089   1    0     1     0     0.75       1.000       0.750
```

Interpretation:

- The split validates the main learning claim better than aggregate reward. In learned-only test, balanced builds on positive and mixed maps but almost completely suppresses learned road building on negative maps.
- Loose is over-permissive: it builds 21 roads in every context, including negative maps, where total road payoff becomes negative.
- Balanced is not merely "building less"; it preserves most positive-road hits while dropping many negative-road builds.
- Off-route builds are zero because the learned evidence gate requires an on-route opportunity. The remaining negative payoffs are on-route roads that did not get enough reuse, not random off-route paving.

Current conclusion: `llm_with_road_learning_balanced_threshold` is the first profile that clearly separates useful transport-road contexts from mostly misleading rough contexts under learned-only test.
