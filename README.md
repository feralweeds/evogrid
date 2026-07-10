# EvoGrid-Mine

Minimal experimental code for environment-shaping agents in a mutable 2D mining world.

The first implementation focuses on:

- a deterministic 32 x 32 grid environment;
- Random, Greedy, DeepSeek, and Hybrid agents;
- rollout metrics for environment shaping;
- smoke tests before adding full PPO training.

## Research Status

The project now has two experiment tracks:

- `full_obs`: a full-info debugging baseline for validating the environment, metrics, DeepSeek API calls, plots, and rollout traces.
- `partial_obs`: the first self-evolution loop, where the agent receives only local observations plus its own memory/reflection summaries.

For the self-evolution research question, `full_obs` is not enough: if an agent starts with the ore coordinates and the full map, the task becomes mostly path planning.

The current self-evolution track is:

```text
partial observation + agent memory + episode reflection
```

In this setting, the agent does not receive global ore positions or the full map at the start. It should discover resources through local observations, store what it has seen, reflect after each episode, and use that memory in later episodes.

Do not hard-code road-building or digging rules as the answer. Generic capabilities such as action legality checks, local observation construction, memory updates, and reflection summaries are allowed.

## Quick Start

```bash
python scripts/smoke_test_env.py
python scripts/smoke_test_deepseek_agent.py
python -m unittest discover tests
```

DeepSeek API keys should be provided through environment variables, not committed files.

## DeepSeek Agent

Offline smoke test:

```bash
python scripts/run_deepseek_agent.py --episodes 1 --max-steps 5 --mock-ok
```

Real API smoke test:

```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
python scripts/run_deepseek_agent.py --episodes 1 --max-steps 20 --require-api
```

## Unified First Experiment

Include the DeepSeek hybrid group in the unified comparison:

```bash
python scripts/run_first_experiment.py --groups full_shaping no_shaping random greedy hybrid_deepseek_greedy --eval-episodes 5 --seeds 0
```

For offline smoke tests without calling the API:

```bash
python scripts/run_first_experiment.py --groups hybrid_deepseek_greedy random greedy --eval-episodes 1 --seeds 0 --deepseek-max-steps 5 --mock-deepseek
```

DeepSeek calls print one terminal line per LLM decision by default. Add `--quiet-llm-calls` to silence those lines.

## Self-Evolution

Offline smoke test:

```bash
python scripts/run_self_evolution_experiment.py --episodes 2 --max-steps 20 --replan-interval 10 --mock-deepseek --out outputs/runs/self_evolution_smoke
```

Partial-observation non-LLM baseline:

```bash
python scripts/run_self_evolution_experiment.py --agent partial_greedy --episodes 5 --max-steps 500 --out outputs/runs/partial_greedy_pilot
```

Real DeepSeek pilot:

```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
python scripts/run_self_evolution_experiment.py --episodes 5 --max-steps 500 --require-api --out outputs/runs/self_evolution_pilot
```

Expected outputs:

```text
outputs/runs/self_evolution_*/
├── memory.json
├── metrics.csv
├── llm_trace.jsonl
├── route_trace.jsonl
├── step_trace.jsonl
├── summary.json
└── episodes/
```

DeepSeek calls print one terminal line per LLM decision by default. Add `--quiet-llm-calls` to silence those lines.

Useful diagnostic fields:

```text
num_mine
final_has_ore
final_agent_pos
first_ore_seen_step
first_mine_step
carrying_steps
known_ore_locations
route_trace.jsonl
step_trace.jsonl
```

`MemoryMapRoutePlanner` is used only as a low-level navigator during return-to-base behavior. It plans over the agent's own observed memory map, not over the hidden environment grid.

Current next research step: road-building should be learned through experience, not hard-coded as a rule. See `接下来工作计划.md`, section 8, for the RoadCreditTracker / RoadLearningModule / ShapingOpportunity plan.

Road-building sanity check:

```bash
python scripts/run_road_oracle_sanity.py --episodes 3 --out outputs/runs/road_oracle_sanity
```

`rule_road_oracle` is a temporary full-observation upper-bound check. It should not be treated as the final self-evolution policy.
In the current road sanity map, roads improve movement payoff and episode reward, but `BUILD_ROAD` still consumes steps, so compare `road_net_payoff` and `episode_reward` alongside `ore_delivered`.
`ShapingOpportunity` uses `candidate_action` for non-binding road-building evidence; it does not auto-execute `BUILD_ROAD`.
`RoadLearningModule` turns past road payoff records into `learned_estimate` values; it does not choose or execute actions.

Road-learning ablation:

```bash
python scripts/run_road_learning_ablation.py --episodes 3 --out outputs/runs/road_learning_ablation
```

This compares `no_shaping`, `route_only`, `rule_road_oracle`, and `learned_road` under both cold-start and warm-start settings. The default config is `configs/env_road_learning.yaml`, a small partial-observation rough-corridor curriculum for checking whether `learned_estimate` changes road-building behavior and produces positive road payoff. Only the oracle baseline and warm-start sampler are forced to full observation.

Current boundary: `learned_road` is a learned-threshold agent, not the final end-to-end autonomous learning agent. It verifies that historical road payoff estimates can drive `BUILD_ROAD`; the next step is exploration-based road learning, where road records come from the agent's own controlled exploration instead of oracle warm-start samples.

Exploration-based road learning:

```bash
python scripts/run_exploration_road_learning.py --episodes 20 --out outputs/runs/exploration_road_learning
```

This runs without oracle warm-start records, but it does not use an LLM. `exploration_road` is a non-LLM exploration-threshold baseline: it creates road payoff samples through capped epsilon / uncertainty exploration, then uses prior-episode records through `RoadLearningModule` so later `BUILD_ROAD` decisions can come from learned estimates. It validates the learning mechanism, not the final claim that an LLM agent autonomously learned road-building.

The next research step is an LLM-mediated version where the LLM receives `ShapingOpportunity`, `learned_estimate`, uncertainty, route context, and an exploration budget, then chooses whether to `BUILD_ROAD` itself.

LLM-mediated road learning:

```bash
python scripts/run_llm_road_learning.py --mock-deepseek --quiet-llm-calls --episodes 6 --out outputs/runs/llm_road_learning_mock
```

The mock run validates the LLM decision interface and metrics only. To test the actual DeepSeek agent:

```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
python scripts/run_llm_road_learning.py --require-api --episodes 3 --groups route_only exploration_threshold llm_no_road_learning llm_with_road_learning --out outputs/runs/llm_road_learning_real_pilot
```

Treat real LLM evidence as valid only if `llm_with_road_learning` builds more often when `learned_value > 0`, its rationale cites learned payoff evidence, and road payoff remains positive.

## Next Research Route

See [`docs/下一阶段实验路线.md`](docs/下一阶段实验路线.md) for the next-stage plan: randomized map curriculum, train/test generalization evaluation, non-stationary environment shifts, stricter baselines, and success criteria for claiming adaptive learning in changing environments.

Randomized-map baseline generalization smoke test:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_random_curriculum.yaml --train-seeds 0:2 --test-seeds 1000:1002 --episodes-per-seed 1 --max-steps 60 --groups route_only random_road rough_rule_road exploration_road --out outputs/runs/generalization_eval_baseline_smoke
```

This runs the non-LLM baseline gate before LLM testing. The outputs are `metrics.csv`, `group_comparison.csv`, and `summary.json`.

Controlled corridor curriculum smoke test:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:3 --test-seeds 1000:1003 --episodes-per-seed 1 --max-steps 120 --groups route_only random_road rough_rule_road exploration_road --out outputs/runs/controlled_corridor_baseline_smoke
```

`configs/env_controlled_corridor_curriculum.yaml` samples positive, mixed, and negative road-learning maps. It is intentionally not the old fixed corridor: base position, ore position, rough transport bands, route roughness, off-route rough distractors, and obstacles are seeded-random. Use `route_rough_tile_count`, `off_route_rough_tile_count`, `positive_road_opportunity_count`, `road_net_payoff`, and `positive_road_ratio` to check whether the curriculum contains both useful and misleading road-building opportunities. See [`docs/controlled_corridor_curriculum.md`](docs/controlled_corridor_curriculum.md).

Mock LLM generalization smoke test:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_random_curriculum.yaml --train-seeds 0:2 --test-seeds 1000:1002 --episodes-per-seed 1 --max-steps 60 --groups route_only exploration_road llm_no_road_learning llm_with_road_learning --mock-deepseek --quiet-llm-calls --out outputs/runs/generalization_eval_llm_mock_smoke
```

Controlled corridor mock LLM smoke test:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:3 --test-seeds 1000:1003 --episodes-per-seed 1 --max-steps 120 --groups route_only exploration_road llm_no_road_learning llm_with_road_learning --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --out outputs/runs/controlled_corridor_llm_mock_smoke
```

Road payoff records are enriched with build-time route context, so `RoadLearningModule` now prefers contextual estimates over legacy tile-only estimates. A larger contextual mock run is:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --max-steps 200 --groups route_only exploration_road llm_no_road_learning llm_with_road_learning --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --out outputs/runs/controlled_corridor_contextual_20x2
```

Current interpretation: contextual learning fixes the worst tile-level overgeneralization, but the mock `llm_with_road_learning` has not yet beaten `llm_no_road_learning`; it mostly changes road builds from exploration-sourced to learned-sourced. See [`docs/controlled_corridor_curriculum.md`](docs/controlled_corridor_curriculum.md).

Evidence-threshold ablation:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --max-steps 200 --groups llm_with_road_learning_no_threshold llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold llm_with_road_learning_medium_threshold llm_with_road_learning_strict_threshold --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --out outputs/runs/controlled_corridor_threshold_ablation_20x2
```

The balanced/medium/strict profiles require contextual evidence, stronger positive rate, positive mean payoff, confidence, and an on-route opportunity before learned evidence can trigger `BUILD_ROAD`.

Learned-only test, with training exploration enabled and test exploration disabled:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --max-steps 200 --groups route_only llm_no_road_learning llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold llm_with_road_learning_medium_threshold llm_with_road_learning_strict_threshold --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --test-exploration-budget 0 --out outputs/runs/controlled_corridor_learned_only_balanced_20x2
```

Current learned-only result: balanced transfers learned road evidence into test-time builds with positive road net payoff while building less than loose. It is the current best mock candidate for the next pilot.

Evidence-density scaling can increase training episodes without increasing test episodes:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --train-episodes-per-seed 4 --test-episodes-per-seed 1 --max-steps 200 --groups route_only llm_no_road_learning llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold llm_with_road_learning_medium_threshold llm_with_road_learning_strict_threshold --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --test-exploration-budget 0 --out outputs/runs/controlled_corridor_evidence_density_20x4_train_20x1_test
```

Current evidence-density result: balanced remains the best mock gate. In test it built 10 learned roads with 8 positive payoffs and road net `4.00`; loose built 21 roads with only 8 positive payoffs and road net `2.05`. Medium/strict still produced no strong evidence, so the next step is a positive/negative context split rather than immediately scaling real LLM calls.

Positive/negative context split:

```bash
python scripts/run_generalization_eval.py --env-config configs/env_controlled_corridor_curriculum.yaml --train-seeds 0:20 --test-seeds 1000:1020 --episodes-per-seed 2 --train-episodes-per-seed 4 --test-episodes-per-seed 1 --max-steps 200 --groups route_only llm_no_road_learning llm_with_road_learning_loose_threshold llm_with_road_learning_balanced_threshold --test-context-scenarios positive negative mixed --mock-deepseek --quiet-llm-calls --max-learned-builds-per-episode 3 --test-exploration-budget 0 --out outputs/runs/controlled_corridor_context_split_20x4_train_20x1_test
```

Current context-split result: balanced separates useful and misleading contexts under learned-only test. On positive maps it built 18 learned roads with 15 positive payoffs and road net `8.30`; on mixed maps it built 11 learned roads with 8 positive payoffs and road net `5.10`; on negative maps it built only 1 learned road. Loose built 21 learned roads in every context and went negative on negative maps.

For a real DeepSeek pilot, replace `--mock-deepseek` with `--require-api` after setting `DEEPSEEK_API_KEY`.
