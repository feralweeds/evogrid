"""Episode metric helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from evogrid.envs.map_state import MapState


@dataclass
class EpisodeMetrics:
    episode_reward: float
    ore_delivered: int
    num_mine: int
    num_dig: int
    num_build_road: int
    invalid_actions: int
    road_cells_built: int
    dug_cells: int
    road_usage_rate: float
    road_total_usage_count: int
    road_saved_cost: float
    road_build_cost: float
    road_net_payoff: float
    positive_road_payoff_count: int
    negative_road_payoff_count: int
    road_credit_records: list[dict]
    transport_steps_per_ore: float
    early_shaping_ratio: float
    late_delivery_rate: float
    steps: int
    final_has_ore: bool
    final_agent_pos: list[int]
    carrying_steps: int
    first_mine_step: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def collect_metrics(state: MapState, early_window: int = 100) -> EpisodeMetrics:
    road_cells_built = len(state.built_roads)
    road_usage_rate = len(state.road_visited) / road_cells_built if road_cells_built else 0.0
    road_credit_summary = state.road_credit_tracker.summary()

    if state.transport_steps:
        transport_steps_per_ore = sum(state.transport_steps) / len(state.transport_steps)
    else:
        transport_steps_per_ore = 0.0

    early_actions = state.action_history[:early_window]
    early_shaping = sum(1 for step in state.shaping_action_steps if step < early_window)
    early_shaping_ratio = early_shaping / len(early_actions) if early_actions else 0.0

    half_step = max(1, state.step_count // 2)
    late_deliveries = sum(1 for steps in state.transport_steps if steps >= 0)
    late_delivery_rate = late_deliveries / max(1, state.step_count - half_step)

    return EpisodeMetrics(
        episode_reward=state.total_reward,
        ore_delivered=state.ore_delivered,
        num_mine=state.num_mine,
        num_dig=state.num_dig,
        num_build_road=state.num_build_road,
        invalid_actions=state.invalid_actions,
        road_cells_built=road_cells_built,
        dug_cells=len(state.dug_cells),
        road_usage_rate=road_usage_rate,
        road_total_usage_count=int(road_credit_summary["road_total_usage_count"]),
        road_saved_cost=float(road_credit_summary["road_saved_cost"]),
        road_build_cost=float(road_credit_summary["road_build_cost"]),
        road_net_payoff=float(road_credit_summary["road_net_payoff"]),
        positive_road_payoff_count=int(road_credit_summary["positive_road_payoff_count"]),
        negative_road_payoff_count=int(road_credit_summary["negative_road_payoff_count"]),
        road_credit_records=state.road_credit_tracker.to_records(),
        transport_steps_per_ore=transport_steps_per_ore,
        early_shaping_ratio=early_shaping_ratio,
        late_delivery_rate=late_delivery_rate,
        steps=state.step_count,
        final_has_ore=state.has_ore,
        final_agent_pos=list(state.agent_pos),
        carrying_steps=state.current_transport_steps if state.has_ore else 0,
        first_mine_step=state.mine_steps[0] if state.mine_steps else None,
    )
