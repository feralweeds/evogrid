from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import _bootstrap  # noqa: F401

from evogrid.constants import Tile
from evogrid.envs.map_builder import build_fixed_map
from evogrid.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "weekly_report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "bg": "#f8fafc",
    "ink": "#111827",
    "muted": "#4b5563",
    "line": "#94a3b8",
    "blue": "#2563eb",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "cyan": "#0891b2",
    "card": "#ffffff",
}

TILE_COLORS = {
    int(Tile.GROUND): "#f8fafc",
    int(Tile.BASE): "#2563eb",
    int(Tile.ORE): "#f59e0b",
    int(Tile.OBSTACLE): "#111827",
    int(Tile.ROUGH): "#a16207",
    int(Tile.ROAD): "#64748b",
}

TILE_LABELS = {
    int(Tile.BASE): "基",
    int(Tile.ORE): "矿",
    int(Tile.OBSTACLE): "障",
    int(Tile.ROUGH): "糙",
    int(Tile.ROAD): "路",
}


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_TITLE = font(34, True)
FONT_H = font(24, True)
FONT = font(20)
FONT_SMALL = font(16)
FONT_TINY = font(13)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def rounded_box(draw, xy, fill, outline="#cbd5e1", radius=16, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, xy, text, fnt=FONT, fill=COLORS["ink"], spacing=5):
    x1, y1, x2, y2 = xy
    lines = text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, height in zip(lines, heights):
        width, _ = text_size(draw, line, fnt)
        draw.text((x1 + ((x2 - x1) - width) / 2, y), line, font=fnt, fill=fill)
        y += height + spacing


def arrow(draw, start, end, color=COLORS["line"], width=4):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 14
    p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)


def save_platform_architecture():
    im = Image.new("RGB", (1600, 900), COLORS["bg"])
    draw = ImageDraw.Draw(im)
    draw.text((60, 40), "EvoGrid-Mine 实验平台", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text(
        (60, 86),
        "可改造网格环境 + 跨回合记忆 + 道路收益学习 + 仅学习证据测试",
        font=FONT,
        fill=COLORS["muted"],
    )
    boxes = {
        "config": (80, 180, 360, 310),
        "env": (500, 160, 860, 340),
        "agent": (1040, 160, 1460, 340),
        "memory": (1020, 430, 1460, 590),
        "eval": (500, 520, 860, 700),
        "outputs": (80, 540, 360, 690),
    }
    labels = {
        "config": "配置文件\n地图 / 奖励 / 大模型 / 实验",
        "env": "采矿网格环境\n局部观测，可改造地图\n挖掘 / 修路",
        "agent": "智能体与基线\n只走路、探索修路\n大模型道路学习",
        "memory": "智能体记忆\n已见地形、访问次数\n道路收益记录",
        "eval": "泛化评估\n训练/测试种子分离\n正/负/混合场景",
        "outputs": "实验输出\n指标表、摘要\n轨迹记录",
    }
    fills = {
        "config": "#e0f2fe",
        "env": "#dcfce7",
        "agent": "#ede9fe",
        "memory": "#fef3c7",
        "eval": "#fee2e2",
        "outputs": "#f1f5f9",
    }
    for key, xy in boxes.items():
        rounded_box(draw, xy, fill=fills[key])
        centered_text(draw, xy, labels[key], FONT)

    arrow(draw, (360, 245), (500, 245), COLORS["blue"])
    arrow(draw, (860, 245), (1040, 245), COLORS["green"])
    arrow(draw, (1250, 340), (1250, 430), COLORS["purple"])
    arrow(draw, (1020, 510), (860, 610), COLORS["amber"])
    arrow(draw, (500, 610), (360, 615), COLORS["red"])
    arrow(draw, (680, 520), (680, 340), COLORS["cyan"])
    draw.text((520, 370), "动作-环境-反馈循环", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((1065, 610), "学习状态跨回合复用", font=FONT_SMALL, fill=COLORS["muted"])
    im.save(ASSETS / "platform_architecture.png")


def save_road_learning_pipeline():
    im = Image.new("RGB", (1700, 760), COLORS["bg"])
    draw = ImageDraw.Draw(im)
    draw.text((60, 40), "道路学习决策链路", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text(
        (60, 86),
        "训练阶段探索产生证据；测试阶段可以关闭探索，只使用学习证据修路。",
        font=FONT,
        fill=COLORS["muted"],
    )
    xs = [70, 335, 600, 865, 1130, 1395]
    y = 220
    width = 225
    height = 170
    labels = [
        "执行修路\n记录轨迹",
        "道路收益归因\n使用次数、节省成本\n净收益",
        "补充上下文\n是否在路线上\n访问次数",
        "道路学习模块\n估计价值、置信度\n正收益比例",
        "证据门槛\n阈值 +\n未来复用回本",
        "智能体 / 大模型\n修路或跳过",
    ]
    fills = ["#fef3c7", "#dcfce7", "#e0f2fe", "#ede9fe", "#fee2e2", "#f1f5f9"]
    for index, x in enumerate(xs):
        box = (x, y, x + width, y + height)
        rounded_box(draw, box, fill=fills[index])
        centered_text(draw, box, labels[index], FONT_SMALL)
        if index < len(xs) - 1:
            arrow(draw, (x + width, y + height // 2), (xs[index + 1] - 20, y + height // 2), COLORS["line"], 3)

    draw.text((90, 500), "本周新增的关键门槛", font=FONT_H, fill=COLORS["ink"])
    bullets = [
        "最小上下文证据数",
        "正收益比例 / 平均收益 / 置信度门槛",
        "仅学习证据测试：测试阶段探索预算 = 0",
        "预计未来复用次数 >= 回本次数 + 安全余量",
    ]
    y_pos = 545
    for bullet in bullets:
        draw.ellipse((105, y_pos + 7, 115, y_pos + 17), fill=COLORS["blue"])
        draw.text((130, y_pos), bullet, font=FONT, fill=COLORS["ink"])
        y_pos += 38
    im.save(ASSETS / "road_learning_pipeline.png")


def force_scenario(config, scenario):
    cfg = copy.deepcopy(config)
    corridor = cfg.setdefault("env", {}).setdefault("random_map", {}).setdefault("controlled_corridor", {})
    for name in ("positive", "mixed", "negative"):
        corridor[f"{name}_weight"] = 1.0 if name == scenario else 0.0
    return cfg


def draw_grid(draw, grid, top_left, cell=24):
    x0, y0 = top_left
    for row_index, row in enumerate(grid):
        for col_index, tile in enumerate(row):
            x = x0 + col_index * cell
            y = y0 + row_index * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=TILE_COLORS.get(int(tile), "#ffffff"), outline="#cbd5e1")
            if int(tile) in TILE_LABELS:
                label = TILE_LABELS[int(tile)]
                text_width, text_height = text_size(draw, label, FONT_TINY)
                fill = "#ffffff" if int(tile) in {int(Tile.BASE), int(Tile.OBSTACLE)} else COLORS["ink"]
                draw.text((x + (cell - text_width) / 2, y + (cell - text_height) / 2 - 1), label, font=FONT_TINY, fill=fill)


def save_controlled_corridor_examples():
    config = load_yaml(ROOT / "configs" / "env_controlled_corridor_curriculum.yaml")
    im = Image.new("RGB", (1600, 760), COLORS["bg"])
    draw = ImageDraw.Draw(im)
    draw.text((60, 35), "受控随机地图课程：三类示例", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text(
        (60, 82),
        "同一套随机生成器，通过不同场景权重控制正样本、负样本和混合样本；场景类型不暴露给智能体。",
        font=FONT,
        fill=COLORS["muted"],
    )
    scenarios = ["positive", "mixed", "negative"]
    names = {"positive": "正样本", "mixed": "混合样本", "negative": "负样本"}
    descriptions = {
        "positive": "主运输路径上有粗糙走廊",
        "mixed": "路线粗糙 + 支路干扰",
        "negative": "粗糙地形多在非路线区域",
    }
    xs = [80, 590, 1100]
    for index, scenario in enumerate(scenarios):
        grid, _, _ = build_fixed_map(force_scenario(config, scenario), seed=1000 + index)
        draw.text((xs[index], 145), names[scenario], font=FONT_H, fill=COLORS["ink"])
        draw.text((xs[index], 175), descriptions[scenario], font=FONT_SMALL, fill=COLORS["muted"])
        draw_grid(draw, grid, (xs[index], 215), cell=24)

    legend_y = 655
    legend = [("基地", Tile.BASE), ("矿", Tile.ORE), ("障碍", Tile.OBSTACLE), ("粗糙", Tile.ROUGH), ("普通地面", Tile.GROUND)]
    x = 80
    for name, tile in legend:
        draw.rectangle((x, legend_y, x + 24, legend_y + 24), fill=TILE_COLORS[int(tile)], outline="#94a3b8")
        draw.text((x + 34, legend_y + 1), name, font=FONT_SMALL, fill=COLORS["ink"])
        x += 170
    im.save(ASSETS / "controlled_corridor_examples.png")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except Exception:
        return 0.0


def grouped_bar_chart(path, title, subtitle, contexts, series, metric, ylabel, out_name, value_fmt="{:.2f}", show_legend=True):
    im = Image.new("RGB", (1500, 900), COLORS["bg"])
    draw = ImageDraw.Draw(im)
    draw.text((60, 35), title, font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((60, 82), subtitle, font=FONT, fill=COLORS["muted"])
    rows = read_csv(path)
    lookup = {(row.get("context_scenario"), row.get("group")): row for row in rows if row.get("phase") == "test"}
    x1, y1, x2, y2 = 110, 160, 1420, 690
    draw.line((x1, y2, x2, y2), fill="#64748b", width=2)
    draw.line((x1, y1, x1, y2), fill="#64748b", width=2)
    values = [value(lookup.get((context, group), {}), metric) for context in contexts for _, group, _ in series]
    max_value = max(values + [0.01])
    min_value = min(values + [0.0])
    span = max_value - min_value if max_value != min_value else 1.0

    def y_for(number):
        return y2 - ((number - min_value) / span) * (y2 - y1 - 25)

    zero_y = y_for(0.0)
    draw.line((x1, zero_y, x2, zero_y), fill="#cbd5e1", width=2)
    draw.text((35, y1 + 10), ylabel, font=FONT_SMALL, fill=COLORS["muted"])
    context_width = (x2 - x1) / len(contexts)
    bar_width = 72
    context_names = {"positive": "正样本", "mixed": "混合", "negative": "负样本"}
    for index, context in enumerate(contexts):
        center = x1 + context_width * (index + 0.5)
        context_label = context_names.get(context, context)
        label_width, _ = text_size(draw, context_label, FONT)
        draw.text((center - label_width / 2, y2 + 28), context_label, font=FONT, fill=COLORS["ink"])
        for series_index, (_, group, color) in enumerate(series):
            number = value(lookup.get((context, group), {}), metric)
            bar_x = center + (series_index - (len(series) - 1) / 2) * (bar_width + 22) - bar_width / 2
            bar_y = y_for(number)
            rect = (bar_x, bar_y, bar_x + bar_width, zero_y) if number >= 0 else (bar_x, zero_y, bar_x + bar_width, bar_y)
            draw.rectangle(rect, fill=color)
            text = value_fmt.format(number)
            text_width, _ = text_size(draw, text, FONT_TINY)
            draw.text((bar_x + (bar_width - text_width) / 2, min(bar_y, zero_y) - 22), text, font=FONT_TINY, fill=COLORS["ink"])

    if show_legend:
        legend_x = 1020
        legend_y = 790
        for label, _, color in series:
            draw.rectangle((legend_x, legend_y, legend_x + 28, legend_y + 18), fill=color)
            draw.text((legend_x + 38, legend_y - 2), label, font=FONT_SMALL, fill=COLORS["ink"])
            legend_x += 210
    im.save(ASSETS / out_name)


def save_context_charts():
    mock_path = ROOT / "outputs" / "runs" / "controlled_corridor_context_split_20x4_train_20x1_test" / "context_comparison.csv"
    deepseek_path = ROOT / "outputs" / "runs" / "deepseek_train_policy_override_context_split_pilot" / "context_comparison.csv"
    grouped_bar_chart(
        mock_path,
        "模拟实验：仅学习证据测试",
        "均衡门槛能保留有效修路，同时抑制负样本中过度修路。",
        ["positive", "mixed", "negative"],
        [
            ("宽松门槛", "llm_with_road_learning_loose_threshold", COLORS["amber"]),
            ("均衡门槛", "llm_with_road_learning_balanced_threshold", COLORS["blue"]),
        ],
        "total_positive_road_ratio",
        "正收益道路比例",
        "mock_context_split_positive_ratio.png",
    )
    grouped_bar_chart(
        mock_path,
        "模拟实验：道路净收益",
        "负样本中，宽松门槛会过度修路；均衡门槛几乎停止无效修路。",
        ["positive", "mixed", "negative"],
        [
            ("宽松门槛", "llm_with_road_learning_loose_threshold", COLORS["amber"]),
            ("均衡门槛", "llm_with_road_learning_balanced_threshold", COLORS["blue"]),
        ],
        "road_net_payoff_sum",
        "道路净收益总和",
        "mock_context_split_road_net.png",
    )
    grouped_bar_chart(
        deepseek_path,
        "真实 DeepSeek：仅学习证据试跑",
        "测试阶段探索预算为 0；正样本和混合样本中仍出现学习修路，负样本中较少。",
        ["positive", "mixed", "negative"],
        [("学习修路", "llm_with_road_learning_balanced_threshold", COLORS["purple"])],
        "llm_learned_build_count_sum",
        "学习触发修路次数",
        "deepseek_pilot_learned_builds.png",
        "{:.0f}",
        show_legend=False,
    )
    grouped_bar_chart(
        deepseek_path,
        "真实 DeepSeek：道路净收益",
        "三个测试场景道路净收益均为正，但总奖励在正/混合场景仍落后于只走路基线。",
        ["positive", "mixed", "negative"],
        [("道路学习", "llm_with_road_learning_balanced_threshold", COLORS["green"])],
        "road_net_payoff_sum",
        "道路净收益总和",
        "deepseek_pilot_road_net.png",
        "{:.2f}",
        show_legend=False,
    )


def save_future_use_smoke():
    path = ROOT / "outputs" / "runs" / "controlled_corridor_future_use_gate_smoke_v2" / "group_comparison.csv"
    rows = read_csv(path)
    lookup = {row["group"]: row for row in rows}
    groups = [
        ("均衡", "llm_with_road_learning_balanced_threshold", COLORS["blue"]),
        ("未来复用", "llm_with_road_learning_balanced_future_use_threshold", COLORS["green"]),
    ]
    metrics = [
        ("学习修路次数", "test_llm_learned_build_count_mean", "{:.2f}"),
        ("道路净收益", "test_road_net_payoff_mean", "{:.3f}"),
        ("单路平均收益", "test_avg_payoff_per_road_mean", "{:.3f}"),
    ]
    im = Image.new("RGB", (1500, 800), COLORS["bg"])
    draw = ImageDraw.Draw(im)
    draw.text((60, 35), "未来复用 / 回本门槛：小规模验证", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((60, 82), "新门槛减少学习修路，并把道路收益从略负推到略正。", font=FONT, fill=COLORS["muted"])
    panels = [(90, 170, 480, 620), (555, 170, 945, 620), (1020, 170, 1410, 620)]
    for panel, (metric_label, metric_key, value_format) in zip(panels, metrics):
        x1, y1, x2, y2 = panel
        rounded_box(draw, panel, fill=COLORS["card"], radius=10, width=1)
        draw.text((x1 + 25, y1 + 20), metric_label, font=FONT_H, fill=COLORS["ink"])
        numbers = [value(lookup[group], metric_key) for _, group, _ in groups]
        min_value = min(numbers + [0.0])
        max_value = max(numbers + [0.01])
        span = max_value - min_value if max_value != min_value else 1.0
        base_y = y2 - 75
        top_y = y1 + 95

        def y_for(number):
            return base_y - ((number - min_value) / span) * (base_y - top_y)

        zero_y = y_for(0.0)
        draw.line((x1 + 40, zero_y, x2 - 40, zero_y), fill="#cbd5e1", width=2)
        for index, (short_label, group, color) in enumerate(groups):
            number = value(lookup[group], metric_key)
            bar_x = x1 + 95 + index * 135
            bar_y = y_for(number)
            rect = (bar_x, bar_y, bar_x + 80, zero_y) if number >= 0 else (bar_x, zero_y, bar_x + 80, bar_y)
            draw.rectangle(rect, fill=color)
            text = value_format.format(number)
            text_width, _ = text_size(draw, text, FONT_SMALL)
            draw.text((bar_x + (80 - text_width) / 2, min(bar_y, zero_y) - 28), text, font=FONT_SMALL, fill=COLORS["ink"])
            label_width, _ = text_size(draw, short_label, FONT_SMALL)
            draw.text((bar_x + (80 - label_width) / 2, base_y + 25), short_label, font=FONT_SMALL, fill=COLORS["muted"])
    im.save(ASSETS / "future_use_gate_smoke.png")


def save_latest_deepseek_future_use_charts():
    latest_path = ROOT / "outputs" / "runs" / "deepseek_future_use_gate_context_split_pilot" / "context_comparison.csv"
    if not latest_path.exists():
        return
    grouped_bar_chart(
        latest_path,
        "最新真实 DeepSeek：future-use gate 上下文试跑",
        "balanced 会在测试阶段触发少量学习修路；future-use gate 在本轮真实试跑中过于保守，测试阶段未触发 learned build。",
        ["positive", "mixed", "negative"],
        [
            ("balanced", "llm_with_road_learning_balanced_threshold", COLORS["blue"]),
            ("future-use", "llm_with_road_learning_balanced_future_use_threshold", COLORS["green"]),
        ],
        "llm_learned_build_count_sum",
        "学习证据触发修路次数",
        "latest_deepseek_future_use_learned_builds.png",
        "{:.0f}",
    )
    grouped_bar_chart(
        latest_path,
        "最新真实 DeepSeek：道路净收益",
        "balanced 在三个测试上下文中均产生正道路净收益；future-use gate 因没有修路，净收益为 0。",
        ["positive", "mixed", "negative"],
        [
            ("balanced", "llm_with_road_learning_balanced_threshold", COLORS["blue"]),
            ("future-use", "llm_with_road_learning_balanced_future_use_threshold", COLORS["green"]),
        ],
        "road_net_payoff_sum",
        "道路净收益总和",
        "latest_deepseek_future_use_road_net.png",
        "{:.2f}",
    )


def main():
    save_platform_architecture()
    save_road_learning_pipeline()
    save_controlled_corridor_examples()
    save_context_charts()
    save_future_use_smoke()
    save_latest_deepseek_future_use_charts()
    sources = {
        "platform_architecture.png": "根据当前代码结构绘制的平台架构图",
        "road_learning_pipeline.png": "根据道路学习模块和证据门槛绘制",
        "controlled_corridor_examples.png": "由 configs/env_controlled_corridor_curriculum.yaml 通过 map_builder 生成",
        "mock_context_split_positive_ratio.png": "outputs/runs/controlled_corridor_context_split_20x4_train_20x1_test/context_comparison.csv",
        "mock_context_split_road_net.png": "outputs/runs/controlled_corridor_context_split_20x4_train_20x1_test/context_comparison.csv",
        "deepseek_pilot_learned_builds.png": "outputs/runs/deepseek_train_policy_override_context_split_pilot/context_comparison.csv",
        "deepseek_pilot_road_net.png": "outputs/runs/deepseek_train_policy_override_context_split_pilot/context_comparison.csv",
        "future_use_gate_smoke.png": "outputs/runs/controlled_corridor_future_use_gate_smoke_v2/group_comparison.csv",
        "latest_deepseek_future_use_learned_builds.png": "outputs/runs/deepseek_future_use_gate_context_split_pilot/context_comparison.csv",
        "latest_deepseek_future_use_road_net.png": "outputs/runs/deepseek_future_use_gate_context_split_pilot/context_comparison.csv",
    }
    (ASSETS / "figure_sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(sources)} Chinese figures to {ASSETS}")


if __name__ == "__main__":
    main()
