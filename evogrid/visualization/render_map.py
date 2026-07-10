"""ASCII rendering helpers."""

from __future__ import annotations

from pathlib import Path

from evogrid.constants import TILE_CHARS, Tile


def render_grid(grid: list[list[int]], agent_pos: tuple[int, int] | None = None) -> str:
    rows = []
    for row_idx, row in enumerate(grid):
        chars = []
        for col_idx, value in enumerate(row):
            if agent_pos == (row_idx, col_idx):
                chars.append("A")
            else:
                chars.append(TILE_CHARS[Tile(value)])
        rows.append("".join(chars))
    return "\n".join(rows)


def save_ascii_map(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def save_map_image(
    grid: list[list[int]],
    output_path: str | Path,
    agent_pos: tuple[int, int] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = [
        "#F8FAFC",  # ground
        "#2563EB",  # base
        "#22C55E",  # ore
        "#111827",  # obstacle
        "#F59E0B",  # rough
        "#A855F7",  # road
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, norm=norm)
    if agent_pos is not None:
        ax.scatter([agent_pos[1]], [agent_pos[0]], c="#EF4444", s=40, marker="o", edgecolors="white", linewidths=0.8)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_before_after_image(
    before_grid: list[list[int]],
    after_grid: list[list[int]],
    output_path: str | Path,
    title: str = "Before / After Map",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#F8FAFC", "#2563EB", "#22C55E", "#111827", "#F59E0B", "#A855F7"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, grid, label in zip(axes, [before_grid, after_grid], ["Before", "After"]):
        ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
