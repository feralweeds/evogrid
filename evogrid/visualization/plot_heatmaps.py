"""Heatmap plotting helpers."""

from __future__ import annotations

from pathlib import Path


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for heatmaps.") from exc
    return plt


def save_heatmap(
    matrix: list[list[float]] | list[list[int]],
    output_path: str | Path,
    title: str,
    cmap: str = "viridis",
) -> Path:
    plt = require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, shrink=0.78)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def counts_to_matrix(counts: dict[tuple[int, int], int], height: int, width: int) -> list[list[int]]:
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    for (row, col), value in counts.items():
        if 0 <= row < height and 0 <= col < width:
            matrix[row][col] = int(value)
    return matrix


def cells_to_matrix(cells: set[tuple[int, int]], height: int, width: int) -> list[list[int]]:
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    for row, col in cells:
        if 0 <= row < height and 0 <= col < width:
            matrix[row][col] = 1
    return matrix
