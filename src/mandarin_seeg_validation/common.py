from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_time_by_channel(path: Path, time_axis: int = 0) -> np.ndarray:
    """Load a 2-D NPY array and return it as time x channels."""
    # Ordinary read-only loading avoids persistent Windows file locks from
    # NumPy memory maps when many short trial files are processed in sequence.
    data = np.load(path, allow_pickle=False)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D NPY array, got shape {data.shape}")
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1")
    if time_axis == 1:
        data = data.T
    return data


def parse_pair(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{name} must contain two increasing values")
    return float(values[0]), float(values[1])


def save_figure(fig: plt.Figure, base: Path, formats: tuple[str, ...]) -> None:
    for extension in formats:
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(base.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

