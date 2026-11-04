from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


CHANCE = 0.25
TONE_LABELS = ["T1", "T2", "T3", "T4"]
COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "orange": "#E07A3F",
    "teal": "#2A9D8F",
    "gray": "#7A8793",
    "grid": "#E7EBEF",
    "white": "#FFFFFF",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    summary = read_json(root / "cv_summary.json")
    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for fold_result in summary["folds"]:
        if "metrics" not in fold_result:
            continue
        fold = int(fold_result["fold"])
        metrics = fold_result["metrics"]["tone"]
        fold_rows.append(
            {
                "fold": fold,
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
        for subject, subject_metrics in fold_result["metrics"]["by_subject"].items():
            tone = subject_metrics["tone"]
            subject_rows.append(
                {
                    "fold": fold,
                    "subject": subject,
                    "n_test": int(tone["trials"]),
                    "balanced_accuracy": float(tone["balanced_accuracy"]),
                }
            )
        payload = np.load(root / f"fold-{fold:02d}" / "predictions.npz", allow_pickle=False)
        targets.append(np.asarray(payload["tone_target"], dtype=int))
        predictions.append(np.asarray(payload["tone_prediction"], dtype=int))
    if len(fold_rows) != 5:
        raise ValueError(f"Expected five completed folds under {root}, found {len(fold_rows)}")
    return (
        pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True),
        pd.DataFrame(subject_rows).sort_values(["subject", "fold"]).reset_index(drop=True),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["gray"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["navy"],
            "ytick.color": COLORS["navy"],
            "text.color": COLORS["navy"],
            "axes.labelcolor": COLORS["navy"],
            "axes.titlecolor": COLORS["navy"],
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["navy"],
    )


def plot_confusion(axis: plt.Axes, target: np.ndarray, prediction: np.ndarray) -> None:
    matrix = confusion_matrix(target, prediction, labels=np.arange(4)).astype(float)
    matrix = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    cmap = LinearSegmentedColormap.from_list(
        "paper_blues", ["#F5F8FB", "#9CC7E4", "#1F5E8C"]
    )
    image = axis.imshow(
        matrix, vmin=0, vmax=max(0.5, float(matrix.max())), cmap=cmap, aspect="equal"
    )
    for row in range(4):
        for column in range(4):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{100 * value:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=COLORS["white"] if value > 0.34 else COLORS["navy"],
            )
    axis.set_xticks(range(4), TONE_LABELS)
    axis.set_yticks(range(4), TONE_LABELS)
    axis.set_xlabel("Predicted tone")
    axis.set_ylabel("True tone")
    axis.set_title("Pooled held-out confusion matrix", loc="left", fontweight="bold")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Row proportion")


def make_figure(
    folds: pd.DataFrame,
    subjects: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    output: Path,
) -> None:
    set_style()
    figure, axes = plt.subplots(
        1, 3, figsize=(14.8, 4.4), gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]}
    )
    figure.subplots_adjust(wspace=0.36)
    x = np.arange(1, 6)

    axis = axes[0]
    axis.bar(
        x,
        folds["balanced_accuracy"],
        width=0.58,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Neural model",
        zorder=3,
    )
    axis.axhline(CHANCE, color=COLORS["orange"], linestyle="--", linewidth=1.2, label="Chance (25%)")
    axis.set_xticks(x)
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 0.45)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axis.legend(fontsize=8, loc="upper left")
    axis.set_title("Cross-participant tone decoding", loc="left", fontweight="bold")
    panel_label(axis, "a")

    axis = axes[1]
    width = 0.34
    participants = list(subjects["subject"].drop_duplicates())
    if len(participants) != 2:
        raise ValueError(f"Expected two held-out participants, found {participants}")
    colors = [COLORS["blue"], COLORS["teal"]]
    for offset, subject, color in zip((-width / 2, width / 2), participants, colors):
        frame = subjects.loc[subjects["subject"] == subject]
        axis.bar(
            frame["fold"].to_numpy() + 1 + offset,
            frame["balanced_accuracy"],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"{subject} (mean={100 * frame['balanced_accuracy'].mean():.1f}%)",
            zorder=3,
        )
    axis.axhline(CHANCE, color=COLORS["orange"], linestyle="--", linewidth=1.2, label="Chance (25%)")
    axis.set_xticks(x)
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 0.46)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axis.legend(fontsize=7.5, loc="upper left")
    axis.set_title("Held-out participant performance", loc="left", fontweight="bold")
    panel_label(axis, "b")

    axis = axes[2]
    plot_confusion(axis, target, prediction)
    panel_label(axis, "c")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the cross-participant Technical Validation figure."
    )
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("figures/cross_participant_validation")
    )
    args = parser.parse_args()
    folds, subjects, target, prediction = load_results(args.cross_root.resolve())
    make_figure(folds, subjects, target, prediction, args.output.resolve())
    print(args.output.resolve().with_suffix(".png"))


if __name__ == "__main__":
    main()
