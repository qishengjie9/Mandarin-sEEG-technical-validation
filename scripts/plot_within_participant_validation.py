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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


CHANCE = 0.25
TONE_LABELS = ["T1", "T2", "T3", "T4"]
COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "orange": "#E07A3F",
    "teal": "#2A9D8F",
    "gray": "#7A8793",
    "light_gray": "#D9E0E6",
    "grid": "#E7EBEF",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cohort_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for subject_dir in sorted(root.glob("sub-*")):
        summary_path = subject_dir / "subject_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        metrics = summary["models"]["stag"]
        subject_rows.append(
            {
                "subject": summary["subject"],
                "balanced_accuracy": float(metrics["balanced_accuracy_mean"]),
            }
        )
        for fold_dir in sorted((subject_dir / "stag").glob("fold-*")):
            fold_metrics = read_json(fold_dir / "best_metrics.json")
            fold_rows.append(
                {
                    "subject": summary["subject"],
                    "fold": int(fold_dir.name.split("-")[-1]),
                    "balanced_accuracy": float(fold_metrics["balanced_accuracy"]),
                }
            )
    if not subject_rows or not fold_rows:
        raise FileNotFoundError(f"No completed within-participant results found under {root}")
    return (
        pd.DataFrame(subject_rows).sort_values("subject").reset_index(drop=True),
        pd.DataFrame(fold_rows).sort_values(["subject", "fold"]).reset_index(drop=True),
    )


def load_rerun(root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for fold_dir in sorted(root.glob("fold-*")):
        payload = np.load(fold_dir / "best_predictions.npz", allow_pickle=False)
        target = np.asarray(payload["target"], dtype=int)
        prediction = np.asarray(payload["prediction"], dtype=int)
        probability = np.asarray(payload["probability"], dtype=float)
        stored = read_json(fold_dir / "best_metrics.json")
        recalculated = {
            "accuracy": float(accuracy_score(target, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "macro_f1": float(f1_score(target, prediction, average="macro")),
            "macro_ovr_auc": float(
                roc_auc_score(target, probability, multi_class="ovr", average="macro")
            ),
        }
        max_error = max(abs(recalculated[key] - float(stored[key])) for key in recalculated)
        if max_error > 1e-10:
            raise ValueError(f"Stored and recalculated metrics disagree in {fold_dir}: {max_error}")
        rows.append(
            {
                "fold": int(fold_dir.name.split("-")[-1]),
                "n_test": len(target),
                **recalculated,
            }
        )
        targets.append(target)
        predictions.append(prediction)
        probabilities.append(probability)
    if len(rows) != 5:
        raise ValueError(f"Expected five rerun folds under {root}, found {len(rows)}")
    return (
        pd.DataFrame(rows).sort_values("fold").reset_index(drop=True),
        np.concatenate(targets),
        np.concatenate(predictions),
        np.concatenate(probabilities),
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
        -0.14,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
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
                color="white" if value > 0.34 else COLORS["navy"],
            )
    axis.set_xticks(range(4), TONE_LABELS)
    axis.set_yticks(range(4), TONE_LABELS)
    axis.set_xlabel("Predicted tone")
    axis.set_ylabel("True tone")
    axis.set_title("Rerun confusion matrix", loc="left", fontweight="bold")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Row proportion")


def make_figure(
    cohort: pd.DataFrame,
    cohort_folds: pd.DataFrame,
    rerun: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    output: Path,
) -> None:
    set_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    figure.subplots_adjust(wspace=0.32, hspace=0.38)

    axis = axes[0, 0]
    positions = np.arange(len(cohort))
    for index, row in cohort.iterrows():
        values = cohort_folds.loc[
            cohort_folds["subject"] == row["subject"], "balanced_accuracy"
        ].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            np.full(len(values), positions[index]) + jitter,
            values,
            s=18,
            color=COLORS["light_gray"],
            edgecolor=COLORS["gray"],
            linewidth=0.4,
            zorder=2,
        )
    axis.scatter(
        positions,
        cohort["balanced_accuracy"],
        s=42,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
        label="Participant mean",
    )
    axis.axhline(CHANCE, color=COLORS["orange"], linestyle="--", linewidth=1.2, label="Chance (25%)")
    axis.set_xticks(
        positions,
        cohort["subject"].str.replace("sub-", "S", regex=False),
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.20, max(0.60, float(cohort_folds["balanced_accuracy"].max()) + 0.04))
    axis.set_title("Within-participant tone decoding", loc="left", fontweight="bold")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axis.legend(loc="upper right", fontsize=8)
    panel_label(axis, "a")

    axis = axes[0, 1]
    folds = np.arange(1, 6)
    axis.bar(
        folds,
        rerun["balanced_accuracy"],
        width=0.58,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axis.axhline(CHANCE, color=COLORS["orange"], linestyle="--", linewidth=1.2, label="Chance (0.25)")
    axis.set_xticks(folds)
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 0.62)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axis.legend(fontsize=7.2, loc="upper left")
    axis.set_title("Independent rerun performance", loc="left", fontweight="bold")
    panel_label(axis, "b")

    axis = axes[1, 0]
    plot_confusion(axis, target, prediction)
    panel_label(axis, "c")

    axis = axes[1, 1]
    binary_target = label_binarize(target, classes=np.arange(4))
    tone_colors = [COLORS["blue"], COLORS["orange"], COLORS["teal"], "#8C6BB1"]
    for tone in range(4):
        false_positive, true_positive, _ = roc_curve(binary_target[:, tone], probability[:, tone])
        tone_auc = roc_auc_score(binary_target[:, tone], probability[:, tone])
        axis.plot(
            false_positive,
            true_positive,
            color=tone_colors[tone],
            linewidth=1.7,
            label=f"{TONE_LABELS[tone]} (AUC={tone_auc:.2f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.grid(color=COLORS["grid"], linewidth=0.7)
    axis.legend(fontsize=7.5, loc="lower right")
    axis.set_title("One-vs-rest ROC curves", loc="left", fontweight="bold")
    panel_label(axis, "d")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the within-participant Technical Validation figure."
    )
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("figures/within_participant_validation")
    )
    args = parser.parse_args()
    cohort, cohort_folds = load_cohort_results(args.cohort_root.resolve())
    rerun, target, prediction, probability = load_rerun(args.rerun_root.resolve())
    make_figure(
        cohort,
        cohort_folds,
        rerun,
        target,
        prediction,
        probability,
        args.output.resolve(),
    )
    print(args.output.resolve().with_suffix(".png"))


if __name__ == "__main__":
    main()
