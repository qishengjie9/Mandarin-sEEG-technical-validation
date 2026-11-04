from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("tone", "initial", "final")


def write_cv_artifacts(output_root: Path, rows: list[dict[str, object]]) -> None:
    completed = [row for row in rows if row["status"] in {"completed", "skipped"} and "metrics" in row]
    if not completed:
        return

    summary: dict[str, object] = {"folds": rows, "tasks": {}}
    for task in TASKS:
        values = np.asarray(
            [row["metrics"][task]["balanced_accuracy"] for row in completed], dtype=float
        )
        summary["tasks"][task] = {
            "balanced_accuracy_mean": float(values.mean()),
            "balanced_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "folds": int(len(values)),
        }
    (output_root / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_root / "cv_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["fold", "status"]
        for task in TASKS:
            fieldnames.extend(
                [
                    f"{task}_accuracy",
                    f"{task}_balanced_accuracy",
                    f"{task}_macro_f1",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in completed:
            metrics = row["metrics"]
            output = {"fold": row["fold"], "status": row["status"]}
            for task in TASKS:
                output[f"{task}_accuracy"] = metrics[task]["accuracy"]
                output[f"{task}_balanced_accuracy"] = metrics[task]["balanced_accuracy"]
                output[f"{task}_macro_f1"] = metrics[task]["macro_f1"]
            writer.writerow(output)

    folds = [row["fold"] for row in completed]
    x = np.arange(len(folds))
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    width = 0.25
    for offset, task in enumerate(TASKS):
        values = [row["metrics"][task]["balanced_accuracy"] for row in completed]
        axis.bar(x + (offset - 1) * width, values, width=width, label=task)
    axis.set_xticks(x, [f"fold-{fold:02d}" for fold in folds])
    axis.set_ylabel("Balanced Accuracy")
    axis.set_ylim(0, 1)
    axis.legend()
    fig.savefig(output_root / "cv_balanced_accuracy.png", dpi=180)
    fig.savefig(output_root / "cv_balanced_accuracy.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable held-out-participant phonology CV.")
    parser.add_argument("--config", default="configs/cross_participant.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/cross_participant/sub-02_sub-08")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = Path(args.config)
    if not config.is_absolute():
        config = (project / config).resolve()
    train_script = project / "scripts" / "train_cross_participant.py"
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (project / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(args.folds):
        fold_root = output_root / f"fold-{fold:02d}"
        metrics_path = fold_root / "metrics.json"
        if metrics_path.exists() and not args.force:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"SKIP fold={fold} tone_BA={metrics['tone']['balanced_accuracy']:.4f}", flush=True)
            rows.append({"fold": fold, "status": "skipped", "metrics": metrics})
            continue
        fold_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(train_script), "--config", str(config),
            "--fold", str(fold), "--output", str(fold_root),
        ]
        if args.dataset_root is not None:
            command.extend(["--dataset-root", str(args.dataset_root.resolve())])
        print(f"START fold={fold}", flush=True)
        with (fold_root / "train.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0 or not metrics_path.exists():
            rows.append({"fold": fold, "status": "failed", "log": str(fold_root / "train.log")})
            print(f"FAILED fold={fold}", flush=True)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"fold": fold, "status": "completed", "metrics": metrics})
        print(f"DONE fold={fold} tone_BA={metrics['tone']['balanced_accuracy']:.4f}", flush=True)
        write_cv_artifacts(output_root, rows)
    summary = {
        "folds": rows,
        "completed": sum(row["status"] in {"completed", "skipped"} for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
    }
    (output_root / "cv_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_cv_artifacts(output_root, rows)
    print(json.dumps({"completed": summary["completed"], "failed": summary["failed"]}))


if __name__ == "__main__":
    main()
