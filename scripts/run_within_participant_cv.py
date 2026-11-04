from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_SUBJECTS = [
    "sub-01",
    "sub-02",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-08",
    "sub-09",
    "sub-10",
    "sub-11",
    "sub-12",
]


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metrics(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def summarize_subject(subject_root: Path, models: list[str], fold_count: int) -> dict[str, object]:
    summary: dict[str, object] = {"subject": subject_root.name, "models": {}}
    for model in models:
        rows = []
        for fold in range(fold_count):
            metrics = load_metrics(subject_root / model / f"fold-{fold:02d}" / "metrics.json")
            if metrics is not None:
                rows.append(metrics)
        model_summary: dict[str, object] = {
            "completed_folds": len(rows),
            "expected_folds": fold_count,
        }
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc"):
            values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float)
            if len(values):
                model_summary[f"{key}_mean"] = float(values.mean())
                model_summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        checkpoint_summaries: dict[str, object] = {}
        for checkpoint in ("best", "last"):
            checkpoint_rows = []
            for fold in range(fold_count):
                metrics = load_metrics(subject_root / model / f"fold-{fold:02d}" / f"{checkpoint}_metrics.json")
                if metrics is not None:
                    checkpoint_rows.append(metrics)
            checkpoint_summary: dict[str, object] = {
                "completed_folds": len(checkpoint_rows),
                "expected_folds": fold_count,
            }
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc"):
                values = np.asarray(
                    [row[key] for row in checkpoint_rows if row.get(key) is not None],
                    dtype=float,
                )
                if len(values):
                    checkpoint_summary[f"{key}_mean"] = float(values.mean())
                    checkpoint_summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            checkpoint_summaries[checkpoint] = checkpoint_summary
        model_summary["checkpoints"] = checkpoint_summaries
        summary["models"][model] = model_summary
    (subject_root / "subject_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable sequential CV across Mandarin sEEG subjects.")
    parser.add_argument("--config", default="configs/within_participant.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--models", nargs="+", choices=["stag", "tcn"], default=["stag"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/within_participant")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = Path(args.config)
    if not config.is_absolute():
        config = (project / config).resolve()
    train_script = project / "scripts" / "train_within_participant.py"
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (project / output_root).resolve()
    status_path = output_root / "run_status.jsonl"
    failures_path = output_root / "failures.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    jobs = [(subject, model, fold) for subject in args.subjects for model in args.models for fold in range(args.folds)]
    print(f"jobs={len(jobs)} subjects={len(args.subjects)} models={args.models} folds={args.folds}", flush=True)
    append_jsonl(
        status_path,
        {
            "event": "launcher_started",
            "time": datetime.now().isoformat(timespec="seconds"),
            "jobs": len(jobs),
            "subjects": args.subjects,
            "models": args.models,
        },
    )

    completed = skipped = failed = 0
    for job_index, (subject, model, fold) in enumerate(jobs, start=1):
        fold_root = output_root / subject / model / f"fold-{fold:02d}"
        metrics_path = fold_root / "metrics.json"
        existing = load_metrics(metrics_path)
        if existing is not None and not args.force:
            skipped += 1
            print(
                f"[{job_index}/{len(jobs)}] SKIP {subject} {model} fold={fold} "
                f"balanced_accuracy={existing.get('balanced_accuracy')}",
                flush=True,
            )
            continue

        fold_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(config),
            "--subject",
            subject,
            "--model",
            model,
            "--fold",
            str(fold),
            "--output",
            str(fold_root),
        ]
        if args.dataset_root is not None:
            command.extend(["--dataset-root", str(args.dataset_root.resolve())])
        print(f"[{job_index}/{len(jobs)}] START {subject} {model} fold={fold}", flush=True)
        job_started = time.time()
        with (fold_root / "train.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - job_started
        metrics = load_metrics(metrics_path)
        last_metrics = load_metrics(fold_root / "last_metrics.json")
        if process.returncode == 0 and metrics is not None:
            completed += 1
            row = {
                "event": "fold_completed",
                "time": datetime.now().isoformat(timespec="seconds"),
                "subject": subject,
                "model": model,
                "fold": fold,
                "seconds": round(elapsed, 2),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "macro_ovr_auc": metrics.get("macro_ovr_auc"),
                "last_balanced_accuracy": None if last_metrics is None else last_metrics.get("balanced_accuracy"),
                "last_macro_f1": None if last_metrics is None else last_metrics.get("macro_f1"),
                "last_macro_ovr_auc": None if last_metrics is None else last_metrics.get("macro_ovr_auc"),
            }
            append_jsonl(status_path, row)
            print(
                f"[{job_index}/{len(jobs)}] DONE {subject} {model} fold={fold} "
                f"best_BA={metrics.get('balanced_accuracy'):.4f} "
                f"last_BA={row['last_balanced_accuracy']} seconds={elapsed:.1f}",
                flush=True,
            )
        else:
            failed += 1
            row = {
                "event": "fold_failed",
                "time": datetime.now().isoformat(timespec="seconds"),
                "subject": subject,
                "model": model,
                "fold": fold,
                "returncode": process.returncode,
                "seconds": round(elapsed, 2),
                "log": str(fold_root / "train.log"),
            }
            append_jsonl(status_path, row)
            append_jsonl(failures_path, row)
            print(
                f"[{job_index}/{len(jobs)}] FAILED {subject} {model} fold={fold}; "
                f"see {fold_root / 'train.log'}",
                flush=True,
            )

        subject_jobs_done = all(
            load_metrics(output_root / subject / current_model / f"fold-{current_fold:02d}" / "metrics.json")
            is not None
            for current_model in args.models
            for current_fold in range(args.folds)
        )
        if subject_jobs_done:
            summary = summarize_subject(output_root / subject, args.models, args.folds)
            print(f"SUBJECT_SUMMARY {json.dumps(summary, ensure_ascii=False)}", flush=True)

    for subject in args.subjects:
        summarize_subject(output_root / subject, args.models, args.folds)
    final = {
        "event": "launcher_finished",
        "time": datetime.now().isoformat(timespec="seconds"),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "seconds": round(time.time() - started, 2),
    }
    append_jsonl(status_path, final)
    (output_root / "launcher_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
