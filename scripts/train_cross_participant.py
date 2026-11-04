from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mandarin_seeg.config import load_config
from mandarin_seeg.cross_subject_data import (
    PhonologyDataset,
    SubjectBalancedBatchSampler,
    build_multisubject_trials,
    collate_phonology,
    cross_subject_stimulus_split,
)
from mandarin_seeg.cross_subject_training import (
    TASK_SIZES,
    evaluate_multitask,
    fit_multitask,
    save_cross_subject_outputs,
    seed_everything,
    write_trials,
)
from mandarin_seeg.model_registry import build_model
from mandarin_seeg.phonology import parse_pinyin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fold of the held-out-participant phonology decoder."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dataset_root is not None:
        config["data"]["root"] = str(args.dataset_root.resolve())
    if args.fold is not None:
        config["split"]["fold"] = args.fold
    if args.output is not None:
        config["output"] = args.output
    seed = int(config["seed"])
    seed_everything(seed)

    data = config["data"]
    subjects = list(data["subjects"])
    heldout_subjects = list(config["split"]["heldout_subjects"])
    if not set(heldout_subjects).issubset(subjects):
        raise ValueError("heldout_subjects must be included in data.subjects")
    trials, excluded = build_multisubject_trials(
        data["root"],
        subjects,
        config["preprocessing"],
        data.get("max_trials_per_subject"),
        seed,
    )
    # Fail before training if any pinyin cannot be mapped to the fixed phonology vocabulary.
    for trial in trials:
        parse_pinyin(trial.pinyin)

    split_config = config["split"]
    split = cross_subject_stimulus_split(
        trials,
        heldout_subjects,
        fold=int(split_config["fold"]),
        folds=int(split_config["folds"]),
        seed=seed,
        val_fraction=float(split_config.get("val_fraction", 0.1)),
        strict_unseen_stimulus=bool(split_config.get("strict_unseen_stimulus", True)),
    )
    development_subjects = sorted(set(subjects) - set(heldout_subjects))
    subject_to_index = {subject: index for index, subject in enumerate(development_subjects)}
    output = Path(config["output"])
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.mkdir(parents=True, exist_ok=True)
    disk_cache = data.get("disk_cache")
    if disk_cache:
        disk_cache = Path(disk_cache)
        if not disk_cache.is_absolute():
            disk_cache = Path(__file__).resolve().parents[1] / disk_cache

    write_trials(output / "train_manifest.csv", split.train)
    write_trials(output / "val_manifest.csv", split.val)
    write_trials(output / "test_manifest.csv", split.test)
    if excluded:
        with (output / "excluded_trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(excluded[0]))
            writer.writeheader()
            writer.writerows(excluded)
    split_summary = {
        "development_subjects": development_subjects,
        "heldout_subjects": heldout_subjects,
        "strict_unseen_stimulus": bool(split_config.get("strict_unseen_stimulus", True)),
        "train_trials": len(split.train),
        "val_trials": len(split.val),
        "test_trials": len(split.test),
        "train_stimuli": len(split.train_stimuli),
        "val_stimuli": len(split.val_stimuli),
        "test_stimuli": len(split.test_stimuli),
    }
    (output / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    datasets = {
        "train": PhonologyDataset(
            data["root"], split.train, config["preprocessing"], subject_to_index,
            cache_items=int(data.get("cache_items", 512)),
            disk_cache=disk_cache,
        ),
        "val": PhonologyDataset(
            data["root"], split.val, config["preprocessing"], subject_to_index,
            cache_items=int(data.get("cache_items", 512)),
            disk_cache=disk_cache,
        ),
        "test": PhonologyDataset(
            data["root"], split.test, config["preprocessing"], subject_to_index,
            cache_items=int(data.get("cache_items", 512)),
            disk_cache=disk_cache,
        ),
    }
    training = config["training"]
    num_workers = int(training.get("num_workers", 0))
    common = {
        "num_workers": num_workers,
        "collate_fn": collate_phonology,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(training.get("persistent_workers", True))
        common["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_sampler=SubjectBalancedBatchSampler(
                datasets["train"].trials, int(training["batch_size"]), seed
            ),
            **common,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=int(training["batch_size"]), shuffle=False, **common
        ),
        "test": DataLoader(
            datasets["test"], batch_size=int(training["batch_size"]), shuffle=False, **common
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")
    print(json.dumps(split_summary, ensure_ascii=False))
    model = build_model(config["model"], TASK_SIZES, len(development_subjects))
    model, history = fit_multitask(
        model, loaders["train"], loaders["val"], training, config["loss"], device
    )
    result = evaluate_multitask(model, loaders["test"], device)
    result["metrics"]["split"] = split_summary
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    save_cross_subject_outputs(output, model, history, result, config)


if __name__ == "__main__":
    main()
