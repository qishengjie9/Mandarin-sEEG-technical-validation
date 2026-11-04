from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mandarin_seeg.config import load_config
from mandarin_seeg.data import StimulusBatchSampler, ToneDataset, collate_variable_channels
from mandarin_seeg.evaluation import predict
from mandarin_seeg.manifest import balanced_limit, build_tone_manifest, write_manifest
from mandarin_seeg.models import build_model
from mandarin_seeg.preprocessing import filter_valid_trials
from mandarin_seeg.splits import make_outer_split, make_train_val_split
from mandarin_seeg.training import fit, save_training_outputs, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fold of the within-participant lexical-tone decoder."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", choices=["stag", "tcn"])
    parser.add_argument("--subject")
    parser.add_argument("--permutation-index", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.dataset_root is not None:
        config["data"]["root"] = str(args.dataset_root.resolve())
    if args.fold is not None:
        config["training"]["fold"] = args.fold
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.subject is not None:
        config["data"]["subject"] = args.subject
    if args.output is not None:
        config["output"] = args.output
    seed = int(config["seed"])
    seed_everything(seed)
    data_config = config["data"]
    manifest_trials = build_tone_manifest(data_config["root"], data_config["subject"])
    all_trials, excluded = filter_valid_trials(
        data_config["root"],
        manifest_trials,
        config["preprocessing"],
    )
    trials = balanced_limit(
        all_trials,
        data_config.get("max_trials"),
        seed,
    )
    if args.permutation_index is not None:
        rng = __import__("numpy").random.default_rng(seed + 10000 + args.permutation_index)
        stimulus_tones: dict[str, int] = {}
        for trial in trials:
            stimulus_tones.setdefault(trial.stimulus, trial.tone)
        stimuli = sorted(stimulus_tones)
        shuffled = [stimulus_tones[stimulus] for stimulus in stimuli]
        rng.shuffle(shuffled)
        mapping = dict(zip(stimuli, shuffled))
        trials = [replace(trial, tone=mapping[trial.stimulus]) for trial in trials]
    training = config["training"]
    split_strategy = str(training.get("split_strategy", "stimulus_grouped"))
    outer_train, test_indices = make_outer_split(
        trials,
        int(training["outer_folds"]),
        int(training["fold"]),
        seed,
        split_strategy,
    )
    train_indices, val_indices = make_train_val_split(
        outer_train,
        trials,
        seed,
        float(training.get("validation_fraction", 0.15)),
        split_strategy,
    )
    subsets = {
        "train": [trials[index] for index in train_indices],
        "val": [trials[index] for index in val_indices],
        "test": [trials[index] for index in test_indices],
    }
    output = Path(config["output"])
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.mkdir(parents=True, exist_ok=True)
    cleaning_summary = {
        "subject": data_config["subject"],
        "manifest_trials": len(manifest_trials),
        "valid_trials_after_qc": len(all_trials),
        "excluded_trials": len(excluded),
        "used_trials_after_balancing": len(trials),
        "excluded_by_reason": dict(Counter(str(row.get("reason", "unknown")) for row in excluded)),
    }
    (output / "cleaning_summary.json").write_text(
        json.dumps(cleaning_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if excluded:
        import csv

        with (output / "excluded_trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = list(excluded[0])
            for row in excluded[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(excluded)
    for name, subset in subsets.items():
        write_manifest(output / f"{name}_manifest.csv", subset)
    overlap = set(trial.stimulus for trial in subsets["train"]) & set(trial.stimulus for trial in subsets["test"])
    if split_strategy == "stimulus_grouped" and overlap:
        raise RuntimeError(f"stimulus leakage across train/test: {sorted(overlap)[:5]}")

    datasets = {
        name: ToneDataset(
            data_config["root"],
            subset,
            config["preprocessing"],
            cache_items=int(data_config.get("cache_items", 1024)),
        )
        for name, subset in subsets.items()
    }
    loaders = {}
    for name, dataset in datasets.items():
        common = {
            "num_workers": int(training.get("num_workers", 0)),
            "collate_fn": collate_variable_channels,
            "pin_memory": torch.cuda.is_available(),
        }
        if name == "train" and bool(training.get("grouped_batches", True)):
            loaders[name] = DataLoader(
                dataset,
                batch_sampler=StimulusBatchSampler(
                    dataset.trials,
                    int(training["batch_size"]),
                    seed,
                ),
                **common,
            )
        else:
            loaders[name] = DataLoader(
                dataset,
                batch_size=int(training["batch_size"]),
                shuffle=name == "train",
                **common,
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")
    print({name: len(dataset) for name, dataset in datasets.items()})
    model = build_model(config["model"])
    model, history, checkpoints = fit(model, loaders["train"], loaders["val"], training, device)
    checkpoint_metrics = {}
    for checkpoint_name in ("best", "last"):
        state = checkpoints.get(f"{checkpoint_name}_state")
        if state is not None:
            model.load_state_dict(state)
        result = predict(model, loaders["test"], device)
        result["metrics"]["model"] = config["model"]["name"]
        result["metrics"]["fold"] = int(training["fold"])
        result["metrics"]["permutation_index"] = args.permutation_index
        result["metrics"]["checkpoint"] = checkpoint_name
        result["metrics"]["checkpoint_epoch"] = int(checkpoints.get(f"{checkpoint_name}_epoch", len(history)))
        result["metrics"]["best_val_loss"] = float(checkpoints["best_val_loss"])
        checkpoint_metrics[checkpoint_name] = result["metrics"]
        save_training_outputs(output, model, history, result, config, checkpoint_name=checkpoint_name)
    (output / "checkpoint_metrics.json").write_text(
        json.dumps(checkpoint_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checkpoint_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
