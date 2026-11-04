from __future__ import annotations

import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from tqdm import tqdm

from .phonology import FINALS, INITIALS, parse_pinyin


TASK_SIZES = {"tone": 4, "initial": len(INITIALS), "final": len(FINALS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels_for_task(trials, task: str) -> list[int]:
    output = []
    for trial in trials:
        label = parse_pinyin(trial.pinyin)
        output.append(
            label.tone if task == "tone" else label.initial_index if task == "initial" else label.final_index
        )
    return output


def class_weights(trials, task: str, classes: int) -> torch.Tensor:
    counts = np.bincount(_labels_for_task(trials, task), minlength=classes)
    present = counts > 0
    weights = np.zeros(classes, dtype=float)
    weights[present] = counts[present].sum() / counts[present]
    weights[present] = weights[present] / weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def cross_subject_tone_contrastive_loss(
    embedding: torch.Tensor,
    tone: torch.Tensor,
    subject_index: torch.Tensor,
    stimulus_index: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    embedding = nn.functional.normalize(embedding, dim=-1)
    logits = embedding @ embedding.T / temperature
    self_mask = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    same_tone = tone[:, None].eq(tone[None, :])
    different_subject = ~subject_index[:, None].eq(subject_index[None, :])
    different_stimulus = ~stimulus_index[:, None].eq(stimulus_index[None, :])
    positive = same_tone & different_subject & different_stimulus & ~self_mask
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positive.sum(dim=1)
    valid = count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    return -(log_probability[valid] * positive[valid]).sum(dim=1).div(count[valid]).mean()


def _batch_string_ids(values: list[str], device: torch.device) -> torch.Tensor:
    mapping: dict[str, int] = {}
    output = []
    for value in values:
        mapping.setdefault(value, len(mapping))
        output.append(mapping[value])
    return torch.tensor(output, dtype=torch.long, device=device)


def fit_multitask(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    loss_config: dict,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criteria = {
        task: nn.CrossEntropyLoss(
            weight=class_weights(train_loader.dataset.trials, task, size).to(device)
        )
        for task, size in TASK_SIZES.items()
    }
    task_weights = {
        "tone": float(loss_config.get("tone", 1.0)),
        "initial": float(loss_config.get("initial", 0.5)),
        "final": float(loss_config.get("final", 0.5)),
    }
    contrastive_weight = float(loss_config.get("contrastive", 0.1))
    domain_weight = float(loss_config.get("domain", 0.1))
    max_grl = float(loss_config.get("max_grl_scale", 1.0))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        progress = (epoch - 1) / max(1, epochs - 1)
        grl_scale = max_grl * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
        model.train()
        started = time.perf_counter()
        train_total = 0.0
        train_count = 0
        train_correct = {task: 0 for task in TASK_SIZES}
        progress_bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress_bar:
            signal = batch["signal"].to(device, non_blocking=True)
            mask = batch["channel_mask"].to(device, non_blocking=True)
            targets = {task: batch[task].to(device, non_blocking=True) for task in TASK_SIZES}
            domains = batch["subject_index"].to(device, non_blocking=True)
            stimulus_ids = _batch_string_ids(batch["stimulus"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(signal, mask, grl_scale=grl_scale)
                losses = {
                    task: criteria[task](output[task], targets[task]) for task in TASK_SIZES
                }
                total = sum(task_weights[task] * losses[task] for task in TASK_SIZES)
                contrastive = cross_subject_tone_contrastive_loss(
                    output["contrastive_embedding"], targets["tone"], domains, stimulus_ids
                )
                total = total + contrastive_weight * contrastive
                if "domain" in output:
                    domain_loss = nn.functional.cross_entropy(output["domain"], domains)
                    total = total + domain_weight * domain_loss
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_total += float(total.item()) * len(signal)
            train_count += len(signal)
            for task in TASK_SIZES:
                train_correct[task] += int((output[task].argmax(dim=1) == targets[task]).sum().item())
            progress_bar.set_postfix(
                loss=f"{train_total / max(1, train_count):.3f}",
                tone=f"{train_correct['tone'] / max(1, train_count):.3f}",
            )

        model.eval()
        val_total = 0.0
        val_count = 0
        val_correct = {task: 0 for task in TASK_SIZES}
        with torch.no_grad():
            for batch in val_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                mask = batch["channel_mask"].to(device, non_blocking=True)
                targets = {task: batch[task].to(device, non_blocking=True) for task in TASK_SIZES}
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    output = model(signal, mask, grl_scale=0.0)
                    total = sum(
                        task_weights[task] * criteria[task](output[task], targets[task])
                        for task in TASK_SIZES
                    )
                val_total += float(total.item()) * len(signal)
                val_count += len(signal)
                for task in TASK_SIZES:
                    val_correct[task] += int((output[task].argmax(dim=1) == targets[task]).sum().item())
        scheduler.step()
        epoch_seconds = time.perf_counter() - started
        row = {
            "epoch": float(epoch),
            "train_loss": train_total / max(1, train_count),
            "val_loss": val_total / max(1, val_count),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "grl_scale": float(grl_scale),
            "epoch_seconds": float(epoch_seconds),
            "examples_per_second": float(train_count / max(epoch_seconds, 1e-9)),
            **{f"train_{task}_accuracy": train_correct[task] / max(1, train_count) for task in TASK_SIZES},
            **{f"val_{task}_accuracy": val_correct[task] / max(1, val_count) for task in TASK_SIZES},
        }
        history.append(row)
        print(row, flush=True)
        if row["val_loss"] < best_loss - 1e-5:
            best_loss = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_history_artifacts(output: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fields = list(history[0])
    with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    for task in TASK_SIZES:
        axes[0, 1].plot(epochs, [row[f"val_{task}_accuracy"] for row in history], label=task)
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["learning_rate"] for row in history])
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, [row["examples_per_second"] for row in history])
    axes[1, 1].set_title("Training Throughput")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Examples/s")

    fig.savefig(output / "training_curves.png", dpi=180)
    fig.savefig(output / "training_curves.svg")
    plt.close(fig)


@torch.no_grad()
def evaluate_multitask(model: nn.Module, loader, device: torch.device) -> dict[str, object]:
    model.eval()
    targets = {task: [] for task in TASK_SIZES}
    probabilities = {task: [] for task in TASK_SIZES}
    embeddings = []
    metadata = {key: [] for key in ("subject", "stimulus", "pinyin", "recording_file")}
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        mask = batch["channel_mask"].to(device, non_blocking=True)
        output = model(signal, mask, grl_scale=0.0)
        embeddings.append(output["embedding"].cpu().numpy())
        for task in TASK_SIZES:
            targets[task].append(batch[task].numpy())
            probabilities[task].append(torch.softmax(output[task], dim=-1).cpu().numpy())
        for key in metadata:
            metadata[key].extend(batch[key])

    result: dict[str, object] = {
        "metrics": {"by_subject": {}}, **metadata, "embedding": np.concatenate(embeddings)
    }
    for task, size in TASK_SIZES.items():
        target = np.concatenate(targets[task])
        probability = np.concatenate(probabilities[task])
        prediction = probability.argmax(axis=1)
        task_metrics = {
            "accuracy": float(accuracy_score(target, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(target, prediction, labels=list(range(size))).tolist(),
        }
        if task == "tone":
            try:
                task_metrics["macro_ovr_auc"] = float(
                    roc_auc_score(target, probability, multi_class="ovr", average="macro")
                )
            except ValueError:
                task_metrics["macro_ovr_auc"] = None
        result["metrics"][task] = task_metrics
        result[f"{task}_target"] = target
        result[f"{task}_prediction"] = prediction
        result[f"{task}_probability"] = probability
    subject_array = np.asarray(result["subject"], dtype=str)
    for subject in sorted(set(subject_array.tolist())):
        subject_mask = subject_array == subject
        subject_metrics: dict[str, object] = {}
        for task, size in TASK_SIZES.items():
            target = result[f"{task}_target"][subject_mask]
            prediction = result[f"{task}_prediction"][subject_mask]
            probability = result[f"{task}_probability"][subject_mask]
            values = {
                "trials": int(subject_mask.sum()),
                "accuracy": float(accuracy_score(target, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
                "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
            }
            if task == "tone":
                try:
                    values["macro_ovr_auc"] = float(
                        roc_auc_score(target, probability, multi_class="ovr", average="macro")
                    )
                except ValueError:
                    values["macro_ovr_auc"] = None
            subject_metrics[task] = values
        result["metrics"]["by_subject"][subject] = subject_metrics
    return result


def save_cross_subject_outputs(
    output: str | Path,
    model: nn.Module,
    history: list[dict[str, float]],
    result: dict[str, object],
    config: dict,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_history_artifacts(output, history)
    (output / "metrics.json").write_text(
        json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    arrays = {
        "embedding": result["embedding"],
        "subject": np.asarray(result["subject"], dtype=str),
        "stimulus": np.asarray(result["stimulus"], dtype=str),
        "pinyin": np.asarray(result["pinyin"], dtype=str),
        "recording_file": np.asarray(result["recording_file"], dtype=str),
    }
    for task in TASK_SIZES:
        arrays[f"{task}_target"] = result[f"{task}_target"]
        arrays[f"{task}_prediction"] = result[f"{task}_prediction"]
        arrays[f"{task}_probability"] = result[f"{task}_probability"]
    np.savez_compressed(output / "predictions.npz", **arrays)


def write_trials(path: Path, trials) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0].to_dict()))
        writer.writeheader()
        writer.writerows(trial.to_dict() for trial in trials)
