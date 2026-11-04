from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: list[int], classes: int = 4) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def supervised_contrastive_loss(embedding: torch.Tensor, group_ids: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    embedding = nn.functional.normalize(embedding, dim=-1)
    logits = embedding @ embedding.T / temperature
    self_mask = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = group_ids[:, None].eq(group_ids[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    return -(log_probability[valid] * positive[valid]).sum(dim=1).div(positive_count[valid]).mean()


def _group_tensor(stimuli: list[str], device: torch.device) -> torch.Tensor:
    mapping: dict[str, int] = {}
    values = []
    for stimulus in stimuli:
        if stimulus not in mapping:
            mapping[stimulus] = len(mapping)
        values.append(mapping[stimulus])
    return torch.tensor(values, dtype=torch.long, device=device)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, object]]:
    epochs = int(config["epochs"])
    early_stopping = bool(config.get("early_stopping", False))
    patience = int(config.get("patience", epochs))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    weights = class_weights([trial.tone for trial in train_loader.dataset.trials]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            x = batch["signal"].to(device, non_blocking=True)
            mask = batch["channel_mask"].to(device, non_blocking=True)
            target = batch["tone"].to(device, non_blocking=True)
            groups = _group_tensor(batch["stimulus"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(x, mask)
                classification = criterion(output["tone"], target)
                contrastive = supervised_contrastive_loss(output["embedding"], groups)
                loss = classification + 0.1 * contrastive
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(target)
            train_count += len(target)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["signal"].to(device, non_blocking=True)
                mask = batch["channel_mask"].to(device, non_blocking=True)
                target = batch["tone"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    output = model(x, mask)
                    loss = criterion(output["tone"], target)
                val_loss += float(loss.item()) * len(target)
                val_count += len(target)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(1, train_count),
            "val_loss": val_loss / max(1, val_count),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(row, flush=True)
        if row["val_loss"] < best_loss - 1e-5:
            best_loss = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if early_stopping and stale >= patience:
                break
    last_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoints = {
        "best_state": best_state,
        "last_state": last_state,
        "best_epoch": best_epoch,
        "last_epoch": len(history),
        "best_val_loss": best_loss,
        "early_stopping": early_stopping,
    }
    return model, history, checkpoints


def save_training_outputs(
    output: str | Path,
    model: nn.Module,
    history: list[dict],
    result: dict,
    config: dict,
    checkpoint_name: str = "best",
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output / f"{checkpoint_name}_model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / f"{checkpoint_name}_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / f"{checkpoint_name}_predictions.npz",
        target=result["target"],
        prediction=result["prediction"],
        probability=result["probability"],
        embedding=result["embedding"],
        recording_file=np.asarray(result["recording_file"], dtype=str),
        stimulus=np.asarray(result["stimulus"], dtype=str),
    )
    if checkpoint_name == "best":
        torch.save({"model": model.state_dict(), "config": config}, output / "model.pt")
        (output / "metrics.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        np.savez_compressed(
            output / "predictions.npz",
            target=result["target"],
            prediction=result["prediction"],
            probability=result["probability"],
            embedding=result["embedding"],
            recording_file=np.asarray(result["recording_file"], dtype=str),
            stimulus=np.asarray(result["stimulus"], dtype=str),
        )
