from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> dict[str, object]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    files: list[str] = []
    stimuli: list[str] = []
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        mask = batch["channel_mask"].to(device, non_blocking=True)
        output = model(signal, mask)
        probabilities.append(torch.softmax(output["tone"], dim=-1).cpu().numpy())
        embeddings.append(output["embedding"].cpu().numpy())
        targets.append(batch["tone"].numpy())
        files.extend(batch["recording_file"])
        stimuli.extend(batch["stimulus"])
    probability = np.concatenate(probabilities)
    target = np.concatenate(targets)
    prediction = probability.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1, 2, 3]).tolist(),
    }
    try:
        metrics["macro_ovr_auc"] = float(roc_auc_score(target, probability, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["macro_ovr_auc"] = None
    return {
        "metrics": metrics,
        "target": target,
        "prediction": prediction,
        "probability": probability,
        "embedding": np.concatenate(embeddings),
        "recording_file": files,
        "stimulus": stimuli,
    }

