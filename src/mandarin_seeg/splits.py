from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .manifest import ToneTrial


def grouped_outer_split(trials: list[ToneTrial], folds: int, fold: int, seed: int) -> tuple[list[int], list[int]]:
    y = np.asarray([trial.tone for trial in trials])
    groups = np.asarray([trial.stimulus for trial in trials])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    train, test = splits[fold]
    return train.tolist(), test.tolist()


def grouped_train_val(indices: list[int], trials: list[ToneTrial], seed: int, fraction: float = 0.15) -> tuple[list[int], list[int]]:
    subset = [trials[index] for index in indices]
    y = np.asarray([trial.tone for trial in subset])
    groups = np.asarray([trial.stimulus for trial in subset])
    folds = max(2, min(5, int(round(1 / fraction))))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(iter(splitter.split(np.zeros(len(y)), y, groups)))
    return [indices[int(i)] for i in train_local], [indices[int(i)] for i in val_local]


def stratified_outer_split(trials: list[ToneTrial], folds: int, fold: int, seed: int) -> tuple[list[int], list[int]]:
    y = np.asarray([trial.tone for trial in trials])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y))
    train, test = splits[fold]
    return train.tolist(), test.tolist()


def stratified_train_val(indices: list[int], trials: list[ToneTrial], seed: int, fraction: float = 0.15) -> tuple[list[int], list[int]]:
    subset = [trials[index] for index in indices]
    y = np.asarray([trial.tone for trial in subset])
    folds = max(2, min(5, int(round(1 / fraction))))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(iter(splitter.split(np.zeros(len(y)), y)))
    return [indices[int(i)] for i in train_local], [indices[int(i)] for i in val_local]


def make_outer_split(
    trials: list[ToneTrial], folds: int, fold: int, seed: int, strategy: str = "stimulus_grouped"
) -> tuple[list[int], list[int]]:
    if strategy == "stimulus_grouped":
        return grouped_outer_split(trials, folds, fold, seed)
    if strategy == "trial_stratified":
        return stratified_outer_split(trials, folds, fold, seed)
    raise ValueError(f"unsupported split_strategy={strategy!r}")


def make_train_val_split(
    indices: list[int],
    trials: list[ToneTrial],
    seed: int,
    fraction: float = 0.15,
    strategy: str = "stimulus_grouped",
) -> tuple[list[int], list[int]]:
    if strategy == "stimulus_grouped":
        return grouped_train_val(indices, trials, seed, fraction)
    if strategy == "trial_stratified":
        return stratified_train_val(indices, trials, seed, fraction)
    raise ValueError(f"unsupported split_strategy={strategy!r}")
