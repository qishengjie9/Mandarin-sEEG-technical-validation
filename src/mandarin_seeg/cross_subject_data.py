from __future__ import annotations

import hashlib
import json
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset, Sampler

from .manifest import ToneTrial, balanced_limit, build_tone_manifest
from .phonology import parse_pinyin
from .preprocessing import filter_valid_trials, preprocess_tone_trial


@dataclass(frozen=True)
class CrossSubjectSplit:
    train: list[ToneTrial]
    val: list[ToneTrial]
    test: list[ToneTrial]
    train_stimuli: tuple[str, ...]
    val_stimuli: tuple[str, ...]
    test_stimuli: tuple[str, ...]


def build_multisubject_trials(
    root: str | Path,
    subjects: list[str],
    preprocessing: dict,
    max_trials_per_subject: int | None,
    seed: int,
) -> tuple[list[ToneTrial], list[dict[str, object]]]:
    trials: list[ToneTrial] = []
    excluded: list[dict[str, object]] = []
    for offset, subject in enumerate(subjects):
        subject_trials, subject_excluded = filter_valid_trials(
            root, build_tone_manifest(root, subject), preprocessing
        )
        subject_trials = balanced_limit(subject_trials, max_trials_per_subject, seed + offset)
        trials.extend(subject_trials)
        excluded.extend(subject_excluded)
    return trials, excluded


def _stimulus_table(trials: list[ToneTrial]) -> tuple[np.ndarray, np.ndarray]:
    labels: dict[str, int] = {}
    for trial in trials:
        if trial.stimulus in labels and labels[trial.stimulus] != trial.tone:
            raise ValueError(f"stimulus has inconsistent tones: {trial.stimulus}")
        labels[trial.stimulus] = trial.tone
    stimuli = np.asarray(sorted(labels))
    tones = np.asarray([labels[value] for value in stimuli])
    return stimuli, tones


def cross_subject_stimulus_split(
    trials: list[ToneTrial],
    heldout_subjects: list[str],
    fold: int,
    folds: int,
    seed: int,
    val_fraction: float = 0.1,
    strict_unseen_stimulus: bool = True,
) -> CrossSubjectSplit:
    heldout = set(heldout_subjects)
    development = [trial for trial in trials if trial.subject not in heldout]
    test_subject_trials = [trial for trial in trials if trial.subject in heldout]
    if not development or not test_subject_trials:
        raise ValueError("both development and held-out subject trials are required")

    stimuli, tones = _stimulus_table(trials)
    if strict_unseen_stimulus:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        train_stimulus_indices, test_stimulus_indices = list(splitter.split(stimuli, tones))[fold]
        candidate_stimuli = stimuli[train_stimulus_indices]
        candidate_tones = tones[train_stimulus_indices]
        test_stimuli = set(stimuli[test_stimulus_indices].tolist())
    else:
        candidate_stimuli = stimuli
        candidate_tones = tones
        test_stimuli = set(stimuli.tolist())

    val_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed + 100 + fold
    )
    train_indices, val_indices = next(
        val_splitter.split(candidate_stimuli, candidate_tones)
    )
    train_stimuli = set(candidate_stimuli[train_indices].tolist())
    val_stimuli = set(candidate_stimuli[val_indices].tolist())

    split = CrossSubjectSplit(
        train=[trial for trial in development if trial.stimulus in train_stimuli],
        val=[trial for trial in development if trial.stimulus in val_stimuli],
        test=[trial for trial in test_subject_trials if trial.stimulus in test_stimuli],
        train_stimuli=tuple(sorted(train_stimuli)),
        val_stimuli=tuple(sorted(val_stimuli)),
        test_stimuli=tuple(sorted(test_stimuli)),
    )
    if set(trial.subject for trial in split.train) & set(trial.subject for trial in split.test):
        raise RuntimeError("subject leakage between train and test")
    if strict_unseen_stimulus and set(split.train_stimuli) & set(split.test_stimuli):
        raise RuntimeError("stimulus leakage between train and test")
    if set(split.train_stimuli) & set(split.val_stimuli):
        raise RuntimeError("stimulus leakage between train and validation")
    return split


class PhonologyDataset(Dataset):
    def __init__(
        self,
        root: str,
        trials: list[ToneTrial],
        preprocessing: dict,
        subject_to_index: dict[str, int],
        cache_items: int = 512,
        disk_cache: str | Path | None = None,
    ) -> None:
        self.root = root
        self.trials = trials
        self.preprocessing = preprocessing
        self.subject_to_index = subject_to_index
        self.cache_items = max(0, int(cache_items))
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.disk_cache = Path(disk_cache) if disk_cache else None
        if self.disk_cache is not None:
            self.disk_cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.trials)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        cache_path = self._cache_path(index)
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            signal = cached["signal"]
            mask = cached["mask"].astype(bool, copy=False)
            self._remember(index, signal, mask)
            return signal, mask
        signal, mask = preprocess_tone_trial(self.root, self.trials[index], self.preprocessing)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(temporary, signal=signal.astype(np.float16), mask=mask)
            os.replace(temporary, cache_path)
        self._remember(index, signal, mask)
        return signal, mask

    def _remember(self, index: int, signal: np.ndarray, mask: np.ndarray) -> None:
        if self.cache_items:
            self.cache[index] = (signal.astype(np.float16), mask)
            while len(self.cache) > self.cache_items:
                self.cache.popitem(last=False)

    def _cache_path(self, index: int) -> Path | None:
        if self.disk_cache is None:
            return None
        trial = self.trials[index]
        payload = {
            "subject": trial.subject,
            "session": trial.session,
            "recording_file": trial.recording_file,
            "start": trial.start,
            "end": trial.end,
            "preprocessing": self.preprocessing,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.disk_cache / trial.subject / f"{digest}.npz"

    def __getitem__(self, index: int) -> dict[str, object]:
        trial = self.trials[index]
        signal, channel_mask = self._load(index)
        label = parse_pinyin(trial.pinyin)
        return {
            "signal": torch.from_numpy(signal.astype(np.float32, copy=False)),
            "channel_mask": torch.from_numpy(channel_mask),
            "tone": torch.tensor(label.tone, dtype=torch.long),
            "initial": torch.tensor(label.initial_index, dtype=torch.long),
            "final": torch.tensor(label.final_index, dtype=torch.long),
            "subject_index": torch.tensor(self.subject_to_index.get(trial.subject, -1), dtype=torch.long),
            "subject": trial.subject,
            "stimulus": trial.stimulus,
            "pinyin": trial.pinyin,
            "recording_file": trial.recording_file,
        }


def collate_phonology(batch: list[dict[str, object]]) -> dict[str, object]:
    max_channels = max(int(item["signal"].shape[0]) for item in batch)
    streams = int(batch[0]["signal"].shape[1])
    times = int(batch[0]["signal"].shape[2])
    signals = torch.zeros(len(batch), max_channels, streams, times, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_channels, dtype=torch.bool)
    for index, item in enumerate(batch):
        channels = int(item["signal"].shape[0])
        signals[index, :channels] = item["signal"]
        masks[index, :channels] = item["channel_mask"]
    return {
        "signal": signals,
        "channel_mask": masks,
        "tone": torch.stack([item["tone"] for item in batch]),
        "initial": torch.stack([item["initial"] for item in batch]),
        "final": torch.stack([item["final"] for item in batch]),
        "subject_index": torch.stack([item["subject_index"] for item in batch]),
        "subject": [item["subject"] for item in batch],
        "stimulus": [item["stimulus"] for item in batch],
        "pinyin": [item["pinyin"] for item in batch],
        "recording_file": [item["recording_file"] for item in batch],
    }


class SubjectBalancedBatchSampler(Sampler[list[int]]):
    """Equalizes subject contribution and cycles through tones within each subject."""

    def __init__(self, trials: list[ToneTrial], batch_size: int, seed: int = 0) -> None:
        if batch_size < 4:
            raise ValueError("batch_size must be at least four")
        self.trials = trials
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.subjects = sorted({trial.subject for trial in trials})
        self.max_subject_trials = max(sum(trial.subject == subject for trial in trials) for subject in self.subjects)

    def __len__(self) -> int:
        target_examples = self.max_subject_trials * len(self.subjects)
        return (target_examples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, trial in enumerate(self.trials):
            pools[(trial.subject, trial.tone)].append(index)
        for values in pools.values():
            rng.shuffle(values)
        cursors = defaultdict(int)

        def draw(subject: str, tone: int) -> int:
            key = (subject, tone)
            values = pools[key]
            if not values:
                alternatives = [index for candidate_tone in range(4) for index in pools[(subject, candidate_tone)]]
                return rng.choice(alternatives)
            cursor = cursors[key]
            if cursor >= len(values):
                rng.shuffle(values)
                cursor = 0
            cursors[key] = cursor + 1
            return values[cursor]

        subject_cursor = 0
        tone_cursor = rng.randrange(4)
        for _ in range(len(self)):
            batch = []
            while len(batch) < self.batch_size:
                subject = self.subjects[subject_cursor % len(self.subjects)]
                tone = tone_cursor % 4
                batch.append(draw(subject, tone))
                subject_cursor += 1
                tone_cursor += 1
                if subject_cursor % len(self.subjects) == 0:
                    rng.shuffle(self.subjects)
            yield batch
