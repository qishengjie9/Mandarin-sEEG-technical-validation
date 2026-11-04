from __future__ import annotations

from collections import OrderedDict
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from .manifest import ToneTrial
from .preprocessing import preprocess_tone_trial


class ToneDataset(Dataset):
    def __init__(
        self,
        root: str,
        trials: list[ToneTrial],
        preprocessing: dict,
        cache_items: int = 1024,
    ) -> None:
        self.root = root
        self.trials = trials
        self.preprocessing = preprocessing
        self.cache_items = max(0, int(cache_items))
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.trials)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        value = preprocess_tone_trial(self.root, self.trials[index], self.preprocessing)
        if self.cache_items:
            self.cache[index] = (value[0].astype(np.float16), value[1])
            while len(self.cache) > self.cache_items:
                self.cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, object]:
        signal, channel_mask = self._load(index)
        trial = self.trials[index]
        return {
            "signal": torch.from_numpy(signal.astype(np.float32, copy=False)),
            "channel_mask": torch.from_numpy(channel_mask),
            "tone": torch.tensor(trial.tone, dtype=torch.long),
            "stimulus": trial.stimulus,
            "recording_file": trial.recording_file,
        }


class StimulusBatchSampler(Sampler[list[int]]):
    """Create batches containing cross-session examples of the same stimulus when possible."""

    def __init__(self, trials: list[ToneTrial], batch_size: int, seed: int = 0) -> None:
        if batch_size < 2:
            raise ValueError("StimulusBatchSampler requires batch_size >= 2")
        self.trials = trials
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.trials) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        grouped: dict[str, list[int]] = {}
        for index, trial in enumerate(self.trials):
            grouped.setdefault(trial.stimulus, []).append(index)

        units: list[list[int]] = []
        singles: list[int] = []
        for indices in grouped.values():
            rng.shuffle(indices)
            if len(indices) == 1:
                singles.extend(indices)
                continue
            for start in range(0, len(indices) - 1, 2):
                units.append(indices[start : start + 2])
            if len(indices) % 2:
                # Pair the leftover trial with another session/example from its stimulus.
                units.append([indices[-1], indices[0]])
        rng.shuffle(units)
        rng.shuffle(singles)
        units.extend([[index] for index in singles])

        batch: list[int] = []
        for unit in units:
            if batch and len(batch) + len(unit) > self.batch_size:
                yield batch
                batch = []
            batch.extend(unit)
        if batch:
            yield batch


def collate_variable_channels(batch: list[dict[str, object]]) -> dict[str, object]:
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
        "stimulus": [item["stimulus"] for item in batch],
        "recording_file": [item["recording_file"] for item in batch],
    }
