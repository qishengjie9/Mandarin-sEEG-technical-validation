from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ToneTrial:
    subject: str
    session: str
    recording_file: str
    npy_path: str
    start: float
    end: float
    tone: int
    stimulus: str
    pinyin: str
    character: str
    original_file_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def _session(recording_file: str) -> str:
    match = re.search(r"_(ses-\d+)_", recording_file)
    if not match:
        raise ValueError(f"cannot parse session from {recording_file}")
    return match.group(1)


def build_tone_manifest(root: str | Path, subject: str) -> list[ToneTrial]:
    root = Path(root)
    subject_dir = root / subject
    trials: list[ToneTrial] = []
    labels = sorted(
        {
            *subject_dir.glob(f"{subject}_ses-*_task-character_desc-words_events.tsv"),
            *subject_dir.glob(f"ses-*/{subject}_ses-*_task-character_desc-words_events.tsv"),
        }
    )
    if not labels:
        raise FileNotFoundError(f"no character word labels below {subject_dir}")
    for label_path in labels:
        for row in _deduplicate(_read_tsv(label_path)):
            tone_text = row.get("tone", "")
            if tone_text not in {"1", "2", "3", "4"}:
                continue
            recording_file = row.get("recording_file", "")
            if not recording_file:
                continue
            session = _session(recording_file)
            npy_path = subject_dir / session / "ieeg" / recording_file
            if not npy_path.exists():
                raise FileNotFoundError(npy_path)
            pinyin = row.get("word", "")
            character = row.get("character", "")
            stimulus = character or pinyin or row.get("original_file_name", recording_file)
            trials.append(
                ToneTrial(
                    subject=subject,
                    session=session,
                    recording_file=recording_file,
                    npy_path=str(npy_path),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    tone=int(tone_text) - 1,
                    stimulus=stimulus,
                    pinyin=pinyin,
                    character=character,
                    original_file_name=row.get("original_file_name", ""),
                )
            )
    return trials


def balanced_limit(trials: list[ToneTrial], max_trials: int | None, seed: int) -> list[ToneTrial]:
    if max_trials is None or len(trials) <= max_trials:
        return trials
    rng = np.random.default_rng(seed)
    per_class = max_trials // 4
    selected: list[ToneTrial] = []
    for tone in range(4):
        items = [trial for trial in trials if trial.tone == tone]
        count = min(per_class, len(items))
        indices = rng.choice(len(items), count, replace=False)
        selected.extend(items[int(index)] for index in indices)
    return sorted(selected, key=lambda trial: (trial.session, trial.recording_file))


def write_manifest(path: str | Path, trials: list[ToneTrial]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0].to_dict()))
        writer.writeheader()
        writer.writerows(trial.to_dict() for trial in trials)
