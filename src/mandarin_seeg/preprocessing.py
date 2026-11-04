from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from .manifest import ToneTrial


SAMPLING_RATES = {
    **{f"sub-{number:02d}": 2000 for number in range(1, 6)},
    **{f"sub-{number:02d}": 2048 for number in range(6, 13)},
}

KNOWN_ACTIVE_CHANNELS = {
    "sub-06": 126,
    "sub-07": 128,
    "sub-08": 108,
    "sub-10": 138,
    "sub-11": 126,
    "sub-12": 146,
}


def _read_textgrid(path: Path) -> str | None:
    if not path.exists():
        return None
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return None


@lru_cache(maxsize=20000)
def textgrid_xmax(root: str, subject: str, session: str, stem: str) -> float | None:
    text = _read_textgrid(Path(root) / "sourcedata" / "annotations" / subject / session / f"{stem}.TextGrid")
    if text is None:
        return None
    values = [float(value) for value in re.findall(r"\bxmax\s*=\s*([0-9.]+)", text)]
    return max(values) if values else None


def _invalid_threshold(config: dict | None = None) -> float:
    if config is None:
        return 1e20
    return float(config.get("invalid_value_threshold", 1e20))


def _valid_sample_mask(values: np.ndarray, config: dict | None = None) -> np.ndarray:
    return np.isfinite(values) & (np.abs(values) < _invalid_threshold(config))


def _use_textgrid_onset_correction(config: dict | None = None) -> bool:
    if config is None:
        return False
    return bool(config.get("use_textgrid_onset_correction", False))


def valid_channels(subject: str, array: np.ndarray, config: dict | None = None) -> np.ndarray:
    upper = min(KNOWN_ACTIVE_CHANNELS.get(subject, array.shape[1]), array.shape[1])
    sample = np.asarray(array[:: max(1, array.shape[0] // 1200), :upper], dtype=np.float64)
    valid = _valid_sample_mask(sample, config)
    valid_fraction = valid.mean(axis=0)
    safe = np.where(valid, sample, np.nan)
    spread = np.nanstd(safe, axis=0)
    rms = np.sqrt(np.nanmean(safe * safe, axis=0))

    min_valid_fraction = float(config.get("min_channel_valid_fraction", 0.98)) if config else 0.98
    min_std = float(config.get("min_channel_std", 1e-10)) if config else 1e-10
    keep = valid_fraction >= min_valid_fraction
    keep &= np.isfinite(spread) & (spread > min_std)

    finite_rms = rms[np.isfinite(rms) & (rms > 0)]
    if finite_rms.size >= 8:
        median = float(np.median(finite_rms))
        mad = float(np.median(np.abs(finite_rms - median)))
        if mad > 0:
            robust_z = np.abs(rms - median) / (1.4826 * mad)
            max_robust_z = float(config.get("max_channel_rms_robust_z", 12.0)) if config else 12.0
            keep &= robust_z <= max_robust_z
    return np.flatnonzero(keep)


def corrected_onset(root: str | Path, trial: ToneTrial, n_samples: int, config: dict | None = None) -> float:
    fs = SAMPLING_RATES[trial.subject]
    duration = n_samples / fs
    onset = trial.start
    if _use_textgrid_onset_correction(config) and int(trial.subject[-2:]) >= 6:
        xmax = textgrid_xmax(str(Path(root)), trial.subject, trial.session, Path(trial.recording_file).stem)
        if xmax is not None:
            onset -= max(0.0, xmax - duration)
    return onset


def _trial_window(root: str | Path, trial: ToneTrial, config: dict) -> tuple[np.ndarray, int, int, int, int, float]:
    array = np.load(trial.npy_path, mmap_mode="r", allow_pickle=False)
    fs = SAMPLING_RATES[trial.subject]
    onset = corrected_onset(root, trial, int(array.shape[0]), config)
    duration = int(array.shape[0]) / fs
    window_start = float(config["window_start"])
    mode = config.get("window_mode", "fixed")
    if mode == "event_normalized":
        event_duration = trial.end - trial.start
        if event_duration <= 0:
            raise ValueError("non_positive_event_duration")
        start_time = onset + window_start
        stop_time = onset + event_duration
    elif mode == "fixed":
        start_time = onset + window_start
        stop_time = onset + float(config["window_end"])
    else:
        raise ValueError(f"unsupported window_mode={mode!r}")
    tolerance = 0.5 / fs
    if start_time < -tolerance:
        raise ValueError("insufficient_pre_event_margin")
    if stop_time > duration + tolerance:
        raise ValueError("event_exceeds_recording")
    if stop_time <= start_time:
        raise ValueError("invalid_window")
    start = int(round(start_time * fs))
    onset_sample = int(round(onset * fs))
    stop = int(round(stop_time * fs))
    if start < 0 or stop > array.shape[0] or stop <= start:
        raise ValueError("invalid_window_samples")
    return array, fs, start, onset_sample, stop, duration


def trial_window_status(root: str | Path, trial: ToneTrial, config: dict) -> tuple[bool, str]:
    """Check only NPY metadata and label timing; do not load the signal into RAM."""
    try:
        _trial_window(root, trial, config)
    except ValueError as error:
        return False, str(error)
    return True, "ok"


def _repair_invalid_samples(raw: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    valid = _valid_sample_mask(raw, config)
    channel_invalid_fraction = 1.0 - valid.mean(axis=0)
    max_channel_invalid = float(config.get("max_channel_invalid_fraction", 0.02))
    keep_channels = channel_invalid_fraction <= max_channel_invalid
    min_channels = int(config.get("min_valid_channels", 4))
    details: dict[str, object] = {
        "window_invalid_fraction": float(1.0 - valid.mean()),
        "max_channel_invalid_fraction_seen": float(channel_invalid_fraction.max()) if channel_invalid_fraction.size else math.nan,
        "n_channels_before_window_qc": int(raw.shape[1]),
        "n_channels_after_window_qc": int(keep_channels.sum()),
        "n_repaired_samples": int((~valid[:, keep_channels]).sum()) if keep_channels.any() else 0,
    }
    if int(keep_channels.sum()) < min_channels:
        raise ValueError("too_few_usable_window_channels")

    repaired = np.asarray(raw[:, keep_channels], dtype=np.float64).copy()
    valid = valid[:, keep_channels]
    times = np.arange(repaired.shape[0])
    for channel in range(repaired.shape[1]):
        mask = valid[:, channel]
        if mask.all():
            continue
        if int(mask.sum()) < 2:
            raise ValueError("too_few_valid_samples_for_repair")
        repaired[:, channel] = np.interp(times, times[mask], repaired[mask, channel])
    return repaired, keep_channels, details


def trial_signal_qc(root: str | Path, trial: ToneTrial, config: dict) -> tuple[bool, str, dict[str, object]]:
    if not bool(config.get("enable_signal_qc", True)):
        return True, "ok", {}
    array, fs, start, _, stop, _ = _trial_window(root, trial, config)
    channels = valid_channels(trial.subject, array, config)
    details: dict[str, object] = {
        "n_samples": int(array.shape[0]),
        "n_channels_total": int(array.shape[1]),
        "n_valid_channels": int(len(channels)),
        "window_start_sample": int(start),
        "window_stop_sample": int(stop),
    }
    if len(channels) < int(config.get("min_valid_channels", 4)):
        return False, "too_few_valid_channels", details
    raw = np.asarray(array[start:stop, channels], dtype=np.float64)
    try:
        _, _, repair_details = _repair_invalid_samples(raw, config)
    except ValueError as error:
        return False, str(error), details
    details.update(repair_details)
    if float(details["window_invalid_fraction"]) > float(config.get("max_trial_invalid_fraction", 0.05)):
        return False, "too_many_invalid_samples", details
    baseline_start, baseline_end = (float(value) for value in config["baseline"])
    b0 = int(round((baseline_start - float(config["window_start"])) * fs))
    b1 = int(round((baseline_end - float(config["window_start"])) * fs))
    b0 = max(0, min(raw.shape[0], b0))
    b1 = max(0, min(raw.shape[0], b1))
    details["baseline_samples"] = int(max(0, b1 - b0))
    if b1 <= b0:
        return False, "invalid_baseline_window", details
    baseline_valid = _valid_sample_mask(raw[b0:b1], config)
    details["baseline_valid_fraction"] = float(baseline_valid.mean())
    if float(details["baseline_valid_fraction"]) < float(config.get("min_baseline_valid_fraction", 0.95)):
        return False, "invalid_baseline_samples", details
    return True, "ok", details


def filter_valid_trials(
    root: str | Path, trials: list[ToneTrial], config: dict
) -> tuple[list[ToneTrial], list[dict[str, object]]]:
    valid: list[ToneTrial] = []
    excluded: list[dict[str, object]] = []
    for trial in trials:
        try:
            is_valid, reason = trial_window_status(root, trial, config)
            details: dict[str, object] = {}
            if is_valid:
                is_valid, reason, details = trial_signal_qc(root, trial, config)
        except Exception as error:  # keep a precise path-level report for manual audit
            is_valid, reason = False, f"{type(error).__name__}: {error}"
            details = {}
        if is_valid:
            valid.append(trial)
        else:
            excluded.append(
                {
                    "subject": trial.subject,
                    "session": trial.session,
                    "recording_file": trial.recording_file,
                    "npy_path": trial.npy_path,
                    "start": trial.start,
                    "end": trial.end,
                    "reason": reason,
                    **details,
                }
            )
    return valid, excluded


def _sos_band(low: float, high: float, fs: float) -> np.ndarray:
    return signal.butter(4, [low, min(high, fs / 2 - 1)], btype="bandpass", fs=fs, output="sos")


def _notch(data: np.ndarray, fs: float, frequencies: list[float]) -> np.ndarray:
    output = data
    for frequency in frequencies:
        if frequency >= fs / 2 - 1:
            continue
        b, a = signal.iirnotch(frequency, 30.0, fs=fs)
        output = signal.filtfilt(b, a, output, axis=0)
    return output


def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(data, target_rate // divisor, source_rate // divisor, axis=0)


def preprocess_tone_trial(root: str | Path, trial: ToneTrial, config: dict) -> tuple[np.ndarray, np.ndarray]:
    array, fs, start, onset_sample, stop, _ = _trial_window(root, trial, config)
    window_start = float(config["window_start"])
    mode = config.get("window_mode", "fixed")
    if mode not in {"event_normalized", "fixed"}:
        raise ValueError(f"unsupported window_mode={mode!r}")
    channels = valid_channels(trial.subject, array, config)
    if len(channels) < 4:
        raise ValueError(f"too few valid channels for {trial.recording_file}")
    raw = np.asarray(array[start:stop, channels], dtype=np.float64)
    del array
    raw, _, _ = _repair_invalid_samples(raw, config)
    raw -= raw.mean(axis=1, keepdims=True)  # common average reference
    raw = _notch(raw, fs, [float(value) for value in config["notch_frequencies"]])
    low_band = tuple(float(value) for value in config["low_band"])
    high_band = tuple(float(value) for value in config["high_gamma_band"])
    low = signal.sosfiltfilt(_sos_band(*low_band, fs), raw, axis=0)
    high = signal.sosfiltfilt(_sos_band(*high_band, fs), raw, axis=0)
    high = np.abs(signal.hilbert(high, axis=0))
    target_rate = int(config["target_rate"])
    if mode == "event_normalized":
        split = onset_sample - start
        pre_samples = int(round(-window_start * target_rate))
        event_samples = int(round(float(config["normalized_event_duration"]) * target_rate))

        def normalize_event(data: np.ndarray) -> np.ndarray:
            pre = signal.resample(data[:split], pre_samples, axis=0)
            event = signal.resample(data[split:], event_samples, axis=0)
            return np.concatenate([pre, event], axis=0)

        low = normalize_event(low)
        high = normalize_event(high)
        output_window_end = float(config["normalized_event_duration"])
    else:
        low = _resample(low, fs, target_rate)
        high = _resample(high, fs, target_rate)
        output_window_end = float(config["window_end"])
    streams = np.stack([low, high], axis=1)  # [time, stream, channel]

    baseline_start, baseline_end = (float(value) for value in config["baseline"])
    b0 = int(round((baseline_start - window_start) * target_rate))
    b1 = int(round((baseline_end - window_start) * target_rate))
    if not (0 <= b0 < b1 <= streams.shape[0]):
        raise ValueError(
            f"baseline {config['baseline']} is outside output window "
            f"[{window_start}, {output_window_end}]"
        )
    baseline = streams[b0:b1]
    center = baseline.mean(axis=0, keepdims=True)
    scale = baseline.std(axis=0, keepdims=True)
    streams = (streams - center) / np.maximum(scale, 1e-6)
    streams = np.clip(streams, -12.0, 12.0)
    output = np.transpose(streams, (2, 1, 0)).astype(np.float32)  # [channel, stream, time]
    mask = np.ones(output.shape[0], dtype=bool)
    return output, mask
