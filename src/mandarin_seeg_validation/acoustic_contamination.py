from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

from .common import configure_matplotlib, load_time_by_channel, save_figure


@dataclass(frozen=True)
class AnalysisConfig:
    max_trials: int = 30
    window_s: float = 0.200
    spectrogram_rate_hz: float = 50.0
    frequency_min_hz: float = 75.0
    frequency_max_hz: float = 1000.0
    permutations: int = 10000
    seed: int = 20261012
    time_axis: int = 0
    default_sfreq: float | None = None


@dataclass
class SessionResult:
    participant: str
    session: str
    task: str
    sampling_rate_hz: float
    n_channels: int
    n_paired_trials_total: int
    n_trials_analyzed: int
    n_time_frequency_frames: int
    max_same_frequency_r: float
    mean_top_1pct_same_frequency_r: float
    mean_frequencywise_max_r: float
    n_channel_frequency_r_ge_02: int
    n_channel_frequency_r_ge_03: int
    top_channel_index_0based: int
    top_frequency_hz: float
    contamination_diagonal_mean: float
    diagonal_dominance_p: float


def read_wav_mono(path: Path) -> tuple[np.ndarray, float]:
    sfreq, audio = wavfile.read(path)
    original_dtype = audio.dtype
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    return audio, float(sfreq)


def infer_sampling_rate(session_dir: Path, participant: str, fallback: float | None) -> float:
    for path in sorted(session_dir.glob("*_ieeg.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("SamplingFrequency", "sampling_frequency", "SamplingRate", "sampling_rate"):
            if key in metadata:
                return float(metadata[key])
    channels_path = session_dir.parent / f"{participant}_channels.tsv"
    if channels_path.exists():
        channels = pd.read_csv(channels_path, sep="\t")
        if "sampling_frequency" in channels.columns:
            values = pd.to_numeric(channels["sampling_frequency"], errors="coerce").dropna()
            if len(values):
                return float(values.iloc[0])
    if fallback is None:
        raise ValueError(f"Sampling frequency was not found for {participant}/{session_dir.name}")
    return float(fallback)


def infer_task(session_dir: Path) -> str:
    names = " ".join(path.name.lower() for path in session_dir.glob("*_events.tsv"))
    for task in ("sentence", "character"):
        if f"task-{task}" in names:
            return task
    return "unknown"


def paired_trials(session_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for ieeg_path in sorted((session_dir / "ieeg").glob("*_ieeg.npy")):
        audio_path = session_dir / "audio" / ieeg_path.name.replace("_ieeg.npy", "_audio.wav")
        if audio_path.exists():
            pairs.append((ieeg_path, audio_path))
    return pairs


def choose_trials(pairs: list[tuple[Path, Path]], maximum: int) -> list[tuple[Path, Path]]:
    if len(pairs) <= maximum:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, maximum).round().astype(int)
    return [pairs[index] for index in indices]


def subtract_moving_average(x: np.ndarray, sfreq: float, window_s: float = 1.0) -> np.ndarray:
    width = max(1, int(round(window_s * sfreq)))
    kernel = np.ones(width, dtype=np.float32) / width
    return x - signal.fftconvolve(x, kernel, mode="same").astype(np.float32)


def spectrogram_psd(
    x: np.ndarray,
    sfreq: float,
    config: AnalysisConfig,
    target_frequencies: np.ndarray | None = None,
    center_audio: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if center_audio:
        x = subtract_moving_average(x, sfreq)
    nperseg = max(16, int(round(config.window_s * sfreq)))
    hop = max(1, int(round(sfreq / config.spectrogram_rate_hz)))
    noverlap = max(0, nperseg - hop)
    pad = nperseg // 2
    if x.shape[0] > 2 and pad:
        x = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    frequencies, _, power = signal.spectrogram(
        x,
        fs=sfreq,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        scaling="density",
        mode="psd",
        axis=0,
    )
    if power.ndim == 3:
        power = np.moveaxis(power, -1, 1)
    keep = (frequencies >= config.frequency_min_hz) & (frequencies <= config.frequency_max_hz)
    frequencies = frequencies[keep]
    power = power[keep].astype(np.float32, copy=False)
    if len(frequencies) < 2:
        raise ValueError(
            f"No usable bins in {config.frequency_min_hz:g}-{config.frequency_max_hz:g} Hz; "
            "check the sampling rate"
        )
    if target_frequencies is None:
        return frequencies, power
    if power.ndim == 2:
        interpolated = np.empty((len(target_frequencies), power.shape[1]), dtype=np.float32)
        for frame in range(power.shape[1]):
            interpolated[:, frame] = np.interp(target_frequencies, frequencies, power[:, frame])
        return target_frequencies, interpolated
    interpolated = np.empty((len(target_frequencies), power.shape[1], power.shape[2]), dtype=np.float32)
    for channel in range(power.shape[2]):
        for frame in range(power.shape[1]):
            interpolated[:, frame, channel] = np.interp(
                target_frequencies, frequencies, power[:, frame, channel]
            )
    return target_frequencies, interpolated


def diagonal_dominance_test(
    matrix: np.ndarray, permutations: int, rng: np.random.Generator
) -> tuple[float, float, np.ndarray]:
    observed = float(np.nanmean(np.diag(matrix)))
    surrogate = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        if rng.integers(0, 2) == 0:
            shuffled = matrix[rng.permutation(matrix.shape[0]), :]
        else:
            shuffled = matrix[:, rng.permutation(matrix.shape[1])]
        surrogate[index] = np.nanmean(np.diag(shuffled))
    p_value = float(np.mean(surrogate >= observed))
    return observed, p_value, surrogate


def compute_session(
    session_dir: Path, config: AnalysisConfig, rng: np.random.Generator
) -> tuple[SessionResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[Path, Path]]]:
    participant = session_dir.parent.name
    session_name = session_dir.name
    task = infer_task(session_dir)
    sfreq = infer_sampling_rate(session_dir, participant, config.default_sfreq)
    pairs = paired_trials(session_dir)
    selected = choose_trials(pairs, config.max_trials)
    if not selected:
        raise RuntimeError(f"No paired audio/iEEG trials in {participant}/{session_name}")

    first = load_time_by_channel(selected[0][0], config.time_axis)
    n_channels = int(first.shape[1])
    frequencies_ref: np.ndarray | None = None
    n_frames = 0
    sum_audio = sum_audio_squared = None
    sum_neural = sum_neural_squared = None
    sum_products = None

    for ieeg_path, audio_path in selected:
        neural = np.asarray(load_time_by_channel(ieeg_path, config.time_axis), dtype=np.float32)
        if neural.shape[1] != n_channels:
            raise ValueError(f"Channel count changed in {ieeg_path.name}")
        frequencies, neural_power = spectrogram_psd(neural, sfreq, config)
        frequencies_ref = frequencies
        audio, audio_sfreq = read_wav_mono(audio_path)
        _, audio_power = spectrogram_psd(
            audio, audio_sfreq, config, target_frequencies=frequencies, center_audio=True
        )
        frames = min(audio_power.shape[1], neural_power.shape[1])
        if frames < 5:
            continue
        audio_frame = audio_power[:, :frames].T.astype(np.float64)
        neural_frame = np.transpose(neural_power[:, :frames, :], (1, 0, 2)).astype(np.float64)
        if sum_audio is None:
            n_frequencies = audio_frame.shape[1]
            sum_audio = np.zeros(n_frequencies, dtype=np.float64)
            sum_audio_squared = np.zeros(n_frequencies, dtype=np.float64)
            sum_neural = np.zeros((n_frequencies, n_channels), dtype=np.float64)
            sum_neural_squared = np.zeros((n_frequencies, n_channels), dtype=np.float64)
            sum_products = np.zeros((n_frequencies, n_frequencies, n_channels), dtype=np.float64)
        sum_audio += np.nansum(audio_frame, axis=0)
        sum_audio_squared += np.nansum(audio_frame**2, axis=0)
        sum_neural += np.nansum(neural_frame, axis=0)
        sum_neural_squared += np.nansum(neural_frame**2, axis=0)
        sum_products += np.einsum("ti,tjc->ijc", audio_frame, neural_frame, optimize=True)
        n_frames += frames

    if n_frames <= 2 or frequencies_ref is None or sum_audio is None:
        raise RuntimeError(f"Insufficient spectrogram frames in {participant}/{session_name}")

    numerator = n_frames * sum_products - sum_audio[:, None, None] * sum_neural[None, :, :]
    audio_variance = n_frames * sum_audio_squared - sum_audio**2
    neural_variance = n_frames * sum_neural_squared - sum_neural**2
    denominator = np.sqrt(
        np.maximum(audio_variance[:, None, None], 0.0)
        * np.maximum(neural_variance[None, :, :], 0.0)
    )
    correlation = numerator / np.where(denominator == 0, np.nan, denominator)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)

    same_frequency = np.stack(
        [correlation[index, index, :] for index in range(correlation.shape[0])], axis=0
    )
    contamination_matrix = np.nanmax(correlation, axis=2)
    diagonal_mean, p_value, surrogate = diagonal_dominance_test(
        contamination_matrix, config.permutations, rng
    )
    flat = same_frequency.ravel()
    top_count = max(1, int(math.ceil(flat.size * 0.01)))
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    maximum_position = np.unravel_index(int(np.nanargmax(same_frequency)), same_frequency.shape)
    frequencywise_max = np.nanmax(same_frequency, axis=1)

    result = SessionResult(
        participant=participant,
        session=session_name,
        task=task,
        sampling_rate_hz=sfreq,
        n_channels=n_channels,
        n_paired_trials_total=len(pairs),
        n_trials_analyzed=len(selected),
        n_time_frequency_frames=n_frames,
        max_same_frequency_r=float(np.nanmax(same_frequency)),
        mean_top_1pct_same_frequency_r=float(np.nanmean(flat[top_indices])),
        mean_frequencywise_max_r=float(np.nanmean(frequencywise_max)),
        n_channel_frequency_r_ge_02=int(np.sum(flat >= 0.2)),
        n_channel_frequency_r_ge_03=int(np.sum(flat >= 0.3)),
        top_channel_index_0based=int(maximum_position[1]),
        top_frequency_hz=float(frequencies_ref[maximum_position[0]]),
        contamination_diagonal_mean=diagonal_mean,
        diagonal_dominance_p=p_value,
    )
    return result, frequencies_ref, same_frequency, contamination_matrix, surrogate, frequencywise_max, selected


def matrix_figure(
    label: str,
    frequencies: np.ndarray,
    matrix: np.ndarray,
    surrogate: np.ndarray,
    observed: float,
    p_value: float,
    output: Path,
    formats: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.08, 1.0]})
    vmax = max(float(np.nanpercentile(matrix, 99.5)), 1e-6)
    image = axes[0].imshow(
        matrix.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]],
        vmin=0,
        vmax=vmax,
    )
    axes[0].plot([frequencies[0], frequencies[-1]], [frequencies[0], frequencies[-1]], color="white", lw=0.9)
    axes[0].set(xlabel="Audio frequency (Hz)", ylabel="sEEG frequency (Hz)", title=f"{label}\ncontamination matrix")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02).set_label("Maximum signed r")
    axes[1].hist(surrogate, bins=45, color="0.25", edgecolor="none")
    axes[1].axvline(observed, color="#E45756", lw=2)
    axes[1].set(xlabel="Mean diagonal r", ylabel="Permutation count", title=f"Diagonal dominance\nP = {p_value:.4f}")
    fig.tight_layout()
    save_figure(fig, output, formats)


def same_frequency_figure(
    label: str,
    frequencies: np.ndarray,
    same_frequency: np.ndarray,
    frequencywise_max: np.ndarray,
    output: Path,
    formats: tuple[str, ...],
) -> None:
    order = np.argsort(np.nanmax(same_frequency, axis=0))[::-1]
    shown = order[: min(50, len(order))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(frequencies, same_frequency, color="0.65", alpha=0.14, lw=0.7)
    axes[0].plot(frequencies, frequencywise_max, color="#4C78A8", lw=1.8)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Same-frequency audio-sEEG r", title=f"{label}\nsame-frequency profile")
    axes[0].grid(alpha=0.2)
    limit = max(float(np.nanpercentile(np.abs(same_frequency[:, shown]), 99)), 1e-6)
    image = axes[1].imshow(
        same_frequency[:, shown].T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=[frequencies[0], frequencies[-1], 0, len(shown)],
    )
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Channels sorted by maximum r", title="Top-channel heatmap")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02).set_label("Signed r")
    fig.tight_layout()
    save_figure(fig, output, formats)


def summary_figure(frame: pd.DataFrame, output: Path, formats: tuple[str, ...]) -> None:
    labels = frame["participant"] + " " + frame["session"] + " " + frame["task"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1.1, 1.25]})
    values = frame[["max_same_frequency_r", "mean_top_1pct_same_frequency_r", "contamination_diagonal_mean"]].to_numpy()
    image = axes[0].imshow(values, aspect="auto", cmap="magma", vmin=0)
    axes[0].set_yticks(np.arange(len(frame)), labels=labels)
    axes[0].set_xticks([0, 1, 2], labels=["Max diagonal r", "Top 1% mean r", "Matrix diagonal mean"])
    axes[0].set_title("Acoustic-contamination screen")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02).set_label("Pearson r")

    positions = np.arange(len(frame))
    axes[1].bar(positions, frame["n_channel_frequency_r_ge_02"], color="#4C78A8", label="r >= 0.20")
    axes[1].bar(positions, frame["n_channel_frequency_r_ge_03"], color="#F58518", label="r >= 0.30")
    axes[1].set_xticks(positions, labels=frame["participant"] + "-" + frame["session"], rotation=90)
    axes[1].set(ylabel="Channel-frequency pairs", title="Same-frequency counts")
    axes[1].legend(frameon=False)

    axes[2].scatter(frame["contamination_diagonal_mean"], frame["diagonal_dominance_p"], s=58, color="#54A24B", edgecolor="black", linewidth=0.5)
    axes[2].axhline(0.05, color="0.25", ls="--", lw=0.8)
    axes[2].set(xlabel="Mean diagonal contamination r", ylabel="Permutation P", title="Diagonal-dominance test", ylim=(-0.02, 1.02))
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output, formats)


def selected_session_dirs(root: Path, participants: set[str], sessions: set[str]) -> list[Path]:
    directories = []
    for path in sorted(root.glob("sub-*/ses-*")):
        if participants and path.parent.name not in participants:
            continue
        if sessions and path.name not in sessions:
            continue
        if (path / "audio").exists() and (path / "ieeg").exists():
            directories.append(path)
    return directories


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir).resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    config = AnalysisConfig(
        max_trials=args.max_trials,
        window_s=args.window_ms / 1000.0,
        spectrogram_rate_hz=args.spectrogram_rate,
        frequency_min_hz=args.freq_min,
        frequency_max_hz=args.freq_max,
        permutations=args.permutations,
        seed=args.seed,
        time_axis=args.time_axis,
        default_sfreq=args.default_sfreq,
    )
    participants = {item.strip() for item in args.participants.split(",") if item.strip()}
    sessions = {item.strip() for item in args.sessions.split(",") if item.strip()}
    session_dirs = selected_session_dirs(root, participants, sessions)
    if not session_dirs:
        raise RuntimeError("No session directories with paired audio and iEEG were found")

    configure_matplotlib()
    rng = np.random.default_rng(config.seed)
    results: list[SessionResult] = []
    cache: dict[str, tuple] = {}
    for session_dir in session_dirs:
        print(f"Analyzing {session_dir.parent.name}/{session_dir.name}...", flush=True)
        result, frequencies, same, matrix, surrogate, frequencywise_max, selected = compute_session(
            session_dir, config, rng
        )
        label = f"{result.participant}_{result.session}_{result.task}"
        results.append(result)
        cache[label] = (result, frequencies, same, matrix, surrogate, frequencywise_max, selected)
        matrix_figure(
            label,
            frequencies,
            matrix,
            surrogate,
            result.contamination_diagonal_mean,
            result.diagonal_dominance_p,
            figures / f"contamination_matrix_{label}",
            formats,
        )

    frame = pd.DataFrame([asdict(item) for item in results]).sort_values(["participant", "session"])
    frame.to_csv(tables / "session_screening.tsv", sep="\t", index=False)
    summary_figure(frame, figures / "session_summary", formats)

    top_row = frame.sort_values("max_same_frequency_r", ascending=False).iloc[0]
    min_p_row = frame.sort_values("diagonal_dominance_p").iloc[0]
    diagnostic_labels = list(
        dict.fromkeys(
            [
                f"{top_row.participant}_{top_row.session}_{top_row.task}",
                f"{min_p_row.participant}_{min_p_row.session}_{min_p_row.task}",
            ]
        )
    )
    for label in diagnostic_labels:
        _, frequencies, same, _, _, frequencywise_max, _ = cache[label]
        same_frequency_figure(
            label,
            frequencies,
            same,
            frequencywise_max,
            figures / f"same_frequency_{label}",
            formats,
        )

    summary = {
        "analysis": {
            "method": "Roussel-style spectrotemporal acoustic-contamination screen",
            "max_trials_per_session": config.max_trials,
            "window_s": config.window_s,
            "window": "Hamming",
            "spectrogram_sampling_hz": config.spectrogram_rate_hz,
            "power_transform": "raw PSD",
            "frequency_range_hz": [config.frequency_min_hz, config.frequency_max_hz],
            "correlation": "signed Pearson r",
            "contamination_matrix": "maximum signed r across channels",
            "n_permutations": config.permutations,
            "seed": config.seed,
        },
        "screening": {
            "sessions_analyzed": int(len(frame)),
            "maximum_same_frequency_r": float(frame["max_same_frequency_r"].max()),
            "median_session_maximum_r": float(frame["max_same_frequency_r"].median()),
            "minimum_diagonal_dominance_p": float(frame["diagonal_dominance_p"].min()),
            "sessions_with_p_lt_0_05": int((frame["diagonal_dominance_p"] < 0.05).sum()),
        },
        "diagnostic_sessions": diagnostic_labels,
    }
    (tables / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Roussel-style audio-sEEG acoustic-contamination screen")
    parser.add_argument("--dataset-root", required=True, help="Root of the controlled-access BIDS-like dataset")
    parser.add_argument("--output-dir", default="outputs/acoustic_contamination")
    parser.add_argument("--participants", default="", help="Optional comma-separated labels, e.g. sub-01,sub-02")
    parser.add_argument("--sessions", default="", help="Optional comma-separated session labels")
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--spectrogram-rate", type=float, default=50.0)
    parser.add_argument("--freq-min", type=float, default=75.0)
    parser.add_argument("--freq-max", type=float, default=1000.0)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--time-axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--default-sfreq", type=float, default=None)
    parser.add_argument("--formats", default="png,pdf,svg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
