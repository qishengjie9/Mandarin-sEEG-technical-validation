from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .common import configure_matplotlib, load_time_by_channel, parse_pair, save_figure


FREQUENCY_BANDS = {
    "Delta (1-4 Hz)": (1.0, 4.0),
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Alpha (8-13 Hz)": (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
    "Gamma (30-70 Hz)": (30.0, 70.0),
    "High-gamma (70-150 Hz)": (70.0, 150.0),
}


def preprocess(
    x: np.ndarray,
    sfreq: float,
    bandpass: tuple[float, float],
    notch_frequencies: tuple[float, ...],
) -> np.ndarray:
    y = signal.detrend(np.asarray(x, dtype=np.float64), type="linear")
    nyquist = sfreq / 2.0
    if not (0 < bandpass[0] < bandpass[1] < nyquist):
        raise ValueError(f"Band-pass {bandpass} must lie below Nyquist ({nyquist:g} Hz)")
    sos = signal.butter(4, bandpass, btype="bandpass", fs=sfreq, output="sos")
    y = signal.sosfiltfilt(sos, y)
    for frequency in notch_frequencies:
        if 0 < frequency < nyquist:
            b, a = signal.iirnotch(frequency, Q=35.0, fs=sfreq)
            y = signal.filtfilt(b, a, y)
    return y


def morlet_power(x: np.ndarray, frequencies: np.ndarray, sfreq: float) -> np.ndarray:
    power = np.empty((len(frequencies), x.size), dtype=np.float32)
    cycles = np.interp(frequencies, [frequencies.min(), frequencies.max()], [3.0, 10.0])
    for index, (frequency, n_cycles) in enumerate(zip(frequencies, cycles)):
        sigma_t = n_cycles / (2.0 * np.pi * frequency)
        half_width = 3.5 * sigma_t
        wavelet_t = np.arange(-half_width, half_width + 1.0 / sfreq, 1.0 / sfreq)
        wavelet = np.exp(2j * np.pi * frequency * wavelet_t)
        wavelet *= np.exp(-(wavelet_t**2) / (2.0 * sigma_t**2))
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        power[index] = np.abs(signal.fftconvolve(x, wavelet, mode="same")) ** 2
    return power


def baseline_db(power: np.ndarray, time: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    mask = (time >= baseline[0]) & (time <= baseline[1])
    if mask.sum() < 2:
        raise ValueError("Baseline window does not contain enough samples")
    reference = np.mean(power[:, mask], axis=1, keepdims=True)
    eps = np.finfo(np.float32).tiny
    return 10.0 * np.log10((power + eps) / (reference + eps))


def add_windows(ax: plt.Axes, baseline: tuple[float, float], speech: tuple[float, float]) -> None:
    ax.axvspan(*baseline, color="#B7B7B7", alpha=0.20, lw=0)
    ax.axvspan(*speech, color="#F4A261", alpha=0.12, lw=0)
    ax.axvline(speech[0], color="#C65D2E", lw=1.0, ls="--")
    ax.axvline(speech[1], color="#C65D2E", lw=1.0, ls="--")


def run(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    baseline = parse_pair(args.baseline, "baseline")
    speech = parse_pair(args.speech, "speech")
    bandpass = parse_pair(args.bandpass, "bandpass")
    notches = tuple(float(item) for item in args.notch.split(",") if item.strip())

    data = load_time_by_channel(input_path, args.time_axis)
    if args.channel_index < 0 or args.channel_index >= data.shape[1]:
        raise IndexError(f"Channel index {args.channel_index} is outside 0-{data.shape[1] - 1}")
    raw = np.asarray(data[:, args.channel_index], dtype=np.float64)
    if args.input_unit == "uV":
        raw *= 1e-6
    time = np.arange(raw.size) / args.sfreq
    if speech[1] > time[-1] or baseline[1] > time[-1]:
        raise ValueError(f"Analysis windows exceed recording duration ({time[-1]:.3f} s)")
    clean = preprocess(raw, args.sfreq, bandpass, notches)

    max_frequency = min(args.tfr_max, bandpass[1] - 5.0, args.sfreq / 2.0 - 5.0)
    if max_frequency <= args.tfr_min:
        raise ValueError("No valid TFR frequency range remains after Nyquist/filter checks")
    frequencies = np.geomspace(args.tfr_min, max_frequency, args.n_frequencies)
    power = morlet_power(clean, frequencies, args.sfreq)
    power_db = baseline_db(power, time, baseline)

    band_db: dict[str, np.ndarray] = {}
    baseline_mask = (time >= baseline[0]) & (time <= baseline[1])
    for name, (low, high) in FREQUENCY_BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            continue
        linear = np.mean(power[mask], axis=0)
        band_db[name] = 10.0 * np.log10(linear / np.mean(linear[baseline_mask]))

    channel_label = args.channel_label or f"channel-{args.channel_index}"
    configure_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.6), sharex=True)
    axes[0].plot(time, raw * 1e6, color="0.55", lw=0.55)
    axes[1].plot(time, clean * 1e6, color="#D55E5E", lw=0.55)
    axes[0].set(title=f"{channel_label}: signal before filtering", ylabel="Amplitude (uV)")
    axes[1].set(title="Signal after band-pass and notch filtering", ylabel="Amplitude (uV)", xlabel="Time (s)")
    for ax in axes:
        add_windows(ax, baseline, speech)
    fig.tight_layout()
    save_figure(fig, output_dir / "signal_preprocessing", formats)

    nperseg = min(raw.size, max(256, int(round(2.0 * args.sfreq))))
    noverlap = nperseg // 2
    f_raw, p_raw = signal.welch(raw, fs=args.sfreq, nperseg=nperseg, noverlap=noverlap)
    f_clean, p_clean = signal.welch(clean, fs=args.sfreq, nperseg=nperseg, noverlap=noverlap)
    keep = (f_raw >= bandpass[0]) & (f_raw <= min(args.psd_max, args.sfreq / 2.0))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.semilogy(f_raw[keep], p_raw[keep] * 1e12, color="0.65", lw=0.9, label="Before filtering")
    ax.semilogy(f_clean[keep], p_clean[keep] * 1e12, color="#0072B2", lw=1.1, label="After filtering")
    for frequency in notches:
        if frequency <= args.psd_max:
            ax.axvline(frequency, color="#D55E00", lw=0.8, ls="--", alpha=0.75)
    ax.axvspan(70, 150, color="#009E73", alpha=0.10, label="High-gamma")
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (uV^2/Hz)", title="Power spectral density")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "power_spectral_density", formats)

    edge = min(0.25, max(0.0, time[-1] / 10.0))
    valid = (time >= edge) & (time <= time[-1] - edge)
    vmax = max(float(np.nanpercentile(np.abs(power_db[:, valid]), 98)), 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    mesh = ax.pcolormesh(time, frequencies, power_db, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
    ax.set_yscale("log")
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Morlet time-frequency power relative to baseline")
    add_windows(ax, baseline, speech)
    fig.colorbar(mesh, ax=ax, pad=0.015).set_label("Power change (dB)")
    fig.tight_layout()
    save_figure(fig, output_dir / "morlet_time_frequency", formats)

    smooth_n = max(1, int(round(args.smoothing_ms / 1000.0 * args.sfreq)))
    kernel = np.ones(smooth_n) / smooth_n
    fig, axes = plt.subplots(len(band_db), 1, figsize=(10.5, 1.45 * len(band_db)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, values) in zip(axes, band_db.items()):
        ax.plot(time, np.convolve(values, kernel, mode="same"), lw=1.0)
        ax.axhline(0, color="0.75", lw=0.6)
        add_windows(ax, baseline, speech)
        ax.set(title=name, ylabel="dB")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_figure(fig, output_dir / "frequency_band_time_courses", formats)

    high_gamma = band_db.get("High-gamma (70-150 Hz)")
    speech_mask = (time >= speech[0]) & (time <= speech[1])
    high_gamma_change = None
    if high_gamma is not None:
        hg_mask = (frequencies >= 70.0) & (frequencies < 150.0)
        hg_linear = np.mean(power[hg_mask], axis=0)
        high_gamma_change = float(10.0 * np.log10(np.mean(hg_linear[speech_mask]) / np.mean(hg_linear[baseline_mask])))

    np.savez_compressed(
        output_dir / "time_frequency_results.npz",
        time_s=time,
        frequencies_hz=frequencies,
        power_db=power_db,
        high_gamma_db=np.array([]) if high_gamma is None else high_gamma,
    )
    summary = {
        "input_file": input_path.name,
        "array_layout": "time_x_channels",
        "channel_index_0based": args.channel_index,
        "channel_label": channel_label,
        "sampling_rate_hz": args.sfreq,
        "baseline_s": baseline,
        "speech_s": speech,
        "bandpass_hz": bandpass,
        "notch_hz": notches,
        "tfr_range_hz": [args.tfr_min, max_frequency],
        "high_gamma_speech_to_baseline_db": high_gamma_change,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-channel sEEG time-frequency validation")
    parser.add_argument("--input", required=True, help="Trial-level 2-D NPY file")
    parser.add_argument("--sfreq", required=True, type=float, help="Sampling frequency in Hz")
    parser.add_argument("--channel-index", required=True, type=int, help="Zero-based channel index")
    parser.add_argument("--channel-label", default=None)
    parser.add_argument("--time-axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--input-unit", choices=("V", "uV"), default="V")
    parser.add_argument("--baseline", nargs=2, type=float, required=True, metavar=("START", "END"))
    parser.add_argument("--speech", nargs=2, type=float, required=True, metavar=("START", "END"))
    parser.add_argument("--bandpass", nargs=2, type=float, default=(1.0, 200.0))
    parser.add_argument("--notch", default="50,100,150")
    parser.add_argument("--tfr-min", type=float, default=1.0)
    parser.add_argument("--tfr-max", type=float, default=195.0)
    parser.add_argument("--n-frequencies", type=int, default=96)
    parser.add_argument("--psd-max", type=float, default=250.0)
    parser.add_argument("--smoothing-ms", type=float, default=50.0)
    parser.add_argument("--output-dir", default="outputs/time_frequency")
    parser.add_argument("--formats", default="png,pdf,svg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
