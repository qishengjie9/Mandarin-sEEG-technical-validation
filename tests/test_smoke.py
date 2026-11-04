from __future__ import annotations

import numpy as np

from mandarin_seeg_validation.acoustic_contamination import (
    AnalysisConfig,
    diagonal_dominance_test,
    spectrogram_psd,
    subtract_moving_average,
)
from mandarin_seeg_validation.time_frequency import baseline_db, morlet_power, preprocess


def test_time_frequency_shapes() -> None:
    sfreq = 500.0
    time = np.arange(0, 4, 1 / sfreq)
    rng = np.random.default_rng(3)
    x = np.sin(2 * np.pi * 90 * time) + 0.05 * rng.standard_normal(time.size)
    clean = preprocess(x, sfreq, (1.0, 200.0), (50.0, 100.0, 150.0))
    frequencies = np.array([10.0, 40.0, 90.0, 150.0])
    power = morlet_power(clean, frequencies, sfreq)
    normalized = baseline_db(power, time, (0.2, 1.0))
    assert power.shape == (len(frequencies), len(time))
    assert normalized.shape == power.shape
    assert np.isfinite(normalized).all()


def test_acoustic_spectrogram_and_permutation() -> None:
    sfreq = 2500.0
    time = np.arange(0, 1.5, 1 / sfreq)
    x = np.sin(2 * np.pi * 200 * time).astype(np.float32)
    centered = subtract_moving_average(x, sfreq)
    config = AnalysisConfig(
        max_trials=1,
        frequency_min_hz=75.0,
        frequency_max_hz=1000.0,
        permutations=100,
    )
    frequencies, power = spectrogram_psd(centered, sfreq, config)
    assert frequencies[0] >= 75.0
    assert frequencies[-1] <= 1000.0
    assert power.shape[0] == len(frequencies)

    matrix = np.eye(12) * 0.5
    observed, p_value, surrogate = diagonal_dominance_test(
        matrix, 100, np.random.default_rng(4)
    )
    assert observed == 0.5
    assert 0.0 <= p_value <= 1.0
    assert surrogate.shape == (100,)
