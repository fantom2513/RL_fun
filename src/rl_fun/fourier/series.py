"""Fourier series of a closed curve, ordered so the biggest circles come first."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def coefficients(points: ComplexArray) -> tuple[FloatArray, ComplexArray]:
    """Return integer frequencies and complex coefficients sorted by descending |c|.

    ``points`` are samples of one period at t = k / len(points); the curve is
    then ``sum(c * exp(2j * pi * f * t))``.
    """
    count = len(points)
    spectrum = np.fft.fft(points) / count
    freqs = np.fft.fftfreq(count, d=1.0 / count)
    order = np.argsort(-np.abs(spectrum), kind="stable")
    return freqs[order], spectrum[order]


def partial_sum(
    freqs: FloatArray, coeffs: ComplexArray, n: int, t: float | FloatArray
) -> ComplexArray:
    """Curve value at each ``t`` in [0, 1] using only the first ``n`` terms."""
    times = np.atleast_1d(np.asarray(t, dtype=float))
    phases = np.exp(2j * np.pi * np.outer(times, freqs[:n]))
    return phases @ coeffs[:n]


def chain_positions(
    freqs: FloatArray, coeffs: ComplexArray, n: int, t: float
) -> ComplexArray:
    """Centres of the epicycle chain: origin, then the tip of each successive arm."""
    arms = coeffs[:n] * np.exp(2j * np.pi * freqs[:n] * t)
    return np.concatenate(([0j], np.cumsum(arms)))
