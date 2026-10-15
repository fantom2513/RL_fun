"""Fourier series of a closed curve, ordered so the biggest circles come first."""

from __future__ import annotations

from dataclasses import dataclass

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

@dataclass(frozen=True)
class RealSeries:
    """Cosine series of one coordinate: offset + sum(A_k * cos(2 pi f_k t + phi_k)).

    Terms are sorted by descending amplitude; the constant term is ``offset``
    and is never counted as a term.
    """

    offset: float
    freqs: FloatArray
    amplitudes: FloatArray
    phases: FloatArray

    def angles(self, n: int, t: float) -> FloatArray:
        return 2 * np.pi * self.freqs[:n] * t + self.phases[:n]

    def value(self, n: int, t: float | FloatArray) -> FloatArray:
        """Sum of the first ``n`` terms at each ``t`` in [0, 1]."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        angles = 2 * np.pi * np.outer(times, self.freqs[:n]) + self.phases[:n]
        return self.offset + np.cos(angles) @ self.amplitudes[:n]

    def chain(self, n: int, t: float) -> FloatArray:
        """Running sum: the offset, then the value after each of the first ``n`` terms."""
        steps = self.amplitudes[:n] * np.cos(self.angles(n, t))
        return self.offset + np.concatenate(([0.0], np.cumsum(steps)))


def real_series(values: FloatArray) -> RealSeries:
    """Real Fourier series of one period sampled at t = k / len(values)."""
    values = np.asarray(values, dtype=float)
    count = len(values)
    spectrum = np.fft.rfft(values)
    amplitudes = 2 * np.abs(spectrum[1:]) / count
    if count % 2 == 0:
        amplitudes[-1] /= 2  # the Nyquist term has no mirror image
    order = np.argsort(-amplitudes, kind="stable")
    return RealSeries(
        offset=float(spectrum[0].real / count),
        freqs=np.arange(1, len(spectrum), dtype=float)[order],
        amplitudes=amplitudes[order],
        phases=np.angle(spectrum[1:])[order],
    )


def approximation_error(x: RealSeries, y: RealSeries, n: int, points: ComplexArray) -> float:
    """RMS distance between the n-term curve and ``points``, relative to the shape's half-size."""
    t = np.arange(len(points)) / len(points)
    approx = x.value(n, t) + 1j * y.value(n, t)
    half = max(np.abs(points.real).max(), np.abs(points.imag).max())
    return float(np.sqrt(np.mean(np.abs(approx - points) ** 2)) / half)
