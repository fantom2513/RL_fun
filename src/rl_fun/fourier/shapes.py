"""Circles and segments as cubic Bezier control points (Manim's native curve type)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

KAPPA = 0.5522847498307936  # handle length of a quarter-circle Bezier, in radii


def _to_points(z: ComplexArray) -> FloatArray:
    return np.column_stack([z.real, z.imag, np.zeros(len(z))])


def circles_as_beziers(centres: ComplexArray, radii: FloatArray) -> FloatArray:
    """Each circle as four cubic arcs (16 points); circles are stacked in order."""
    if len(centres) == 0:
        return np.zeros((0, 3))
    quarter = np.arange(4)
    start = np.exp(0.5j * np.pi * quarter)[None, :]  # (1, 4)
    end = np.exp(0.5j * np.pi * (quarter + 1))[None, :]
    c = np.asarray(centres)[:, None]
    r = np.asarray(radii)[:, None]
    p0 = c + r * start
    p3 = c + r * end
    p1 = p0 + KAPPA * r * 1j * start
    p2 = p3 - KAPPA * r * 1j * end
    stacked = np.stack([p0, p1, p2, p3], axis=-1)  # (circles, 4 arcs, 4 points)
    return _to_points(stacked.reshape(-1))


def segments_as_beziers(starts: ComplexArray, ends: ComplexArray) -> FloatArray:
    """Each straight segment as one cubic curve (4 points); segments are stacked in order."""
    if len(starts) == 0:
        return np.zeros((0, 3))
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    delta = ends - starts
    stacked = np.stack([starts, starts + delta / 3, starts + 2 * delta / 3, ends], axis=-1)
    return _to_points(stacked.reshape(-1))