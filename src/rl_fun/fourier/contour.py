"""Turn SVG files, images and presets into a closed, normalised complex curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def resample(points: ComplexArray, count: int) -> ComplexArray:
    """Closed polyline -> ``count`` points spaced equally by arc length."""
    closed = np.append(points, points[0])
    cumulative = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(closed)))))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    real = np.interp(targets, cumulative, closed.real)
    imag = np.interp(targets, cumulative, closed.imag)
    return real + 1j * imag


def normalize(points: ComplexArray, fit: float = 1.0) -> ComplexArray:
    """Centre the bounding box on the origin and scale its longer half-side to ``fit``."""
    centre = complex(
        (points.real.min() + points.real.max()) / 2,
        (points.imag.min() + points.imag.max()) / 2,
    )
    shifted = points - centre
    half = max(np.abs(shifted.real).max(), np.abs(shifted.imag).max())
    return shifted * (fit / half)


def join_subpaths(paths: list[ComplexArray]) -> ComplexArray:
    """Chain separate closed sub-paths into one traversal, nearest start first."""
    if not paths:
        raise ValueError("no sub-paths to join")
    remaining = list(paths[1:])
    ordered = [paths[0]]
    end = paths[0][-1]
    while remaining:
        index = min(range(len(remaining)), key=lambda i: abs(remaining[i][0] - end))
        following = remaining.pop(index)
        ordered.append(following)
        end = following[-1]
    return np.concatenate(ordered)

def _path_length(points: ComplexArray) -> float:
    return float(np.abs(np.diff(points)).sum())


def contour_from_image(
    path: str | Path,
    count: int,
    threshold: float = 0.5,
    invert: bool = False,
) -> ComplexArray:
    """Outline of the largest dark shape on a light background.

    ``invert=True`` treats light shapes on a dark background instead. The image
    y axis points down, so it is flipped to get an upright curve.
    """
    import contourpy
    from PIL import Image

    gray = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    if invert:
        gray = 1.0 - gray
    padded = np.pad(gray, 1, constant_values=1.0)  # background border closes every loop
    lines = contourpy.contour_generator(z=padded).lines(threshold)
    candidates = [line[:, 0] - 1j * line[:, 1] for line in lines if len(line) > 2]
    if not candidates:
        raise ValueError(f"no contour found in {path}; try another threshold or invert=True")
    longest = max(candidates, key=_path_length)
    return normalize(resample(longest, count))
