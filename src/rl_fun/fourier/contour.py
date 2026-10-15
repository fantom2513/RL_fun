"""Turn SVG files, images and presets into a closed, normalised complex curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def resample(points: ComplexArray, count: int) -> ComplexArray:
    """Closed polyline -> ``count`` points spaced equally by arc length."""
    if len(points) < 2:
        raise ValueError("need at least 2 points")
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
    if half == 0:
        raise ValueError("shape has zero extent")
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

    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    gray = np.asarray(image.convert("L"), dtype=float) / 255.0
    if invert:
        gray = 1.0 - gray
    padded = np.pad(gray, 1, constant_values=1.0)  # background border closes every loop
    lines = contourpy.contour_generator(z=padded).lines(threshold)
    candidates = [line[:, 0] - 1j * line[:, 1] for line in lines if len(line) > 2]
    if not candidates:
        raise ValueError(f"no contour found in {path}; try another threshold or invert=True")
    longest = max(candidates, key=_path_length)
    return normalize(resample(longest, count))


ASSETS_DIR = Path(__file__).parent / "assets"
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def available_presets() -> list[str]:
    return sorted(p.stem for p in ASSETS_DIR.glob("*.svg"))


def _sample_beziers(points: NDArray[np.float64], per_curve: int) -> ComplexArray:
    """Cubic Bezier control points (4k x 3) -> dense polyline as complex numbers."""
    curves = points.reshape(-1, 4, 3)
    t = np.linspace(0.0, 1.0, per_curve, endpoint=False)[:, None, None]
    p0, p1, p2, p3 = (curves[None, :, i, :] for i in range(4))
    dense = (
        (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    )  # (per_curve, k, 3)
    ordered = dense.transpose(1, 0, 2).reshape(-1, 3)
    ordered = np.vstack([ordered, curves[-1, 3][None, :]])
    return ordered[:, 0] + 1j * ordered[:, 1]


def contour_from_svg(path: str | Path, count: int, per_curve: int = 24) -> ComplexArray:
    """Outline of every path in an SVG file as one closed curve (Manim parses the SVG)."""
    from manim import SVGMobject

    drawing = SVGMobject(str(path))
    paths = [
        _sample_beziers(sub, per_curve)
        for part in drawing.family_members_with_points()
        for sub in part.get_subpaths()
    ]
    if not paths:
        raise ValueError(f"{path}: no drawable paths found")
    return normalize(resample(join_subpaths(paths), count))


def load_contour(
    source: str | Path,
    count: int,
    threshold: float = 0.5,
    invert: bool = False,
) -> ComplexArray:
    """Preset name, ``.svg`` path or raster image path -> normalised closed curve."""
    text = str(source)
    if text in available_presets():
        return contour_from_svg(ASSETS_DIR / f"{text}.svg", count)
    path = Path(text)
    if path.suffix.lower() == ".svg" and path.exists():
        return contour_from_svg(path, count)
    if path.suffix.lower() in RASTER_SUFFIXES and path.exists():
        return contour_from_image(path, count, threshold=threshold, invert=invert)
    raise ValueError(
        f"unknown source {text!r}: use a preset ({', '.join(available_presets())}) "
        "or a path to an existing .svg/.png/.jpg file"
    )
