# Fourier Epicycles Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a vertical 9:16 mp4 in which a chain of rotating circles draws an SVG/image contour while the number of circles grows by stages.

**Architecture:** A pure-numpy core (`series.py`: FFT coefficients, partial sums, chain positions; `contour.py`: SVG/raster to a normalised closed curve) is tested without rendering. A thin Manim scene (`scene.py`) redraws circles, arms and trail every frame from `t` and the current `N`. A CLI wrapper (`scripts/fourier.py` -> `rl_fun.fourier.cli`) ties it together.

**Tech Stack:** Python 3.12, numpy, Manim Community (SVG parsing + rendering), Pillow + contourpy (raster contours), pytest, uv dependency group `animation`.

Spec: `docs/superpowers/specs/2026-10-16-fourier-epicycles-design.md`.

**Conventions (from the repo):** code lives in `src/rl_fun/<area>/`, scripts in `scripts/` are 3-line wrappers, tests in `tests/<area>/` with **no `__init__.py`**, so test file basenames must be unique across the repo (we use the `test_fourier_` prefix). Conventional Commits; the failing-test (RED) commit is separate from the implementation (GREEN) commit. Run animation tests with `uv run --group animation pytest ...`; modules needing optional deps use `pytest.importorskip` so plain `uv run pytest` still passes.

---

## File structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | new dependency group `animation` |
| `src/rl_fun/fourier/__init__.py` | package marker |
| `src/rl_fun/fourier/series.py` | `coefficients`, `partial_sum`, `chain_positions` (numpy only) |
| `src/rl_fun/fourier/contour.py` | `resample`, `normalize`, `join_subpaths`, `contour_from_image`, `contour_from_svg`, `load_contour`, `available_presets` |
| `src/rl_fun/fourier/assets/{pi,heart,star}.svg` | preset outlines |
| `src/rl_fun/fourier/scene.py` | `DrawingSpec`, `FourierDrawing`, `render` (imports Manim) |
| `src/rl_fun/fourier/cli.py` | `parse_stages`, `build_parser`, `main` |
| `scripts/fourier.py` | entry point wrapper |
| `tests/fourier/test_fourier_series.py` | series tests |
| `tests/fourier/test_fourier_geometry.py` | resample / normalize / join tests |
| `tests/fourier/test_fourier_raster.py` | raster contour tests |
| `tests/fourier/test_fourier_svg.py` | SVG + preset tests |
| `tests/fourier/test_fourier_cli.py` | CLI tests |

---

### Task 1: Dependency group and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/rl_fun/fourier/__init__.py`

- [ ] **Step 1: Add the dependency group (the user approved the `animation` group with `manim`; `contourpy` and `pillow` are part of the approved design)**

Run: `uv add --group animation manim contourpy pillow`
Expected: `pyproject.toml` gains a `[dependency-groups]` entry `animation = ["contourpy>=…", "manim>=…", "pillow>=…"]`; `uv.lock` updates. If installation of `manim` fails on Windows (pycairo/manimpango wheel), stop and report the error to the user instead of working around it.

- [ ] **Step 2: Create the package**

`src/rl_fun/fourier/__init__.py`:

```python
"""Fourier epicycle animation: contour -> Fourier series -> Manim video."""
```

- [ ] **Step 3: Verify the environment imports**

Run: `uv run --group animation python -c "import manim, contourpy, PIL; print(manim.__version__)"`
Expected: prints a version, no traceback.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/rl_fun/fourier/__init__.py
git commit -m "chore: add the animation dependency group and fourier package"
```

---

### Task 2: Fourier series core (`series.py`)

**Files:**
- Create: `tests/fourier/test_fourier_series.py`
- Create: `src/rl_fun/fourier/series.py`

- [ ] **Step 1: Write the failing tests**

`tests/fourier/test_fourier_series.py`:

```python
import numpy as np

from rl_fun.fourier.series import chain_positions, coefficients, partial_sum


def _square(count: int = 256) -> np.ndarray:
    side = count // 4
    ramp = np.linspace(-1.0, 1.0, side, endpoint=False)
    return np.concatenate(
        [
            ramp - 1j,
            1 + 1j * ramp,
            -ramp + 1j,
            -1 - 1j * ramp,
        ]
    )


def _mean_squared_error(points: np.ndarray, n: int) -> float:
    freqs, coeffs = coefficients(points)
    t = np.arange(len(points)) / len(points)
    return float(np.mean(np.abs(partial_sum(freqs, coeffs, n, t) - points) ** 2))


def test_coefficients_are_sorted_by_descending_magnitude():
    freqs, coeffs = coefficients(_square())

    magnitudes = np.abs(coeffs)

    assert np.all(np.diff(magnitudes) <= 1e-12)
    assert len(freqs) == len(coeffs)


def test_full_series_reproduces_the_points():
    points = _square()
    freqs, coeffs = coefficients(points)
    t = np.arange(len(points)) / len(points)

    rebuilt = partial_sum(freqs, coeffs, len(points), t)

    assert np.allclose(rebuilt, points, atol=1e-9)


def test_error_never_grows_with_more_terms():
    points = _square()

    errors = [_mean_squared_error(points, n) for n in (1, 3, 10, 50, 200)]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < errors[0] / 100


def test_pure_circle_needs_one_term():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = 2.0 * np.exp(1j * angles)

    assert _mean_squared_error(points, 1) < 1e-12


def test_chain_ends_at_the_partial_sum():
    points = _square()
    freqs, coeffs = coefficients(points)

    positions = chain_positions(freqs, coeffs, 7, 0.3)

    assert positions[0] == 0
    assert len(positions) == 8
    assert np.isclose(positions[-1], partial_sum(freqs, coeffs, 7, 0.3)[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fourier/test_fourier_series.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'rl_fun.fourier.series'`.

- [ ] **Step 3: Commit the failing tests (RED checkpoint)**

```bash
git add tests/fourier/test_fourier_series.py
git commit -m "test: specify the Fourier series of a closed curve"
```

- [ ] **Step 4: Write the implementation**

`src/rl_fun/fourier/series.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/fourier/test_fourier_series.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/fourier/series.py
git commit -m "feat: add the Fourier series, its partial sums and the epicycle chain"
```

---

### Task 3: Curve geometry (`resample`, `normalize`, `join_subpaths`)

**Files:**
- Create: `tests/fourier/test_fourier_geometry.py`
- Create: `src/rl_fun/fourier/contour.py`

- [ ] **Step 1: Write the failing tests**

`tests/fourier/test_fourier_geometry.py`:

```python
import numpy as np
import pytest

from rl_fun.fourier.contour import join_subpaths, normalize, resample


def _unit_square_corners() -> np.ndarray:
    return np.array([0 + 0j, 4 + 0j, 4 + 4j, 0 + 4j])


def test_resample_returns_requested_count_with_equal_steps():
    points = resample(_unit_square_corners(), 64)

    closed = np.append(points, points[0])
    steps = np.abs(np.diff(closed))

    assert len(points) == 64
    assert np.allclose(steps, 16 / 64, atol=1e-9)


def test_resample_starts_at_the_first_point():
    points = resample(_unit_square_corners(), 32)

    assert points[0] == 0


def test_normalize_centres_and_fits_the_longer_side():
    points = np.array([10 + 0j, 30 + 0j, 30 + 10j, 10 + 10j])

    result = normalize(points)

    assert np.isclose(result.real.min(), -1) and np.isclose(result.real.max(), 1)
    assert np.isclose(result.imag.min(), -0.5) and np.isclose(result.imag.max(), 0.5)


def test_join_subpaths_visits_the_nearest_path_next():
    near = np.array([10 + 0j, 11 + 0j, 11 + 1j])
    far = np.array([100 + 0j, 101 + 0j, 101 + 1j])
    first = np.array([0 + 0j, 1 + 0j, 1 + 1j])

    joined = join_subpaths([first, far, near])

    assert len(joined) == 9
    assert np.array_equal(joined[3:6], near)
    assert np.array_equal(joined[6:], far)


def test_join_subpaths_rejects_an_empty_list():
    with pytest.raises(ValueError):
        join_subpaths([])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fourier/test_fourier_geometry.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'rl_fun.fourier.contour'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/fourier/test_fourier_geometry.py
git commit -m "test: specify resampling, normalising and joining of contours"
```

- [ ] **Step 4: Write the implementation**

`src/rl_fun/fourier/contour.py`:

```python
"""Turn SVG files, images and presets into a closed, normalised complex curve."""

from __future__ import annotations

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/fourier/test_fourier_geometry.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/fourier/contour.py
git commit -m "feat: add contour resampling, normalising and sub-path joining"
```

---

### Task 4: Raster contour

**Files:**
- Create: `tests/fourier/test_fourier_raster.py`
- Modify: `src/rl_fun/fourier/contour.py` (append)

- [ ] **Step 1: Write the failing tests**

`tests/fourier/test_fourier_raster.py`:

```python
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("contourpy")
Image = pytest.importorskip("PIL.Image")

from rl_fun.fourier.contour import contour_from_image  # noqa: E402


def _save(path: Path, gray: np.ndarray) -> Path:
    Image.fromarray(gray.astype(np.uint8)).save(path)
    return path


def test_dark_disc_on_white_gives_a_round_contour(tmp_path):
    yy, xx = np.mgrid[0:100, 0:100]
    gray = np.where((xx - 50) ** 2 + (yy - 50) ** 2 < 30**2, 0, 255)
    path = _save(tmp_path / "disc.png", gray)

    points = contour_from_image(path, 200)

    assert points.shape == (200,)
    assert np.allclose(np.abs(points), 1.0, atol=0.06)


def test_longest_contour_wins_over_small_specks(tmp_path):
    gray = np.full((100, 100), 255)
    gray[40:60, 20:80] = 0  # 60 x 20 bar
    gray[2:6, 2:6] = 0  # 4 x 4 speck
    path = _save(tmp_path / "bar.png", gray)

    points = contour_from_image(path, 200)

    assert np.isclose(points.real.max(), 1.0, atol=0.02)
    assert np.isclose(points.imag.max(), 1 / 3, atol=0.05)


def test_image_is_not_upside_down(tmp_path):
    gray = np.full((100, 100), 255)
    gray[10:30, 30:70] = 0  # wide blob in the upper part of the picture
    gray[30:90, 48:52] = 0  # thin tail going down
    path = _save(tmp_path / "t.png", gray)

    points = contour_from_image(path, 400)

    # after normalising, imag spans [-1, 1] and the 40 px wide blob reaches |real| = 0.5
    assert np.abs(points.real[points.imag > 0.5]).max() > 0.3  # wide blob on top
    assert np.abs(points.real[points.imag < -0.5]).max() < 0.1  # thin tail at the bottom


def test_invert_flips_which_side_is_the_shape(tmp_path):
    gray = np.full((100, 100), 0)
    gray[40:60, 20:80] = 255  # white bar on black
    path = _save(tmp_path / "inv.png", gray)

    points = contour_from_image(path, 200, invert=True)

    assert np.isclose(points.imag.max(), 1 / 3, atol=0.05)


def test_blank_image_is_rejected(tmp_path):
    path = _save(tmp_path / "blank.png", np.full((50, 50), 255))

    with pytest.raises(ValueError):
        contour_from_image(path, 100)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group animation pytest tests/fourier/test_fourier_raster.py -v`
Expected: FAIL with `ImportError: cannot import name 'contour_from_image'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/fourier/test_fourier_raster.py
git commit -m "test: specify contour extraction from raster images"
```

- [ ] **Step 4: Write the implementation (append to `contour.py`)**

Add these imports at the top of `contour.py` (below the existing ones):

```python
from pathlib import Path
```

Append:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --group animation pytest tests/fourier/test_fourier_raster.py -v`
Expected: 5 passed. If `test_longest_contour_wins_over_small_specks` is off by pixel rounding, widen `atol` to 0.06 but do not change the shapes.

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/fourier/contour.py
git commit -m "feat: extract the outline of the largest shape in a raster image"
```

---

### Task 5: SVG contour, presets and `load_contour`

**Files:**
- Create: `src/rl_fun/fourier/assets/pi.svg`, `heart.svg`, `star.svg`
- Create: `tests/fourier/test_fourier_svg.py`
- Modify: `src/rl_fun/fourier/contour.py` (append)

SVG coordinates are y-down, as in any SVG file; Manim flips them to y-up when parsing.

- [ ] **Step 1: Create the preset assets**

`src/rl_fun/fourier/assets/pi.svg`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-12 -13 24 26">
  <path d="M -11 -12 L 11 -12 L 11 -9 L 7 -9 L 7 8 L 10 8 L 10 11 L 4 11 L 4 -9 L -3 -9 L -3 11 L -9 11 L -9 8 L -6 8 L -6 -9 L -11 -9 Z"/>
</svg>
```

`src/rl_fun/fourier/assets/heart.svg`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-10 -10 20 21">
  <path d="M 0 -6 C -2 -9 -9 -8 -9 -2 C -9 3 -3 6 0 10 C 3 6 9 3 9 -2 C 9 -8 2 -9 0 -6 Z"/>
</svg>
```

`src/rl_fun/fourier/assets/star.svg`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-11 -11 22 22">
  <path d="M 0 -10 L 2.351 -3.236 L 9.511 -3.090 L 3.804 1.236 L 5.878 8.090 L 0 4 L -5.878 8.090 L -3.804 1.236 L -9.511 -3.090 L -2.351 -3.236 Z"/>
</svg>
```

- [ ] **Step 2: Write the failing tests**

`tests/fourier/test_fourier_svg.py`:

```python
import numpy as np
import pytest

pytest.importorskip("manim")

from rl_fun.fourier.contour import (  # noqa: E402
    available_presets,
    contour_from_svg,
    load_contour,
)

SQUARE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <path d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/>
</svg>"""

TWO_SQUARES = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 10">
  <path d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/>
  <path d="M 20 0 L 30 0 L 30 10 L 20 10 Z"/>
</svg>"""


def test_square_svg_lies_on_the_normalised_square(tmp_path):
    path = tmp_path / "square.svg"
    path.write_text(SQUARE)

    points = contour_from_svg(path, 400)

    assert points.shape == (400,)
    assert np.allclose(np.maximum(np.abs(points.real), np.abs(points.imag)), 1.0, atol=0.02)


def test_svg_with_two_parts_becomes_one_curve_through_both(tmp_path):
    path = tmp_path / "two.svg"
    path.write_text(TWO_SQUARES)

    points = contour_from_svg(path, 800)

    assert np.isclose(points.real.min(), -1) and np.isclose(points.real.max(), 1)
    left = np.any(points.real < -0.9)
    right = np.any(points.real > 0.9)
    assert left and right


def test_svg_is_upright(tmp_path):
    path = tmp_path / "tri.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M 5 0 L 10 10 L 0 10 Z"/></svg>'
    )

    points = contour_from_svg(path, 600)

    apex = points[np.argmax(points.imag)]
    assert abs(apex.real) < 0.1  # the apex (svg y=0) must be at the top


def test_presets_are_listed_and_load():
    names = available_presets()

    assert {"pi", "heart", "star"} <= set(names)
    for name in names:
        points = load_contour(name, 300)
        assert points.shape == (300,)
        assert np.isclose(max(points.real.max(), points.imag.max()), 1.0, atol=0.02)


def test_load_contour_dispatches_on_extension(tmp_path):
    svg = tmp_path / "square.svg"
    svg.write_text(SQUARE)

    assert load_contour(svg, 100).shape == (100,)


def test_unknown_source_is_a_clear_error():
    with pytest.raises(ValueError, match="no-such-shape"):
        load_contour("no-such-shape", 100)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --group animation pytest tests/fourier/test_fourier_svg.py -v`
Expected: FAIL with `ImportError: cannot import name 'available_presets'`.

- [ ] **Step 4: Commit assets and failing tests**

```bash
git add src/rl_fun/fourier/assets tests/fourier/test_fourier_svg.py
git commit -m "test: specify SVG contours and the bundled presets"
```

- [ ] **Step 5: Write the implementation (append to `contour.py`)**

```python
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
    return normalize(resample(join_subpaths(paths), count))


def load_contour(source: str | Path, count: int) -> ComplexArray:
    """Preset name, ``.svg`` path or raster image path -> normalised closed curve."""
    text = str(source)
    preset = ASSETS_DIR / f"{text}.svg"
    if preset.exists():
        return contour_from_svg(preset, count)
    path = Path(text)
    if path.suffix.lower() == ".svg" and path.exists():
        return contour_from_svg(path, count)
    if path.suffix.lower() in RASTER_SUFFIXES and path.exists():
        return contour_from_image(path, count)
    raise ValueError(
        f"unknown source {text!r}: use a preset ({', '.join(available_presets())}) "
        "or a path to an existing .svg/.png/.jpg file"
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --group animation pytest tests/fourier/test_fourier_svg.py -v`
Expected: 6 passed. Notes if something fails: (a) if `get_subpaths` returns nothing for a plain-line path, print `len(part.get_points())` to confirm Manim produced points and adapt `_sample_beziers` input; (b) if `test_svg_is_upright` fails, Manim's y flip is not what is assumed: flip the sign of `imag` in `contour_from_svg` and keep the test as the source of truth; (c) the π preset is drawn so it must read as π when upright, so check it visually in Task 7.

- [ ] **Step 7: Commit**

```bash
git add src/rl_fun/fourier/contour.py
git commit -m "feat: load contours from SVG files and bundled presets"
```

---

### Task 6: CLI parsing and orchestration

**Files:**
- Create: `tests/fourier/test_fourier_cli.py`
- Create: `src/rl_fun/fourier/cli.py`
- Create: `scripts/fourier.py`

- [ ] **Step 1: Write the failing tests**

`tests/fourier/test_fourier_cli.py`:

```python
import argparse

import pytest

from rl_fun.fourier import cli


def test_parse_stages_reads_a_comma_list():
    assert cli.parse_stages("3,10,50") == (3, 10, 50)


@pytest.mark.parametrize("bad", ["", "a,b", "0,5", "5,3", "3,3", "-1,4"])
def test_parse_stages_rejects_bad_input(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_stages(bad)


def test_defaults():
    args = cli.build_parser().parse_args(["--source", "pi"])

    assert args.stages == (3, 10, 50, 200)
    assert args.quality == "high"
    assert args.points == 2048


def test_stages_larger_than_points_exit_with_usage_error():
    with pytest.raises(SystemExit) as error:
        cli.main(["--source", "pi", "--stages", "3,500", "--points", "256"])

    assert error.value.code == 2


def test_main_builds_the_spec_and_hands_it_to_the_renderer(monkeypatch, tmp_path):
    pytest.importorskip("manim")
    captured = {}

    def fake_render(spec, out, quality):
        captured.update(spec=spec, out=out, quality=quality)
        return out

    monkeypatch.setattr(cli, "_render", fake_render)
    out = tmp_path / "pi.mp4"

    code = cli.main(["--source", "pi", "--stages", "3,10", "--quality", "low", "--out", str(out)])

    assert code == 0
    assert captured["out"] == out
    assert captured["quality"] == "low"
    assert captured["spec"].stages == (3, 10)
    assert len(captured["spec"].coeffs) == 2048
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group animation pytest tests/fourier/test_fourier_cli.py -v`
Expected: collection error `ModuleNotFoundError`/`ImportError` for `rl_fun.fourier.cli`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/fourier/test_fourier_cli.py
git commit -m "test: specify the Fourier animation command line"
```

- [ ] **Step 4: Write the CLI**

The CLI imports `DrawingSpec` from `scene.py` lazily (Manim is optional), so `DrawingSpec` must not import Manim at module level. It is defined in its own tiny module for that reason: create `src/rl_fun/fourier/spec.py` first.

`src/rl_fun/fourier/spec.py`:

```python
"""Plain data describing one animation; kept free of Manim imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DrawingSpec:
    freqs: NDArray[np.float64]
    coeffs: NDArray[np.complex128]
    outline: NDArray[np.complex128]
    stages: tuple[int, ...]
    seconds_per_stage: float = 5.0
    scale: float = 3.6  # frame units for a normalised half-extent of 1
    centre: complex = 1.5j  # moves the drawing up to leave room for the counter
```

`src/rl_fun/fourier/cli.py`:

```python
"""Command line: ``python scripts/fourier.py --source pi --stages 3,10,50,200``."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_fun.fourier.contour import available_presets, load_contour
from rl_fun.fourier.series import coefficients
from rl_fun.fourier.spec import DrawingSpec


def parse_stages(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"stages must be integers separated by commas: {text!r}")
    if any(v < 1 for v in values) or list(values) != sorted(set(values)):
        raise argparse.ArgumentTypeError("stages must be positive and strictly increasing")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw a contour with Fourier epicycles.")
    parser.add_argument(
        "--source",
        required=True,
        help=f"preset ({', '.join(available_presets())}) or path to an .svg/.png/.jpg file",
    )
    parser.add_argument("--stages", type=parse_stages, default=(3, 10, 50, 200))
    parser.add_argument("--points", type=int, default=2048, help="samples of the contour")
    parser.add_argument("--seconds-per-stage", type=float, default=5.0)
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--out", type=Path, default=None, help="default: runs/fourier/<name>.mp4")
    return parser


def _render(spec: DrawingSpec, out: Path, quality: str) -> Path:
    from rl_fun.fourier.scene import render

    return render(spec, out, quality)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if max(args.stages) > args.points:
        parser.error(f"largest stage {max(args.stages)} exceeds --points {args.points}")
    try:
        outline = load_contour(args.source, args.points)
    except ValueError as error:
        parser.error(str(error))
    freqs, coeffs = coefficients(outline)
    spec = DrawingSpec(
        freqs=freqs,
        coeffs=coeffs,
        outline=outline,
        stages=args.stages,
        seconds_per_stage=args.seconds_per_stage,
    )
    out = args.out or Path("runs") / "fourier" / f"{Path(args.source).stem}.mp4"
    print(f"wrote {_render(spec, out, args.quality)}")
    return 0
```

`scripts/fourier.py`:

```python
from rl_fun.fourier.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --group animation pytest tests/fourier/test_fourier_cli.py -v`
Expected: 10 passed (1 valid list, 6 invalid-input cases, defaults, usage error, orchestration).

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/fourier/spec.py src/rl_fun/fourier/cli.py scripts/fourier.py
git commit -m "feat: add the Fourier animation command line"
```

---

### Task 7: Manim scene and render function

**Files:**
- Create: `src/rl_fun/fourier/scene.py`

This module is not covered by unit tests (spec: rendering is checked manually). Manim API names below are from Manim Community 0.18+; if one differs in the installed version, fix it against `uv run --group animation python -c "import manim; help(manim.VMobject.add_points_as_corners)"` rather than guessing.

- [ ] **Step 1: Write the scene**

`src/rl_fun/fourier/scene.py`:

```python
"""Manim scene: a chain of epicycles draws a contour while N grows by stages."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from manim import UP, WHITE, Scene, Text, ValueTracker, VMobject, linear, tempconfig

from rl_fun.fourier.series import chain_positions, partial_sum
from rl_fun.fourier.spec import DrawingSpec

CIRCLE_SEGMENTS = 32
MIN_RADIUS = 0.01  # frame units; smaller circles are sub-pixel
TRAIL_STEPS = 800
QUALITY = {"low": (540, 960, 30), "high": (1080, 1920, 60)}
FRAME_WIDTH, FRAME_HEIGHT = 9.0, 16.0


def _to_points(z: np.ndarray) -> np.ndarray:
    return np.column_stack([z.real, z.imag, np.zeros(len(z))])


class FourierDrawing(Scene):
    def __init__(self, spec: DrawingSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self._n = spec.stages[0]
        self._show_chain = True

    def _place(self, z: np.ndarray) -> np.ndarray:
        return z * self.spec.scale + self.spec.centre

    def construct(self):
        spec = self.spec
        t = ValueTracker(0.0)

        outline = VMobject().set_stroke(WHITE, 2, opacity=0.12).set_fill(opacity=0)
        closed = np.append(spec.outline, spec.outline[0])
        outline.set_points_as_corners(_to_points(self._place(closed)))

        circles = VMobject().set_stroke(WHITE, 1, opacity=0.35).set_fill(opacity=0)
        arms = VMobject().set_stroke("#FFD166", 2).set_fill(opacity=0)
        trail = VMobject().set_stroke("#FF4D6D", 4).set_fill(opacity=0)
        angles = np.linspace(0.0, 2 * np.pi, CIRCLE_SEGMENTS + 1)
        ring = np.exp(1j * angles)

        def positions() -> np.ndarray:
            chain = chain_positions(spec.freqs, spec.coeffs, self._n, t.get_value())
            return self._place(chain)

        def update_circles(mob: VMobject) -> None:
            mob.clear_points()
            if not self._show_chain:
                return
            centres = positions()
            for k in range(self._n):
                radius = abs(spec.coeffs[k]) * spec.scale
                if spec.freqs[k] == 0 or radius < MIN_RADIUS:
                    continue  # the zero frequency is a fixed offset, not a circle
                pts = _to_points(centres[k] + radius * ring)
                mob.start_new_path(pts[0])
                mob.add_points_as_corners(pts[1:])

        def update_arms(mob: VMobject) -> None:
            mob.clear_points()
            if self._show_chain:
                mob.set_points_as_corners(_to_points(positions()))

        def update_trail(mob: VMobject) -> None:
            tv = t.get_value()
            times = np.linspace(0.0, tv, max(2, int(TRAIL_STEPS * tv)))
            curve = partial_sum(spec.freqs, spec.coeffs, self._n, times)
            mob.set_points_as_corners(_to_points(self._place(curve)))

        circles.add_updater(update_circles)
        arms.add_updater(update_arms)
        trail.add_updater(update_trail)
        self.add(outline, circles, arms, trail)

        label: Text | None = None
        for n in spec.stages:
            self._n = n
            new_label = Text(f"N = {n}", font_size=56).to_edge(UP, buff=1.2)
            if label is not None:
                self.remove(label)
            label = new_label
            self.add(label)
            t.set_value(0.0)
            self.wait(0.3)
            self.play(t.animate.set_value(1.0), run_time=spec.seconds_per_stage, rate_func=linear)

        self.wait(1.0)
        self._show_chain = False
        self.wait(2.0)


def render(spec: DrawingSpec, out_path: Path, quality: str = "high") -> Path:
    """Render ``spec`` to ``out_path`` (mp4) and return the final path."""
    width, height, fps = QUALITY[quality]
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "pixel_width": width,
        "pixel_height": height,
        "frame_rate": fps,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "media_dir": str(out_path.parent / "_manim"),
        "output_file": out_path.stem,
        "format": "mp4",
        "disable_caching": True,
        "verbosity": "WARNING",
    }
    with tempconfig(settings):
        scene = FourierDrawing(spec)
        scene.render()
        produced = Path(scene.renderer.file_writer.movie_file_path)
    shutil.copyfile(produced, out_path)
    return out_path
```

- [ ] **Step 2: Import check**

Run: `uv run --group animation python -c "import rl_fun.fourier.scene"`
Expected: no output, no traceback. Fix any wrong Manim import name before continuing.

- [ ] **Step 3: Smoke render at low quality (π, two short stages)**

Run: `uv run --group animation python scripts/fourier.py --source pi --stages 3,20 --seconds-per-stage 2 --quality low --out runs/fourier/smoke-pi.mp4`
Expected: prints `wrote …\runs\fourier\smoke-pi.mp4`; the file exists and is a non-zero 540x960 video. If `movie_file_path` is not an attribute of the installed Manim, look inside `runs/fourier/_manim/videos/` for the produced mp4 and adapt `render()` accordingly. `runs/` is gitignored.

- [ ] **Step 4: Look at the result**

Open the mp4 (or extract a frame) and confirm: π is upright and reads as π, circles and arms visible, trail grows, the counter reads `N = 3` then `N = 20`, nothing is cut off at the 9:16 edges. Fix scale/centre in `DrawingSpec` if the drawing is clipped.

- [ ] **Step 5: Commit**

```bash
git add src/rl_fun/fourier/scene.py
git commit -m "feat: add the Manim scene and renderer for the epicycle animation"
```

---

### Task 8: Full-quality render, lint, docs

**Files:**
- Modify: `README.md` (add a short section)

- [ ] **Step 1: Full test suite and lint**

Run: `uv run --group animation pytest -q` then `uv run ruff check src tests scripts`
Expected: all tests pass (new ones included); ruff reports no errors. Fix lint in the new files only.

- [ ] **Step 2: Plain-environment check**

Run: `uv run pytest tests/fourier -q`
Expected: series, geometry and CLI-parsing tests pass; raster/SVG tests are *skipped* (not errors) when the `animation` group is absent. If `uv run` already installs the group, this step is a no-op: just confirm there are no errors.

- [ ] **Step 3: Add the README section**

Append to `README.md`:

````markdown
## Fourier epicycle animation

A video of circles drawing a contour (π, a heart, your own SVG or a silhouette image).
It is unrelated to the RL code and uses the optional `animation` dependency group:

```bash
uv run --group animation python scripts/fourier.py --source pi --quality low
uv run --group animation python scripts/fourier.py --source my-shape.svg --stages 3,10,50,300
```

Presets: `pi`, `heart`, `star` (see `src/rl_fun/fourier/assets/`). Raster images should
be dark shapes on a light background. Output goes to `runs/fourier/<name>.mp4`
(1080x1920, 60 fps with `--quality high`).
````

- [ ] **Step 4: Final render of the headline clip**

Run: `uv run --group animation python scripts/fourier.py --source pi --quality high`
Expected: `runs/fourier/pi.mp4` is written (may take several minutes; run in the background if needed). Report the duration and rendering time to the user.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: describe the Fourier epicycle animation"
```
