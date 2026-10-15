# Fourier x/y chains, error readout and smooth curves — Implementation Plan (revision 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the single complex epicycle chain by two real Fourier chains (x on top, y on the left) whose guide lines cross at the drawing point, show the approximation error under the `N` counter, and draw circles and trail smoothly at any N.

**Architecture:** `series.py` gets a `RealSeries` (real cosine series of one coordinate, sorted by amplitude) and `approximation_error`. A new numpy-only `shapes.py` converts circles and segments into cubic Bezier point arrays (exact circles, no polygons). `scene.py` rebuilds circles, hands, projections, guide lines and the trail every frame from these. The old complex-series functions are removed.

**Tech Stack:** unchanged (numpy, Manim CE 0.22, pytest, uv group `animation`).

Builds on `docs/superpowers/plans/2026-10-16-fourier-epicycles.md` (done). Conventions are unchanged: tests in `tests/fourier/` with unique `test_fourier_*` basenames, no `__init__.py`, RED commit separate from GREEN, Conventional Commits, commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

**Geometry of the two chains** (frame 9 x 16, origin at centre, y up). For a series with `offset`, amplitudes `A_k`, angles `theta_k = 2*pi*f_k*t + phase_k`, and a shape scale `s`:
- x-chain sits on the row `y = R`. Circle k is centred at `(X_k, R)` with radius `s*A_k`, where `X_0 = cx + s*offset` and `X_{k+1} = X_k + s*A_k*cos(theta_k)` (`cx` = x of the shape centre). Its hand goes from `(X_k, R)` to `(X_{k+1}, R + s*A_k*sin(theta_k))`; a thin projection line drops from the hand tip to `(X_{k+1}, R)`. After n terms `X_n` is the x of the drawing point.
- y-chain sits on the column `x = C`. Circle k is centred at `(C, Y_k)`, `Y_0 = cy + s*offset_y`, `Y_{k+1} = Y_k + s*A_k*cos(theta_k)`. Its hand goes from `(C, Y_k)` to `(C + s*A_k*sin(theta_k), Y_{k+1})`; the projection line runs from the hand tip to `(C, Y_{k+1})`.
- The drawing point is `P = (X_n, Y_n)`. Guide lines run from `(X_n, R)` to `P` and from `(C, Y_n)` to `P`. The red trail is `x(t'), y(t')` for t' in [0, t].
- N counts circles per chain; the constant term `offset` is only the chain's starting position, never a circle.

---

### Task 1: Real series, approximation error and error label

**Files:**
- Modify: `src/rl_fun/fourier/series.py` (append), `src/rl_fun/fourier/spec.py` (append `format_error`)
- Test: `tests/fourier/test_fourier_realseries.py` (create), `tests/fourier/test_fourier_spec.py` (create)

- [ ] **Step 1: Write the failing tests**

`tests/fourier/test_fourier_realseries.py`:

```python
import numpy as np
import pytest

from rl_fun.fourier.series import RealSeries, approximation_error, real_series


def _signal(count: int) -> np.ndarray:
    t = np.arange(count) / count
    return 0.5 + np.cos(2 * np.pi * 3 * t + 0.4) - 0.7 * np.sin(2 * np.pi * 8 * t) + 0.1 * np.cos(
        2 * np.pi * 21 * t
    )


@pytest.mark.parametrize("count", [64, 65, 100])
def test_all_terms_reproduce_the_samples(count):
    values = np.random.default_rng(0).normal(size=count)
    series = real_series(values)
    t = np.arange(count) / count

    rebuilt = series.value(len(series.freqs), t)

    assert np.allclose(rebuilt, values, atol=1e-9)


def test_terms_are_sorted_by_descending_amplitude():
    series = real_series(_signal(256))

    assert np.all(np.diff(series.amplitudes) <= 1e-12)


def test_single_cosine_is_recovered():
    count = 128
    t = np.arange(count) / count
    series = real_series(2.0 + 3.0 * np.cos(2 * np.pi * 5 * t + 0.7))

    assert np.isclose(series.offset, 2.0)
    assert series.freqs[0] == 5
    assert np.isclose(series.amplitudes[0], 3.0)
    assert np.isclose(series.phases[0], 0.7)
    assert np.allclose(series.value(1, t), 2.0 + 3.0 * np.cos(2 * np.pi * 5 * t + 0.7))


def test_chain_starts_at_the_offset_and_ends_at_the_value():
    series = real_series(_signal(256))

    chain = series.chain(6, 0.37)

    assert len(chain) == 7
    assert np.isclose(chain[0], series.offset)
    assert np.isclose(chain[-1], series.value(6, 0.37)[0])


def test_error_never_grows_with_more_terms_and_vanishes_at_the_end():
    side = 64
    ramp = np.linspace(-1.0, 1.0, side, endpoint=False)
    square = np.concatenate([ramp - 1j, 1 + 1j * ramp, -ramp + 1j, -1 - 1j * ramp])
    x, y = real_series(square.real), real_series(square.imag)

    errors = [approximation_error(x, y, n, square) for n in (1, 3, 10, 50, len(x.freqs))]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:], strict=False))
    assert errors[0] > 0.1
    assert errors[-1] < 1e-9


def test_circle_needs_one_term_per_coordinate():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.exp(1j * angles)

    error = approximation_error(real_series(circle.real), real_series(circle.imag), 1, circle)

    assert error < 1e-9


def test_real_series_is_a_frozen_dataclass():
    series = real_series(_signal(64))

    assert isinstance(series, RealSeries)
    with pytest.raises(AttributeError):
        series.offset = 1.0  # type: ignore[misc]
```

`tests/fourier/test_fourier_spec.py`:

```python
import pytest

from rl_fun.fourier.spec import format_error


@pytest.mark.parametrize(
    ("fraction", "text"),
    [(0.42, "42 %"), (0.0421, "4.2 %"), (0.0042, "0.42 %"), (0.00001, "<0.01 %"), (0.0, "<0.01 %")],
)
def test_format_error(fraction, text):
    assert format_error(fraction) == text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --group animation pytest tests/fourier/test_fourier_realseries.py tests/fourier/test_fourier_spec.py -v`
Expected: collection errors (`cannot import name 'RealSeries'` / `'format_error'`).

- [ ] **Step 3: Commit RED**

```bash
git add tests/fourier/test_fourier_realseries.py tests/fourier/test_fourier_spec.py
git commit -m "test: specify real Fourier series per coordinate, the error and its label"
```

- [ ] **Step 4: Implement**

Append to `src/rl_fun/fourier/series.py` (add `from dataclasses import dataclass` to the imports):

```python
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
```

Append to `src/rl_fun/fourier/spec.py`:

```python
def format_error(fraction: float) -> str:
    """Error as a short percentage: '42 %', '4.2 %', '0.42 %', '<0.01 %'."""
    percent = fraction * 100
    if percent >= 10:
        return f"{percent:.0f} %"
    if percent >= 1:
        return f"{percent:.1f} %"
    if percent >= 0.01:
        return f"{percent:.2f} %"
    return "<0.01 %"
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run --group animation pytest tests/fourier -q` then `uv run ruff check src tests scripts`
Expected: all pass (old complex-series tests still pass at this point), ruff clean.

- [ ] **Step 6: Commit GREEN**

```bash
git add src/rl_fun/fourier/series.py src/rl_fun/fourier/spec.py
git commit -m "feat: add real Fourier series per coordinate, the approximation error and its label"
```

---

### Task 2: Smooth shapes as Bezier point arrays

**Files:**
- Create: `src/rl_fun/fourier/shapes.py`, `tests/fourier/test_fourier_shapes.py`

- [ ] **Step 1: Write the failing tests**

`tests/fourier/test_fourier_shapes.py`:

```python
import numpy as np

from rl_fun.fourier.shapes import circles_as_beziers, segments_as_beziers


def _bezier_midpoints(points: np.ndarray) -> np.ndarray:
    curves = points.reshape(-1, 4, 3)
    return (curves[:, 0] + 3 * curves[:, 1] + 3 * curves[:, 2] + curves[:, 3]) / 8


def test_circle_has_sixteen_points_and_lies_on_the_circle():
    points = circles_as_beziers(np.array([2 + 1j]), np.array([3.0]))

    assert points.shape == (16, 3)
    flat = points[:, 0] + 1j * points[:, 1]
    anchors = flat.reshape(4, 4)[:, [0, 3]]
    assert np.allclose(np.abs(anchors - (2 + 1j)), 3.0)
    mid = _bezier_midpoints(points)
    assert np.allclose(np.hypot(mid[:, 0] - 2, mid[:, 1] - 1), 3.0, rtol=1e-3)
    assert np.all(points[:, 2] == 0)


def test_many_circles_are_stacked():
    points = circles_as_beziers(np.array([0j, 5 + 0j]), np.array([1.0, 2.0]))

    assert points.shape == (32, 3)


def test_no_circles_gives_an_empty_array():
    points = circles_as_beziers(np.array([], dtype=complex), np.array([]))

    assert points.shape == (0, 3)


def test_segments_are_straight_cubic_curves():
    points = segments_as_beziers(np.array([0j, 1 + 1j]), np.array([3 + 0j, 1 + 4j]))

    assert points.shape == (8, 3)
    first = points[:4, 0] + 1j * points[:4, 1]
    assert np.allclose(first, [0, 1, 2, 3])
    last = points[4:, 0] + 1j * points[4:, 1]
    assert np.allclose(last, [1 + 1j, 1 + 2j, 1 + 3j, 1 + 4j])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/fourier/test_fourier_shapes.py -v`
Expected: `ModuleNotFoundError: No module named 'rl_fun.fourier.shapes'`.

- [ ] **Step 3: Commit RED**

```bash
git add tests/fourier/test_fourier_shapes.py
git commit -m "test: specify exact circles and straight segments as Bezier points"
```

- [ ] **Step 4: Implement**

`src/rl_fun/fourier/shapes.py`:

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/fourier/test_fourier_shapes.py -v` then `uv run ruff check src tests scripts`
Expected: 4 passed; ruff clean.

- [ ] **Step 6: Commit GREEN**

```bash
git add src/rl_fun/fourier/shapes.py
git commit -m "feat: add exact circle and segment Bezier point builders"
```

---

### Task 3: Switch spec, CLI and scene to the x/y chains; drop the complex series

**Files:**
- Modify: `src/rl_fun/fourier/spec.py`, `src/rl_fun/fourier/cli.py`, `src/rl_fun/fourier/scene.py`, `src/rl_fun/fourier/series.py` (delete `coefficients`, `partial_sum`, `chain_positions`), `tests/fourier/test_fourier_cli.py`, `tests/fourier/test_fourier_series.py` (delete)

These existing tests are changed *because the API they specify is being replaced*, not to make failing code pass; say so in the commit message and the report.

- [ ] **Step 1: Update the CLI test first (RED)**

In `tests/fourier/test_fourier_cli.py`, replace the last assertion of `test_main_builds_the_spec_and_hands_it_to_the_renderer`:

```python
    assert len(captured["spec"].coeffs) == 2048
```
with
```python
    assert len(captured["spec"].x.freqs) == 1024  # 2048 samples -> 1024 cosine terms
    assert len(captured["spec"].y.freqs) == 1024
    assert len(captured["spec"].outline) == 2048
```
and change `test_stages_larger_than_points_exit_with_usage_error` to use a stage that fits `points` but not the term count:

```python
def test_stages_larger_than_the_term_count_exit_with_usage_error():
    with pytest.raises(SystemExit) as error:
        cli.main(["--source", "pi", "--stages", "3,200", "--points", "256"])

    assert error.value.code == 2
```
(256 points give 128 terms per coordinate, so 200 is too many.) Run `uv run --group animation pytest tests/fourier/test_fourier_cli.py -q`; expect these two tests to FAIL. Commit: `test: follow the x/y series in the command line tests`.

- [ ] **Step 2: Implement spec and CLI**

`src/rl_fun/fourier/spec.py` — replace the `DrawingSpec` dataclass (keep `format_error`) with:

```python
@dataclass(frozen=True)
class Layout:
    """Frame is 9 x 16 units, origin at the centre, y up. Starting values; tune by eye."""

    shape_scale: float = 2.2
    shape_centre: complex = 1.4 - 1.2j
    x_row_y: float = 4.4
    y_col_x: float = -2.9


@dataclass(frozen=True)
class DrawingSpec:
    x: RealSeries
    y: RealSeries
    outline: NDArray[np.complex128]
    stages: tuple[int, ...]
    seconds_per_stage: float = 5.0
    layout: Layout = field(default_factory=Layout)
```
with imports `from dataclasses import dataclass, field` and `from rl_fun.fourier.series import RealSeries`.

`cli.py`: import `real_series` instead of `coefficients`; build `x = real_series(outline.real)`, `y = real_series(outline.imag)`; the guard becomes `if max(args.stages) > args.points // 2: parser.error(f"largest stage {max(args.stages)} exceeds the {args.points // 2} terms that --points {args.points} provides")`; construct `DrawingSpec(x=x, y=y, outline=outline, stages=..., seconds_per_stage=...)`.

- [ ] **Step 3: Rewrite the scene**

Replace the body of `src/rl_fun/fourier/scene.py` above `render()` (keep `render`, `QUALITY`, `FRAME_*` as they are, keep the `frozen_frame=False` waits) with:

```python
"""Manim scene: two real Fourier chains (x on top, y on the left) draw a contour."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from manim import UP, WHITE, Scene, Text, ValueTracker, VMobject, linear, tempconfig

from rl_fun.fourier.series import approximation_error
from rl_fun.fourier.shapes import circles_as_beziers, segments_as_beziers
from rl_fun.fourier.spec import DrawingSpec, format_error

MIN_RADIUS = 0.01  # frame units; smaller circles are sub-pixel
TRAIL_MIN_STEPS = 1500
TRAIL_STEPS_PER_FREQ = 32  # trail samples per unit of the highest frequency in play
QUALITY = {"low": (540, 960, 30), "high": (1080, 1920, 60)}
FRAME_WIDTH, FRAME_HEIGHT = 9.0, 16.0


def _to_points(z: np.ndarray) -> np.ndarray:
    return np.column_stack([z.real, z.imag, np.zeros(len(z))])


def _set(mob: VMobject, points: np.ndarray) -> None:
    mob.clear_points()
    if len(points):
        mob.set_points(points)


class FourierDrawing(Scene):
    def __init__(self, spec: DrawingSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self._n = min(spec.stages[0], len(spec.x.freqs))
        self._trail_steps = TRAIL_MIN_STEPS
        self._show_chain = True

    def construct(self):
        spec, lay = self.spec, self.spec.layout
        s, row, col = lay.shape_scale, lay.x_row_y, lay.y_col_x
        cx, cy = lay.shape_centre.real, lay.shape_centre.imag
        t = ValueTracker(0.0)

        def place(z: np.ndarray) -> np.ndarray:
            return z * s + lay.shape_centre

        outline = VMobject().set_stroke(WHITE, 2, opacity=0.12).set_fill(opacity=0)
        outline.set_points_as_corners(_to_points(place(np.append(spec.outline, spec.outline[0]))))

        circles = VMobject().set_stroke(WHITE, 1.2, opacity=0.4).set_fill(opacity=0)
        hands = VMobject().set_stroke("#FFD166", 2.5).set_fill(opacity=0)
        drops = VMobject().set_stroke(WHITE, 1, opacity=0.25).set_fill(opacity=0)
        guides = VMobject().set_stroke(WHITE, 1.5, opacity=0.5).set_fill(opacity=0)
        trail = VMobject().set_stroke("#FF4D6D", 4).set_fill(opacity=0)

        def geometry():
            """Centres, radii, hand tips and projection feet of both chains at the current t."""
            n, tv = self._n, t.get_value()
            ax, ay = spec.x.amplitudes[:n] * s, spec.y.amplitudes[:n] * s
            th_x, th_y = spec.x.angles(n, tv), spec.y.angles(n, tv)
            xs = cx + s * spec.x.chain(n, tv)  # X_0 .. X_n
            ys = cy + s * spec.y.chain(n, tv)  # Y_0 .. Y_n
            x_centres = xs[:-1] + 1j * row
            x_tips = xs[1:] + 1j * (row + ax * np.sin(th_x))
            x_feet = xs[1:] + 1j * row
            y_centres = col + 1j * ys[:-1]
            y_tips = (col + ay * np.sin(th_y)) + 1j * ys[1:]
            y_feet = col + 1j * ys[1:]
            return (x_centres, ax, x_tips, x_feet), (y_centres, ay, y_tips, y_feet), xs[-1], ys[-1]

        def update_chains(_mob: VMobject) -> None:
            if not self._show_chain:
                for mob in (circles, hands, drops, guides):
                    mob.clear_points()
                return
            (xc, xr, xt, xf), (yc, yr, yt, yf), px, py = geometry()
            keep_x, keep_y = xr >= MIN_RADIUS, yr >= MIN_RADIUS
            _set(
                circles,
                np.vstack(
                    [
                        circles_as_beziers(xc[keep_x], xr[keep_x]),
                        circles_as_beziers(yc[keep_y], yr[keep_y]),
                    ]
                ),
            )
            _set(hands, segments_as_beziers(np.concatenate([xc, yc]), np.concatenate([xt, yt])))
            _set(drops, segments_as_beziers(np.concatenate([xt, yt]), np.concatenate([xf, yf])))
            point = px + 1j * py
            _set(
                guides,
                segments_as_beziers(
                    np.array([px + 1j * row, col + 1j * py]), np.array([point, point])
                ),
            )

        def update_trail(mob: VMobject) -> None:
            tv = t.get_value()
            times = np.linspace(0.0, tv, max(2, int(self._trail_steps * tv)))
            curve = spec.x.value(self._n, times) + 1j * spec.y.value(self._n, times)
            mob.set_points_as_corners(_to_points(place(curve)))

        circles.add_updater(update_chains)  # one updater drives all four chain mobjects
        trail.add_updater(update_trail)
        self.add(outline, circles, hands, drops, guides, trail)

        label: Text | None = None
        error_label: Text | None = None
        for stage in spec.stages:
            n = min(stage, len(spec.x.freqs))
            self._n = n
            top = max(float(np.max(spec.x.freqs[:n])), float(np.max(spec.y.freqs[:n])))
            self._trail_steps = max(TRAIL_MIN_STEPS, int(TRAIL_STEPS_PER_FREQ * top))
            error = approximation_error(spec.x, spec.y, n, spec.outline)
            new_label = Text(f"N = {n}", font_size=56).move_to([0, 7.2, 0])
            new_error = Text(f"error {format_error(error)}", font_size=36).move_to([0, 6.4, 0])
            for old in (label, error_label):
                if old is not None:
                    self.remove(old)
            label, error_label = new_label, new_error
            self.add(label, error_label)
            t.set_value(0.0)
            self.wait(0.3, frozen_frame=False)
            self.play(t.animate.set_value(1.0), run_time=spec.seconds_per_stage, rate_func=linear)

        self.wait(1.0, frozen_frame=False)
        self._show_chain = False
        self.wait(2.0, frozen_frame=False)
```
Remove the now unused imports (`UP` if unused). Note `circles.add_updater(update_chains)` is deliberate: Manim runs the updater of each mobject on the scene; one function updating all four keeps the geometry computed once per frame. If Manim 0.22 does not call updaters of mobjects whose own points are empty, attach a no-op-free alternative: add the updater to a dummy `VMobject()` that is also added to the scene.

- [ ] **Step 4: Delete the complex series**

Remove `coefficients`, `partial_sum`, `chain_positions` (and the `ComplexArray` alias only if now unused — `approximation_error` still uses it) from `series.py`, and `git rm tests/fourier/test_fourier_series.py` (its five tests specify only the removed functions; the real-series tests in `test_fourier_realseries.py` cover the replacement). Check `grep -rn "coefficients\|partial_sum\|chain_positions" src tests scripts` finds nothing.

- [ ] **Step 5: Verify**

Run: `uv run --group animation pytest tests/fourier -q` (all pass), `uv run ruff check src tests scripts` (clean), `uv run --group animation python -c "import rl_fun.fourier.scene"` (imports).

- [ ] **Step 6: Commit**

```bash
git add -A src tests scripts
git commit -m "refactor: draw with two real x/y chains and drop the complex series

The complex series tests specified functions that no longer exist; the
real-series tests replace them. CLI tests follow the new spec fields."
```
(Do not `git add` `docs/reference/` or `runs/`; stage by path.)

---

### Task 4: Smoke render, layout tuning, smoothness check

**Files:**
- Modify (as needed): `src/rl_fun/fourier/spec.py` (`Layout` defaults), `src/rl_fun/fourier/scene.py`

This task is visual; the verification is looking at frames.

- [ ] **Step 1: Low-quality renders**

Run: `uv run --group animation python scripts/fourier.py --source pi --stages 3,10,50 --seconds-per-stage 3 --quality low --out runs/fourier/smoke-pi.mp4`
and the same for `heart` with `--stages 5,40`.

- [ ] **Step 2: Extract and read frames**

With PyAV (installed by Manim) decode frames at about 20 %, 60 % and 95 % of each stage into the scratchpad directory and Read them. Check and fix until all hold:
- the x-chain row (top) and y-chain column (left) are visible and not clipped by the 9:16 edges; the biggest circles may overlap the drawing but must not leave the frame — if they do, tune `Layout` (`shape_scale`, `shape_centre`, `x_row_y`, `y_col_x`);
- the two thin guide lines cross exactly at the red tip of the trail, and the trail grows from that tip;
- `N = …` and `error … %` are readable, not overlapped by circles beyond a faint overlap, and the error decreases across stages;
- circles are round (zoom into one) and the trail has no visible corners at N = 3 (zoom on a curve) or at N = 50;
- N circles per chain are visible at N = 3 (count them).
Report what was wrong and what was changed.

- [ ] **Step 3: Checks**

Run: `uv run --group animation pytest tests/fourier -q`, `uv run ruff check src tests scripts`.

- [ ] **Step 4: Commit**

```bash
git add src/rl_fun/fourier
git commit -m "fix: tune the two-chain layout from rendered frames"
```
(skip if nothing changed.)

---

### Task 5: Docs and final render

- [ ] **Step 1: Update `README.md`** section "Fourier epicycle animation": replace the first sentence with "Two chains of rotating circles (x on top, y on the left) draw a contour (π, a heart, your own SVG or a silhouette image); the error of the current approximation is shown under the counter." and document `--threshold` / `--invert` if the README does not already.
- [ ] **Step 2: Update the spec status** in `docs/superpowers/specs/2026-10-16-fourier-epicycles-design.md`: add the section "Revision 2" summarising the x/y chains, error readout and smooth rendering (as in this plan's Goal and Geometry).
- [ ] **Step 3: Full checks and final render**

Run: `uv run --group animation pytest -q`, `uv run ruff check src tests scripts`, then `uv run --group animation python scripts/fourier.py --source pi --quality high`. Report the output path, size and render time.
- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers
git commit -m "docs: describe the x/y chains and the error readout"
```
