# Fourier epicycles animation — design

Date: 2026-10-16. Status: approved, not yet implemented. Plan:
`docs/superpowers/plans/2026-10-16-fourier-epicycles.md`.

## Goal

A short vertical video (Reels/Instagram style) in which a chain of rotating
circles (epicycles, a discrete Fourier series) draws an arbitrary closed
contour: the letter π, a heart, a portrait silhouette. Unrelated to the RL
code; it only shares the repository and `uv` environment.

## Decisions

- Engine: Manim Community, rendered to mp4 (not an interactive HTML page).
- Output: 1080x1920 (9:16), 60 fps. A low-quality preview preset is available.
- Scene: "chain + accuracy stages". The chain draws one period with few
  circles, then the number of circles N grows by stages (default 3, 10, 50,
  200); each stage redraws the curve faster and the picture sharpens. A
  counter shows `N = …`. The final stage shows the full chain and a bright
  trail. No formulas in v1, so LaTeX is not required.
- Sources: SVG files, raster images (PNG/JPG), and bundled SVG presets.
  Hand drawing is out of scope.
- Dependencies: new uv dependency group `animation` with `manim` and
  `contourpy`. It stays out of the RL dependencies. No system ffmpeg (Manim
  CE uses PyAV).

## Layout

New package `src/rl_fun/fourier/`, following the repository layout (scripts are
thin wrappers, tests in `tests/fourier/`). Existing RL code is untouched. Manim
is imported only by `scene.py`, so the rest works and tests without it.

| File | Responsibility |
| --- | --- |
| `contour.py` | Turn a source into a closed curve of M complex points, uniformly spaced by arc length. |
| `series.py` | FFT of the point sequence into coefficients sorted by descending `abs(c)`; partial sum for N terms; chain positions. Pure numpy. |
| `spec.py` | `DrawingSpec` dataclass (no Manim import). |
| `scene.py` | Manim scene `FourierDrawing`: circles, arms, trail, stage schedule, counter; `render()`. |
| `cli.py` | Argument parsing and orchestration. |
| `assets/` | Preset SVGs (π, heart, star). |
| `scripts/fourier.py` | Wrapper: `--source pi\|path.svg\|path.png --stages 3,10,50,200 --quality low\|high --out …`. |

### contour.py

- SVG: parse with Manim's `SVGMobject`, sample the Bézier points of all
  sub-paths. Several sub-paths (e.g. π is several shapes) are joined into one
  traversal by straight bridge segments; bridges are traversed once.
- Raster: Pillow converts to grayscale and thresholds; `contourpy` extracts
  contours; the longest one is used. Intended for silhouettes and
  high-contrast images; the threshold is a parameter.
- Output is normalised (centred, scaled to fit the frame) so the scene does
  not depend on the source size.

### series.py

- `coefficients(points)` returns frequencies and complex coefficients, with
  frequency order by descending magnitude so the largest circles come first.
- `partial_sum(coeffs, n, t)` evaluates the curve using the first n terms.
- Positive and negative frequencies are interleaved by magnitude, so N counts
  circles, not frequency pairs.

## Testing

TDD per project rules (tests first, committed separately). Tests live in
`tests/fourier/` and do not render video.

- Reconstruction error decreases monotonically (within tolerance) as N grows
  and is ~0 for N equal to the number of points.
- Coefficients are sorted by descending magnitude.
- Arc-length resampling gives equal step lengths.
- A multi-sub-path SVG yields one closed curve passing through all parts.
- Raster extraction on a synthetic image (filled disc/square) returns a
  contour near the expected shape.
- Rendering is checked manually at low quality, not in the test suite.

## Out of scope

Hand drawing, interactive browser version, formula overlays, audio, multiple
disconnected figures animated separately, automatic portrait cleanup (the
user supplies a clean silhouette or SVG).

## Open risks

- Manim on Windows needs wheels for `pycairo`/`manimpango`; if installation
  fails, fall back to documenting the missing system library.
- Raster contours of noisy photos will be jagged; mitigated by the threshold
  parameter and optional smoothing before the FFT.

## Revision 2 (2026-10-16): x/y chains, error readout, smooth curves

Requested after the first render. Plan: `docs/superpowers/plans/2026-10-16-fourier-xy-chains.md`.

- The single complex epicycle chain is replaced by two real Fourier chains: x(t) as a
  horizontal chain on top, y(t) as a vertical chain on the left. Thin guide lines from
  the chain ends cross at the point that draws the red trail. N counts circles per chain;
  the constant term is only the chain's starting position.
- The label shows `N = …` and, under it, `error …`: the RMS distance between the n-term
  curve and the contour, as a percentage of the shape's half-size (one value per stage).
- Circles are exact (four cubic Bezier arcs each) and the trail is sampled in proportion to
  the highest frequency in play, so lines are smooth at any N.
- The complex series functions (`coefficients`, `partial_sum`, `chain_positions`) were removed.
