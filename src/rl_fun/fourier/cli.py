"""Command line: ``python scripts/fourier.py --source pi --stages 3,10,50,200``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fun.fourier.contour import available_presets, load_contour
from rl_fun.fourier.series import coefficients
from rl_fun.fourier.spec import DrawingSpec

MISSING_DEPENDENCIES = (
    "this source needs the optional animation dependencies: "
    "run with `uv run --group animation ...`"
)


def parse_stages(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"stages must be integers separated by commas: {text!r}"
        ) from None
    if any(v < 1 for v in values) or list(values) != sorted(set(values)):
        raise argparse.ArgumentTypeError("stages must be positive and strictly increasing")
    return values


def unit_interval(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"must be strictly between 0 and 1: {text!r}")
    return value


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
    parser.add_argument(
        "--threshold",
        type=unit_interval,
        default=0.5,
        help="raster sources: gray level (0-1, exclusive) separating shape from background",
    )
    parser.add_argument(
        "--invert", action="store_true", help="raster sources: the shape is light on dark"
    )
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--out", type=Path, default=None, help="default: runs/fourier/<name>.mp4")
    return parser


def _render(spec: DrawingSpec, out: Path, quality: str) -> Path:
    from rl_fun.fourier.scene import render

    return render(spec, out, quality)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if max(args.stages) > args.points:
        parser.error(f"largest stage {max(args.stages)} exceeds --points {args.points}")
    try:
        outline = load_contour(
            args.source, args.points, threshold=args.threshold, invert=args.invert
        )
    except ValueError as error:
        parser.error(str(error))
    except ImportError:
        parser.error(MISSING_DEPENDENCIES)
    freqs, coeffs = coefficients(outline)
    spec = DrawingSpec(
        freqs=freqs,
        coeffs=coeffs,
        outline=outline,
        stages=args.stages,
        seconds_per_stage=args.seconds_per_stage,
    )
    out = args.out or Path("runs") / "fourier" / f"{Path(args.source).stem}.mp4"
    try:
        written = _render(spec, out, args.quality)
    except ImportError:
        parser.error(MISSING_DEPENDENCIES)
    print(f"wrote {written}")
    return 0
