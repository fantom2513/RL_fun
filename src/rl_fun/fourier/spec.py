"""Plain data describing one animation; kept free of Manim imports."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rl_fun.fourier.series import RealSeries


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