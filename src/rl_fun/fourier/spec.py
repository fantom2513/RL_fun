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
