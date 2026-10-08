"""Small 2D vector helpers shared by track and sensor code."""

import numpy as np


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the z component of the 2D cross product along the last axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
