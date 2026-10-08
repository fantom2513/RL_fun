"""Map a network's raw outputs to car controls: steering, throttle and boost."""

from __future__ import annotations

import numpy as np


def map_controls(
    outputs: tuple[str, ...], raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn raw network outputs (N, len(outputs)) into (steer, throttle, boost), each (N,).

    Raw values are clipped to [-1, 1]. `steer` and `throttle` pass through as they are.
    `accelerate`, `brake` and `boost` are rescaled from [-1, 1] to [0, 1]. The final throttle is
    throttle + accelerate - brake, clipped to [-1, 1]. A missing channel reads as zero.
    """
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(outputs):
        raise ValueError(
            f"raw must have shape (N, {len(outputs)}) for outputs {outputs}, "
            f"got {values.shape}"
        )
    values = np.clip(values, -1.0, 1.0)
    columns = {name: values[:, index] for index, name in enumerate(outputs)}
    zeros = np.zeros(values.shape[0])

    def unit(name: str) -> np.ndarray:
        return (columns[name] + 1.0) / 2.0 if name in columns else zeros

    steer = columns.get("steer", zeros)
    throttle = columns.get("throttle", zeros) + unit("accelerate") - unit("brake")
    boost = unit("boost")
    return steer, np.clip(throttle, -1.0, 1.0), boost
