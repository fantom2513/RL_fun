"""Bookkeeping of crash positions for the fleet view (pure, no pygame)."""

from __future__ import annotations

import numpy as np


class CrashMarkers:
    """Remembers where cars died; forgets everything when a new episode starts.

    A car counts as crashed when it was alive in the previous frame and is not now, unless it
    finished. A new episode is recognised by `steps` returning to zero (or going backwards), a car
    coming back to life, or a different fleet size.
    """

    def __init__(self, limit: int = 2000) -> None:
        self.points: list[tuple[float, float]] = []
        self._limit = limit
        self._alive: np.ndarray | None = None
        self._steps: int | None = None

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alive: np.ndarray,
        steps: int | None = None,
        finished: np.ndarray | None = None,
    ) -> None:
        """Record cars that stopped being alive since the previous call."""
        alive = np.asarray(alive, dtype=bool)
        previous = self._alive
        went_back = steps is not None and (
            steps == 0 or (self._steps is not None and steps < self._steps)
        )
        restarted = (
            previous is None
            or previous.shape != alive.shape
            or bool((alive & ~previous).any())
            or went_back
        )
        if restarted:
            self.points = []
            previous = np.ones_like(alive)
        crashed = previous & ~alive
        if finished is not None:
            crashed &= ~np.asarray(finished, dtype=bool)
        for index in np.flatnonzero(crashed):
            if len(self.points) < self._limit:
                self.points.append((float(x[index]), float(y[index])))
        self._alive = alive.copy()
        self._steps = steps
