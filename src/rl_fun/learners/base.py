"""Learner abstraction: the protocol every training algorithm of the lab implements.

The worker owns the run (fleet, view, artifacts, messages) and only loops over iterations; what an
iteration is, how it learns and which parameters it has is up to the learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from rl_fun.lab.config import RunConfig
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import FleetDrawable


@dataclass(frozen=True)
class IterationResult:
    """Summary of one iteration, the fields of a `gen` message.

    `best`, `mean` and `finished` describe progress in laps; `best_fitness` is the learner's own
    quality number; `params` echoes the live parameters the iteration ran with; `extra` holds
    numeric metrics specific to the learner (empty for evolution).
    """

    best: float
    mean: float
    finished: int
    best_fitness: float
    best_lap_steps: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)


class Learner(Protocol):
    """A training algorithm driven by the worker, one iteration at a time."""

    def setup(
        self,
        config: RunConfig,
        fleet: RacingFleet,
        rng: np.random.Generator,
        view: FleetDrawable | None,
    ) -> None:
        """Prepare for a run; `view` receives a frame after every simulation step (or is None)."""

    def run_iteration(self, iteration: int) -> IterationResult:
        """Run one iteration (simulation included), drawing frames through the view."""

    def apply_update(self, params: dict[str, Any]) -> list[str]:
        """Apply live parameter changes; returns notices (Russian) for rejected ones."""

    def best_snapshot(self) -> dict[str, Any] | None:
        """Best network so far in the `best_weights.json` format, or None before any iteration."""

    def schema(self) -> dict[str, Any]:
        """Description of the learner's parameters for the catalog."""
