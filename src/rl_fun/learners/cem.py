"""Cross-entropy method: sample networks around a mean, keep the best, move the mean to them."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rl_fun.lab.config import RunConfig
from rl_fun.learners.base import IterationResult
from rl_fun.learners.schemas import CEM_SCHEMA, copy_schema
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import FleetDrawable, run_generation

UPDATABLE = ("elite", "mutation_scale", "fitness")
NOISE_FRACTION = 0.1
"""The extra noise added to the spread is `mutation_scale` times this: 0.3 gives 0.03."""


def cem_update(
    samples: np.ndarray, scores: np.ndarray, elite: int, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """New mean and spread of the weights: those of the `elite` best samples (the spread plus
    `noise` so the search never freezes). `samples` has one row of weights per sample."""
    samples = np.asarray(samples, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not 1 <= elite < len(samples):
        raise ValueError(f"elite должна быть от 1 до {len(samples) - 1}, получено {elite}")
    best = samples[np.argsort(scores)[::-1][:elite]]
    return best.mean(axis=0), best.std(axis=0) + noise


class CEMLearner:
    """One iteration is one generation of samples drawn from N(mean, spread²) per weight.

    The distribution is updated at the start of the next iteration, after the worker has had the
    chance to apply parameter updates. Sample 0 is the mean itself, so the current best guess is
    always driven and measured.
    """

    @staticmethod
    def schema() -> dict[str, Any]:
        """Parameter description of the cross-entropy learner."""
        return copy_schema(CEM_SCHEMA)

    def setup(
        self,
        config: RunConfig,
        fleet: RacingFleet,
        rng: np.random.Generator,
        view: FleetDrawable | None,
    ) -> None:
        self.config = config
        self._fleet = fleet
        self._rng = rng
        self._view = view
        self._sizes = config.model.layer_sizes
        self._activation = config.model.activation
        self._mean = reference.init_population(rng, 1, self._sizes, config.init_scale)[0]
        self._std = np.full(self._mean.shape, float(config.init_scale))
        self._samples: np.ndarray | None = None
        self._scores: np.ndarray | None = None
        self._best_weights = self._mean.copy()
        self._best_fitness = -np.inf
        self._iteration = -1

    def spread(self) -> float:
        """The average spread (standard deviation) of the weights now."""
        return float(np.mean(self._std))

    def run_iteration(self, iteration: int) -> IterationResult:
        config = self.config
        if self._scores is not None:
            self._mean, self._std = cem_update(
                self._samples,  # type: ignore[arg-type]
                self._scores,
                config.elite,
                config.mutation_scale * NOISE_FRACTION,
            )
        noise = self._rng.standard_normal((config.population, self._mean.size))
        samples = self._mean + self._std * noise
        samples[0] = self._mean
        sizes, activation = self._sizes, self._activation
        result = run_generation(
            self._fleet,
            samples,
            lambda weights, observation: reference.forward(weights, observation, sizes, activation),
            view=self._view,
            inspect=lambda weights, observation: reference.inspect(
                weights, observation, sizes, activation
            ),
            label=f"Лаборатория, итерация {iteration + 1}",
        )
        scores = config.fitness.score(result)
        self._samples, self._scores = samples, scores
        top = int(np.argmax(scores))
        if scores[top] > self._best_fitness:
            self._best_fitness = float(scores[top])
            self._best_weights = samples[top].copy()
        self._iteration = iteration

        progress = np.asarray(result.progress, dtype=np.float64)
        finished = np.asarray(result.finished, dtype=bool)
        lap_steps = np.asarray(result.lap_steps, dtype=np.float64)
        finished_laps = lap_steps[finished & np.isfinite(lap_steps)]
        return IterationResult(
            best=float(np.max(progress)),
            mean=float(np.mean(progress)),
            finished=int(np.count_nonzero(finished)),
            best_fitness=float(np.max(scores)),
            best_lap_steps=int(finished_laps.min()) if finished_laps.size > 0 else None,
            params={"elite": config.elite, "mutation_scale": config.mutation_scale},
            extra={},
        )

    def apply_update(self, params: dict[str, Any]) -> list[str]:
        notices: list[str] = []
        for key, value in params.items():
            if key not in UPDATABLE:
                notices.append(f"параметр {key} нельзя менять во время запуска")
                continue
            try:
                self.config = replace(self.config, **{key: value})
            except ValueError as error:
                notices.append(f"параметр {key} не применён: {error}")
        return notices

    def best_snapshot(self) -> dict[str, Any] | None:
        if self._iteration < 0:
            return None
        return {
            "sizes": self._sizes,
            "activation": self._activation,
            "generation": self._iteration,
            "fitness": self._best_fitness,
            "weights": self._best_weights.tolist(),
        }
