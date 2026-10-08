"""Evolution learner: the neuroevolution loop of the lab (elitism plus Gaussian mutation)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rl_fun.lab.config import RunConfig
from rl_fun.learners.base import IterationResult
from rl_fun.learners.schemas import EVOLUTION_SCHEMA, copy_schema
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import FleetDrawable, run_generation

UPDATABLE = ("mutation_rate", "mutation_scale", "elite", "fitness")


class EvolutionLearner:
    """One iteration is one generation: drive every car, score, remember the best.

    The next generation is bred at the start of the following iteration, after the worker has had
    the chance to apply parameter updates.
    """

    @staticmethod
    def schema() -> dict[str, Any]:
        """Parameter description of the evolution learner."""
        return copy_schema(EVOLUTION_SCHEMA)

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
        self._population = reference.init_population(
            rng, config.population, self._sizes, config.init_scale
        )
        self._scores: np.ndarray | None = None
        self._best_weights = self._population[0].copy()
        self._best_fitness = -np.inf
        self._iteration = -1

    def run_iteration(self, iteration: int) -> IterationResult:
        if self._scores is not None:
            self._breed()
        sizes, activation = self._sizes, self._activation
        result = run_generation(
            self._fleet,
            self._population,
            lambda weights, observation: reference.forward(weights, observation, sizes, activation),
            view=self._view,
            inspect=lambda weights, observation: reference.inspect(
                weights, observation, sizes, activation
            ),
            label=f"Лаборатория, поколение {iteration + 1}",
        )
        scores = self.config.fitness.score(result)
        self._scores = scores

        top = int(np.argmax(scores))
        if scores[top] > self._best_fitness:
            self._best_fitness = float(scores[top])
            self._best_weights = self._population[top].copy()
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
            params={
                "mutation_rate": self.config.mutation_rate,
                "mutation_scale": self.config.mutation_scale,
                "elite": self.config.elite,
            },
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

    def _breed(self) -> None:
        config = self.config
        self._population = reference.next_generation(
            self._population,
            self._scores,  # type: ignore[arg-type]
            self._rng,
            reference.EvolutionParams(
                population=config.population,
                elite=config.elite,
                mutation_rate=config.mutation_rate,
                mutation_scale=config.mutation_scale,
                init_scale=config.init_scale,
            ),
        )
