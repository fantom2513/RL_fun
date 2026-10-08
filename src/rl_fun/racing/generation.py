"""Run one generation: every car in a population drives the fleet with its own controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rl_fun.racing.fleet import RacingFleet

Controller = Callable[[np.ndarray, np.ndarray], np.ndarray]
Inspector = Callable[[np.ndarray, np.ndarray], Any]


class FleetDrawable(Protocol):
    """Anything that can draw a fleet frame, for example `rl_fun.racing.fleet_view.FleetView`."""

    def draw(self, fleet: RacingFleet, lines: list[str], network: Any) -> Any:
        """Draw one frame with HUD lines and optional network state."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Per-car outcome of one generation, arrays of shape (n_cars,)."""

    progress: np.ndarray
    finished: np.ndarray
    steps_alive: np.ndarray
    lap_steps: np.ndarray
    mean_speed: np.ndarray | None = None
    centering: np.ndarray | None = None
    smoothness: np.ndarray | None = None
    max_steps: int | None = None

    @property
    def best_index(self) -> int:
        """Index of the car with the largest lap progress."""
        return int(np.argmax(self.progress))


def run_generation(
    fleet: RacingFleet,
    population: np.ndarray,
    forward: Controller,
    *,
    view: FleetDrawable | None = None,
    inspect: Inspector | None = None,
    label: str = "",
) -> GenerationResult:
    """Drive the whole population until the fleet is done and return per-car results.

    `population` has one row of controller weights per car. `forward(weights, observation)`
    returns the [steer, throttle] action for a living car; dead cars are not queried.
    If `view` is given it is drawn after every step; `inspect(weights, observation)` supplies
    the network state of the leading living car for the panel.
    """
    population = np.asarray(population)
    if population.shape[0] != fleet.n_cars:
        raise ValueError(
            f"population size {population.shape[0]} does not match fleet size {fleet.n_cars}"
        )

    observation = fleet.reset()
    while not fleet.done:
        actions = np.zeros((fleet.n_cars, fleet.action_size))
        for index in np.flatnonzero(fleet.alive):
            actions[index] = forward(population[index], observation[index])
        observation, _ = fleet.step(actions)

        if view is not None:
            living = np.flatnonzero(fleet.alive)
            lines = [label, f"Живых: {len(living)}/{fleet.n_cars}", f"Шаг: {fleet.steps}"]
            network = None
            if inspect is not None and len(living) > 0:
                leader = living[int(np.argmax(fleet.progress[living]))]
                network = inspect(population[leader], observation[leader])
            view.draw(fleet, lines, network)

    return GenerationResult(
        progress=fleet.progress.copy(),
        finished=fleet.finished.copy(),
        steps_alive=fleet.steps_alive.copy(),
        lap_steps=fleet.lap_steps.copy(),
        mean_speed=fleet.mean_speed.copy(),
        centering=fleet.centering.copy(),
        smoothness=fleet.smoothness.copy(),
        max_steps=fleet.max_steps,
    )
