"""Reference neuroevolution: a small tanh MLP evolved with elitism and Gaussian mutation.

The weights of one network live in a flat vector, layer after layer: a matrix of shape
(out, in) followed by a bias vector of shape (out,). The reference uses the same
`run_generation` as a user notebook would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import FleetDrawable, run_generation


@dataclass(frozen=True)
class EvolutionParams:
    """Hyperparameters of the reference evolution."""

    population: int = 60
    hidden: tuple[int, ...] = (6, 5)
    elite: int = 6
    mutation_rate: float = 0.15
    mutation_scale: float = 0.3
    init_scale: float = 1.0


@dataclass
class ReferenceRun:
    """Outcome of `train`: per-generation history and the best weights seen overall."""

    history: list[dict[str, Any]]
    best_weights: np.ndarray
    sizes: list[int]
    params: EvolutionParams
    seed: int
    track: str
    best_fitness: float = field(default=-np.inf)


def layer_sizes(n_inputs: int, hidden: tuple[int, ...], n_outputs: int = 2) -> list[int]:
    """Sizes of all layers from the input to the output."""
    return [n_inputs, *hidden, n_outputs]


def weight_count(sizes: list[int]) -> int:
    """Number of weights including biases."""
    return sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:], strict=True))


def init_population(rng: np.random.Generator, n: int, sizes: list[int], scale: float) -> np.ndarray:
    """Random population of shape (n, weight_count); weights ~ N(0, scale^2 / fan_in), biases 0."""
    population = np.zeros((n, weight_count(sizes)))
    offset = 0
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        size = fan_in * fan_out
        weights = rng.normal(0.0, scale / np.sqrt(fan_in), size=(n, size))
        population[:, offset : offset + size] = weights
        offset += size + fan_out
    return population


def _unpack(weights: np.ndarray, sizes: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    offset = 0
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        matrix = weights[offset : offset + fan_in * fan_out].reshape(fan_out, fan_in)
        offset += fan_in * fan_out
        bias = weights[offset : offset + fan_out]
        offset += fan_out
        layers.append((matrix, bias))
    return layers


def forward(weights: np.ndarray, observation: np.ndarray, sizes: list[int]) -> np.ndarray:
    """Network output [steer, throttle] in [-1, 1]."""
    activation = np.asarray(observation, dtype=np.float64)
    for matrix, bias in _unpack(weights, sizes):
        activation = np.tanh(matrix @ activation + bias)
    return activation


def inspect(
    weights: np.ndarray, observation: np.ndarray, sizes: list[int]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Weight matrices (without biases) and activations [input, layer1, ..., output]."""
    activation = np.asarray(observation, dtype=np.float64)
    matrices = []
    activations = [activation]
    for matrix, bias in _unpack(weights, sizes):
        activation = np.tanh(matrix @ activation + bias)
        matrices.append(matrix)
        activations.append(activation)
    return matrices, activations


def next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    params: EvolutionParams,
) -> np.ndarray:
    """Keep the elite unchanged (best first); fill the rest with mutated elite children."""
    order = np.argsort(-fitness, kind="stable")
    elite = population[order[: params.elite]]
    n_children = population.shape[0] - params.elite
    parents = elite[rng.choice(params.elite, size=n_children)]
    mask = rng.random(parents.shape) < params.mutation_rate
    noise = rng.normal(0.0, params.mutation_scale, size=parents.shape)
    children = parents + mask * noise
    return np.concatenate([elite, children], axis=0)


def _fitness(progress: np.ndarray, finished: np.ndarray, lap_steps: np.ndarray, max_steps: int):
    lap_time = np.where(finished, lap_steps, max_steps)
    bonus = np.where(finished, 1.0 - lap_time / max_steps, 0.0)
    return progress + bonus


def train(
    track: str = "oval",
    seed: int = 7,
    generations: int = 50,
    params: EvolutionParams | None = None,
    fleet_kwargs: dict[str, Any] | None = None,
    view: FleetDrawable | None = None,
) -> ReferenceRun:
    """Evolve a population on `track` and record best/mean progress per generation."""
    params = params or EvolutionParams()
    rng = np.random.default_rng(seed)
    fleet = RacingFleet(params.population, track=track, **(fleet_kwargs or {}))
    sizes = layer_sizes(fleet.observation_size, params.hidden)
    population = init_population(rng, params.population, sizes, params.init_scale)

    history: list[dict[str, Any]] = []
    best_weights = population[0].copy()
    best_fitness = -np.inf
    for generation in range(generations):
        result = run_generation(
            fleet,
            population,
            lambda w, o: forward(w, o, sizes),
            view=view,
            inspect=lambda w, o: inspect(w, o, sizes),
            label=f"Эталон, поколение {generation + 1}/{generations}",
        )
        fitness = _fitness(result.progress, result.finished, result.lap_steps, fleet.max_steps)
        history.append(
            {
                "generation": generation,
                "best": float(result.progress.max()),
                "mean": float(result.progress.mean()),
                "finished": int(result.finished.sum()),
            }
        )
        top = int(np.argmax(fitness))
        if fitness[top] > best_fitness:
            best_fitness = float(fitness[top])
            best_weights = population[top].copy()
        population = next_generation(population, fitness, rng, params)

    return ReferenceRun(history, best_weights, sizes, params, seed, track, best_fitness)
