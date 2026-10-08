"""Composite fitness: a weighted sum of per-car terms, with ready-made presets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rl_fun.racing.generation import GenerationResult

_TERMS = ("progress", "lap_bonus", "speed", "survival", "centering", "smoothness")


def available_terms() -> tuple[str, ...]:
    """Names of the terms that can be used in a `FitnessSpec`."""
    return _TERMS


def _require(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"term {name!r} needs driving statistics that this result does not carry")
    return value


def _max_steps(result: GenerationResult) -> int:
    if result.max_steps is None:
        raise ValueError("terms 'lap_bonus' and 'survival' need the result's max_steps")
    return result.max_steps


def _term(name: str, result: GenerationResult) -> np.ndarray:
    if name == "progress":
        return np.asarray(result.progress, dtype=np.float64)
    if name == "lap_bonus":
        finished = np.asarray(result.finished, dtype=bool)
        bonus = 1.0 - np.asarray(result.lap_steps, dtype=np.float64) / _max_steps(result)
        return np.where(finished, bonus, 0.0)
    if name == "survival":
        return np.asarray(result.steps_alive, dtype=np.float64) / _max_steps(result)
    if name == "speed":
        return np.asarray(_require(result.mean_speed, name), dtype=np.float64)
    if name == "centering":
        return np.asarray(_require(result.centering, name), dtype=np.float64)
    return np.asarray(_require(result.smoothness, name), dtype=np.float64)


@dataclass(frozen=True)
class FitnessSpec:
    """Weighted sum of terms, e.g. `{"progress": 1.0, "centering": 0.5}`.

    Negative weights act as penalties. An empty mapping, an unknown term or a non-finite weight
    raises `ValueError`.
    """

    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = {name: float(value) for name, value in dict(self.weights).items()}
        if not weights:
            raise ValueError("fitness needs at least one term")
        for name, value in weights.items():
            if name not in _TERMS:
                raise ValueError(f"unknown fitness term {name!r}; supported: {_TERMS}")
            if not math.isfinite(value):
                raise ValueError(f"weight of {name!r} must be finite, got {value}")
        object.__setattr__(self, "weights", weights)

    def score(self, result: GenerationResult) -> np.ndarray:
        """Fitness of every car in the result, shape (n_cars,)."""
        total = np.zeros(np.shape(result.progress), dtype=np.float64)
        for name, weight in self.weights.items():
            total = total + weight * _term(name, result)
        return total

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation."""
        return {"weights": dict(self.weights)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitnessSpec:
        """Build from `to_dict` output; unknown keys are rejected."""
        unknown = sorted(set(data) - {"weights"})
        if unknown:
            raise ValueError(f"unknown fitness keys: {', '.join(unknown)}")
        if "weights" not in data:
            raise ValueError("fitness dict needs a 'weights' mapping")
        return cls(weights=dict(data["weights"]))


PRESETS: dict[str, FitnessSpec] = {
    "racer": FitnessSpec({"progress": 1.0, "lap_bonus": 0.5, "speed": 0.3}),
    "careful": FitnessSpec({"progress": 1.0, "centering": 0.5, "smoothness": 0.3, "survival": 0.2}),
    "balanced": FitnessSpec({"progress": 1.0, "lap_bonus": 0.3, "speed": 0.15, "centering": 0.15}),
}
