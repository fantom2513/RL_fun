"""Run configuration of the lab: validated, JSON-serializable parameters of one evolution run."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from rl_fun.racing.fitness import PRESETS, FitnessSpec
from rl_fun.racing.model_spec import ModelSpec
from rl_fun.racing.track import available_tracks

MAX_NAME_LENGTH = 40
LEARNERS = ("evolution", "ppo")

def _default_fitness() -> FitnessSpec:
    return FitnessSpec(dict(PRESETS["balanced"].weights))


def _int_field(name: str, value: Any, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"поле {name}: нужно целое число, получено {value!r}")
    if value < low or (high is not None and value > high):
        bounds = f"от {low} до {high}" if high is not None else f"не меньше {low}"
        raise ValueError(f"поле {name}: значение {value} вне допустимого диапазона ({bounds})")
    return value


def _number_field(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"поле {name}: нужно число, получено {value!r}")
    return float(value)


def _positive_field(name: str, value: Any) -> float:
    number = _number_field(name, value)
    if not number > 0:
        raise ValueError(f"поле {name}: значение должно быть больше нуля, получено {number:g}")
    return number


def _range_field(
    name: str,
    value: Any,
    low: float | None = None,
    high: float | None = None,
    *,
    low_open: bool = False,
    high_open: bool = False,
) -> float:
    """A finite number inside an interval; each end is open or closed, `None` means unbounded."""
    number = _number_field(name, value)
    below = low is not None and (number <= low if low_open else number < low)
    above = high is not None and (number >= high if high_open else number > high)
    if not math.isfinite(number) or below or above:
        left = "(" if low_open else "["
        right = ")" if high_open else "]"
        low_text = "-∞" if low is None else f"{low:g}"
        high_text = "∞" if high is None else f"{high:g}"
        raise ValueError(
            f"поле {name}: значение должно быть в интервале {left}{low_text}, {high_text}{right}, "
            f"получено {number:g}"
        )
    return number


def _model_from(value: Any) -> ModelSpec:
    if isinstance(value, ModelSpec):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("поле model: ожидался объект с описанием сети")
    try:
        return ModelSpec.from_dict(dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"поле model: некорректная модель ({error})") from error


def _fitness_from(value: Any) -> FitnessSpec:
    if isinstance(value, FitnessSpec):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("поле fitness: ожидался объект с весами слагаемых")
    try:
        if "weights" in value:
            return FitnessSpec.from_dict(dict(value))
        return FitnessSpec(dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"поле fitness: некорректная награда ({error})") from error


@dataclass(frozen=True)
class PPOParams:
    """Hyperparameters of the PPO learner. Invalid values raise `ValueError` in Russian."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 256
    initial_std: float = 0.6
    max_grad_norm: float = 0.5
    crash_penalty: float = 1.0

    def __post_init__(self) -> None:
        values: dict[str, Any] = {
            "learning_rate": _range_field(
                "ppo.learning_rate", self.learning_rate, 0, low_open=True
            ),
            "gamma": _range_field("ppo.gamma", self.gamma, 0, 1, high_open=True),
            "gae_lambda": _range_field("ppo.gae_lambda", self.gae_lambda, 0, 1),
            "clip_epsilon": _range_field(
                "ppo.clip_epsilon", self.clip_epsilon, 0, 1, low_open=True, high_open=True
            ),
            "entropy_coef": _range_field("ppo.entropy_coef", self.entropy_coef, 0),
            "value_coef": _range_field("ppo.value_coef", self.value_coef, 0),
            "rollout_steps": _int_field("ppo.rollout_steps", self.rollout_steps, 16, 1024),
            "epochs": _int_field("ppo.epochs", self.epochs, 1, 20),
            "minibatch_size": _int_field("ppo.minibatch_size", self.minibatch_size, 16, 8192),
            "initial_std": _range_field(
                "ppo.initial_std", self.initial_std, 0.05, 2, low_open=True
            ),
            "max_grad_norm": _range_field(
                "ppo.max_grad_norm", self.max_grad_norm, 0, low_open=True
            ),
            "crash_penalty": _range_field("ppo.crash_penalty", self.crash_penalty, 0),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation; `from_dict` restores an equal object."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PPOParams:
        """Build from a dict; missing keys take defaults, unknown keys are rejected."""
        if not isinstance(data, Mapping):
            raise ValueError("поле ppo: ожидался объект с параметрами PPO")
        known = {item.name for item in fields(cls)}
        for key in data:
            if key not in known:
                raise ValueError(f"неизвестное поле ppo: {key}")
        return cls(**dict(data))


def _ppo_from(value: Any) -> PPOParams:
    if isinstance(value, PPOParams):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("поле ppo: ожидался объект с параметрами PPO")
    return PPOParams.from_dict(value)


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to start one lab run. Invalid values raise `ValueError` in Russian."""

    name: str = "run"
    track: str = "circuit"
    model: ModelSpec = field(default_factory=ModelSpec)
    fitness: FitnessSpec = field(default_factory=_default_fitness)
    population: int = 60
    elite: int = 6
    mutation_rate: float = 0.15
    mutation_scale: float = 0.3
    init_scale: float = 1.0
    max_steps: int = 1500
    ray_range: float = 40.0
    generations: int | None = None
    seed: int = 7
    learner: str = "evolution"
    ppo: PPOParams = field(default_factory=PPOParams)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("поле name: название запуска не должно быть пустым")
        if len(self.name) > MAX_NAME_LENGTH:
            raise ValueError(
                f"поле name: не длиннее {MAX_NAME_LENGTH} символов, получено {len(self.name)}"
            )
        if not isinstance(self.track, str):
            raise ValueError("поле track: нужно имя трассы")
        _check_track(self.track)
        model = _model_from(self.model)
        fitness = _fitness_from(self.fitness)
        population = _int_field("population", self.population, 2, 200)
        if self.learner not in LEARNERS:
            raise ValueError(
                f"поле learner: неизвестный обучатель {self.learner!r}; "
                f"допустимые: {', '.join(LEARNERS)}"
            )
        # `elite` belongs to evolution; PPO runs ignore it, so it is not tied to the population.
        elite = _int_field(
            "elite", self.elite, 1, population - 1 if self.learner == "evolution" else None
        )
        mutation_rate = _number_field("mutation_rate", self.mutation_rate)
        if not 0 < mutation_rate <= 1:
            raise ValueError(
                f"поле mutation_rate: значение должно быть в интервале (0, 1], "
                f"получено {mutation_rate:g}"
            )
        max_steps = _int_field("max_steps", self.max_steps, 50, 5000)
        generations = self.generations
        if generations is not None:
            generations = _int_field("generations", generations, 1)
        seed = _int_field("seed", self.seed, 0)
        mutation_scale = _positive_field("mutation_scale", self.mutation_scale)
        init_scale = _positive_field("init_scale", self.init_scale)
        ray_range = _positive_field("ray_range", self.ray_range)
        ppo = _ppo_from(self.ppo)

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "fitness", fitness)
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "elite", elite)
        object.__setattr__(self, "mutation_rate", mutation_rate)
        object.__setattr__(self, "mutation_scale", mutation_scale)
        object.__setattr__(self, "init_scale", init_scale)
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "ray_range", ray_range)
        object.__setattr__(self, "generations", generations)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "ppo", ppo)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation; `from_dict` restores an equal config."""
        return {
            "name": self.name,
            "track": self.track,
            "model": self.model.to_dict(),
            "fitness": self.fitness.to_dict(),
            "population": self.population,
            "elite": self.elite,
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.mutation_scale,
            "init_scale": self.init_scale,
            "max_steps": self.max_steps,
            "ray_range": self.ray_range,
            "generations": self.generations,
            "seed": self.seed,
            "learner": self.learner,
            "ppo": self.ppo.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        """Build from a dict, as received from JSON. Missing keys take defaults."""
        if not isinstance(data, Mapping):
            raise ValueError("конфигурация запуска должна быть объектом")
        known = {item.name for item in fields(cls)}
        for key in data:
            if key not in known:
                raise ValueError(f"неизвестное поле конфигурации: {key}")
        values = dict(data)
        if "model" in values:
            values["model"] = _model_from(values["model"])
        if "fitness" in values:
            values["fitness"] = _fitness_from(values["fitness"])
        if "ppo" in values:
            values["ppo"] = _ppo_from(values["ppo"])
        return cls(**values)


def _check_track(track: str) -> None:
    names = available_tracks()
    if track not in names:
        raise ValueError(
            f"поле track: трасса {track!r} не встроенная; "
            f"допустимые имена: {', '.join(names)}"
        )
