from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    id: str
    kwargs: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    id: str
    kwargs: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunConfig:
    seeds: tuple[int, ...]
    total_steps: int
    workers: int = 1
    output_root: str = "runs"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes: int = 5
    max_episode_steps: int | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    run: RunConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        _require_non_empty_string(self.name, "name")
        _validate_component(self.environment.id, self.environment.kwargs, "environment")
        _validate_component(self.algorithm.id, self.algorithm.kwargs, "algorithm")
        _validate_run_config(self.run)
        _validate_evaluation_config(self.evaluation)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "environment": {
                "id": self.environment.id,
                "kwargs": _copy_json_mapping(self.environment.kwargs, "environment.kwargs"),
            },
            "algorithm": {
                "id": self.algorithm.id,
                "kwargs": _copy_json_mapping(self.algorithm.kwargs, "algorithm.kwargs"),
            },
            "run": {
                "seeds": list(self.run.seeds),
                "total_steps": self.run.total_steps,
                "workers": self.run.workers,
                "output_root": self.run.output_root,
            },
            "evaluation": {
                "episodes": self.evaluation.episodes,
                "max_episode_steps": self.evaluation.max_episode_steps,
            },
        }

    @classmethod
    def from_dict(cls, values: object) -> ExperimentConfig:
        top_level = _require_json_mapping(values, "configuration")
        _require_keys(
            top_level,
            {"name", "environment", "algorithm", "run"},
            "configuration version 2",
        )
        _reject_unknown_keys(
            top_level,
            {"name", "environment", "algorithm", "run", "evaluation"},
            "top-level",
        )

        config = cls(
            name=_require_non_empty_string(top_level["name"], "name"),
            environment=_parse_environment(top_level["environment"]),
            algorithm=_parse_algorithm(top_level["algorithm"]),
            run=_parse_run(top_level["run"]),
            evaluation=_parse_evaluation(top_level.get("evaluation", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_environment(value: object) -> EnvironmentConfig:
    section = _require_json_mapping(value, "environment")
    _require_keys(section, {"id"}, "environment")
    _reject_unknown_keys(section, {"id", "kwargs"}, "environment")
    kwargs = _require_json_mapping(section.get("kwargs", {}), "environment.kwargs")
    if "render_mode" in kwargs:
        raise ValueError("environment.kwargs must not contain render_mode")
    return EnvironmentConfig(
        id=_require_non_empty_string(section["id"], "environment.id"),
        kwargs=_copy_json_mapping(kwargs, "environment.kwargs"),
    )


def _parse_algorithm(value: object) -> AlgorithmConfig:
    section = _require_json_mapping(value, "algorithm")
    _require_keys(section, {"id"}, "algorithm")
    _reject_unknown_keys(section, {"id", "kwargs"}, "algorithm")
    return AlgorithmConfig(
        id=_require_non_empty_string(section["id"], "algorithm.id"),
        kwargs=_copy_json_mapping(
            _require_json_mapping(section.get("kwargs", {}), "algorithm.kwargs"),
            "algorithm.kwargs",
        ),
    )


def _parse_run(value: object) -> RunConfig:
    section = _require_json_mapping(value, "run")
    _require_keys(section, {"seeds", "total_steps"}, "run")
    _reject_unknown_keys(section, {"seeds", "total_steps", "workers", "output_root"}, "run")
    seeds_value = section["seeds"]
    if not isinstance(seeds_value, list):
        raise ValueError("seeds must be a list of integers")
    if not seeds_value:
        raise ValueError("at least one seed is required")
    seeds = tuple(_require_int(seed, "seeds") for seed in seeds_value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return RunConfig(
        seeds=seeds,
        total_steps=_require_positive_int(section["total_steps"], "total_steps"),
        workers=_require_positive_int(section.get("workers", 1), "workers"),
        output_root=_require_non_empty_string(section.get("output_root", "runs"), "output_root"),
    )


def _parse_evaluation(value: object) -> EvaluationConfig:
    section = _require_json_mapping(value, "evaluation")
    _reject_unknown_keys(section, {"episodes", "max_episode_steps"}, "evaluation")
    max_episode_steps = section.get("max_episode_steps")
    if max_episode_steps is not None:
        max_episode_steps = _require_positive_int(max_episode_steps, "max_episode_steps")
    return EvaluationConfig(
        episodes=_require_positive_int(section.get("episodes", 5), "episodes"),
        max_episode_steps=max_episode_steps,
    )


def _validate_component(identifier: object, kwargs: object, name: str) -> None:
    _require_non_empty_string(identifier, f"{name}.id")
    mapping = _require_json_mapping(kwargs, f"{name}.kwargs")
    if name == "environment" and "render_mode" in mapping:
        raise ValueError("environment.kwargs must not contain render_mode")
    _copy_json_mapping(mapping, f"{name}.kwargs")


def _validate_run_config(config: RunConfig) -> None:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    for seed in config.seeds:
        _require_int(seed, "seeds")
        if seed < 0:
            raise ValueError("seeds must be non-negative")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")
    _require_positive_int(config.total_steps, "total_steps")
    _require_positive_int(config.workers, "workers")
    _require_non_empty_string(config.output_root, "output_root")


def _validate_evaluation_config(config: EvaluationConfig) -> None:
    _require_positive_int(config.episodes, "episodes")
    if config.max_episode_steps is not None:
        _require_positive_int(config.max_episode_steps, "max_episode_steps")


def _require_keys(values: dict[str, JSONValue], required: set[str], section: str) -> None:
    missing = sorted(required - values.keys())
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"{section} is missing required keys: {missing_keys}")


def _reject_unknown_keys(values: dict[str, JSONValue], allowed: set[str], section: str) -> None:
    unknown = sorted(values.keys() - allowed)
    if unknown:
        unknown_keys = ", ".join(unknown)
        raise ValueError(f"unknown {section} keys: {unknown_keys}")


def _require_positive_int(value: object, name: str) -> int:
    number = _require_int(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _require_json_mapping(value: object, name: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _copy_json_mapping(values: dict[str, JSONValue], name: str) -> dict[str, JSONValue]:
    return {key: _copy_json_value(value, f"{name}.{key}") for key, value in values.items()}


def _copy_json_value(value: object, name: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be JSON-compatible")
        return value
    if isinstance(value, list):
        return [_copy_json_value(item, f"{name}[]") for item in value]
    if isinstance(value, dict):
        return _copy_json_mapping(_require_json_mapping(value, name), name)
    raise ValueError(f"{name} must be JSON-compatible")
