from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    algorithm: str
    seeds: tuple[int, ...]
    steps: int
    workers: int = 1
    output_root: str = "runs"
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.algorithm.strip():
            raise ValueError("algorithm must not be empty")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        config = cls(
            name=str(values["name"]),
            algorithm=str(values["algorithm"]),
            seeds=tuple(int(seed) for seed in values["seeds"]),
            steps=int(values["steps"]),
            workers=int(values.get("workers", 1)),
            output_root=str(values.get("output_root", "runs")),
            parameters=dict(values.get("parameters", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
