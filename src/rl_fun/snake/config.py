"""Validated, bounded training configuration for the Snake sandbox."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class SnakeConfig:
    algorithm: str = "dqn"
    size: int = 16
    seed: int = 42
    episodes: int = 500
    generations: int = 60
    max_steps: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.95
    batch_size: int = 64
    warmup: int = 256
    replay_capacity: int = 20000
    train_interval: int = 4
    target_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 400
    population: int = 32
    elite_fraction: float = 0.15
    mutation_scale: float = 0.1
    evaluation_episodes: int = 3

    def __post_init__(self):
        if self.algorithm not in ("dqn", "evolution"):
            raise ValueError("алгоритм: dqn или evolution")
        integer_bounds = {
            "size": (8, 32),
            "seed": (0, 2**31 - 1),
            "episodes": (1, 100000),
            "generations": (1, 10000),
            "max_steps": (10, 10000),
            "batch_size": (4, 512),
            "warmup": (4, 20000),
            "replay_capacity": (8, 100000),
            "train_interval": (1, 100),
            "target_interval": (1, 10000),
            "epsilon_decay": (1, 100000),
            "population": (4, 128),
            "evaluation_episodes": (1, 10),
        }
        for key, (low, high) in integer_bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key}: нужно целое число от {low} до {high}")
        real_bounds = {
            "learning_rate": (1e-6, 0.1),
            "gamma": (0, 1),
            "epsilon_start": (0, 1),
            "epsilon_end": (0, 1),
            "elite_fraction": (0.01, 0.5),
            "mutation_scale": (0.001, 2),
        }
        for key, (low, high) in real_bounds.items():
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{key}: нужно число от {low} до {high}")
        if self.batch_size > self.replay_capacity:
            raise ValueError("размер выборки не может превышать память")
        if self.warmup > self.replay_capacity:
            raise ValueError("опыт до обучения не может превышать память")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("конечное исследование не может превышать начальное")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("параметры должны быть объектом")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("неизвестные параметры: " + ", ".join(sorted(unknown)))
        return cls(**data)

    def to_dict(self):
        return asdict(self)
