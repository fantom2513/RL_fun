"""Validated, bounded training configuration for the Snake sandbox."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class SnakeConfig:
    algorithm: str = "dqn"
    observation_mode: str = "local"
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
    mutation_scale: float = 0.02
    evaluation_episodes: int = 8
    evaluation_interval: int = 1
    win_reward: float = 0.0
    food_reward: float = 10.0
    death_penalty: float = 10.0
    step_penalty: float = 0.01
    distance_reward: float = 0.0
    idle_steps: int = 0
    idle_penalty: float = 0.05
    rollout_steps: int = 512
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    action_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    tau: float = 0.005
    curriculum_steps: int = 0

    def __post_init__(self):
        if self.algorithm not in (
            "dqn",
            "double_dqn",
            "evolution",
            "reinforce",
            "a2c",
            "ppo",
            "td3",
        ):
            raise ValueError("алгоритм: dqn, double_dqn, reinforce, a2c, ppo, td3 или evolution")
        if self.observation_mode not in ("local", "board", "egocentric"):
            raise ValueError("наблюдение: local, board или egocentric")
        integer_bounds = {
            "size": (8, 32),
            "seed": (0, 2**31 - 1),
            "episodes": (1, 100000),
            "generations": (1, 10000),
            "max_steps": (10, 100000),
            "batch_size": (4, 512),
            "warmup": (4, 20000),
            "replay_capacity": (8, 100000),
            "train_interval": (1, 100),
            "target_interval": (1, 10000),
            "epsilon_decay": (1, 100000),
            "population": (4, 128),
            "evaluation_episodes": (1, 100),
            "evaluation_interval": (1, 10000),
            "idle_steps": (0, 10000),
            "rollout_steps": (4, 10000),
            "ppo_epochs": (1, 50),
            "policy_delay": (1, 100),
            "curriculum_steps": (0, 10000000),
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
            "food_reward": (0, 1000),
            "win_reward": (0, 10000),
            "death_penalty": (0, 1000),
            "step_penalty": (0, 10),
            "distance_reward": (0, 10),
            "idle_penalty": (0, 100),
            "gae_lambda": (0, 1),
            "entropy_coefficient": (0, 1),
            "value_coefficient": (0, 10),
            "ppo_clip": (0.01, 0.5),
            "action_noise": (0, 1),
            "target_noise": (0, 1),
            "noise_clip": (0, 1),
            "tau": (0.000001, 1),
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
        if self.algorithm == "evolution" and self.curriculum_steps:
            raise ValueError("curriculum поддерживается алгоритмами RL")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("параметры должны быть объектом")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("неизвестные параметры: " + ", ".join(sorted(unknown)))
        return cls(**data)

    def env_options(self):
        return {
            "observation_mode": self.observation_mode,
            "win_reward": self.win_reward,
            "food_reward": self.food_reward,
            "death_penalty": self.death_penalty,
            "step_penalty": self.step_penalty,
            "distance_reward": self.distance_reward,
            "idle_steps": self.idle_steps,
            "idle_penalty": self.idle_penalty,
        }

    def training_reset_options(self, steps, rng):
        if not self.curriculum_steps or rng.random() < 0.5:
            return None
        progress = min(1.0, steps / self.curriculum_steps)
        longest = int(self.size**2 * (0.25 + 0.7 * progress))
        return {"initial_length": int(rng.integers(4, max(5, longest + 1)))}

    def to_dict(self):
        return asdict(self)
