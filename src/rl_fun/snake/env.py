"""Authoritative seeded Snake rules shared by learning and interactive play."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
TURNS = (0, -1, 1)


def relative_action(direction: int, requested_direction: int) -> int:
    """Translate absolute right/down/left/up input; ignore reversals."""
    if type(requested_direction) is not int or requested_direction not in range(4):
        raise ValueError("направление должно быть от 0 до 3")
    return {0: 0, 1: 2, 2: 0, 3: 1}[(requested_direction - direction) % 4]


class SnakeEnv(gym.Env):
    """An 11-feature environment with straight/left/right relative actions."""

    metadata = {"render_modes": []}

    def __init__(self, size: int = 16, starvation_factor: int = 100):
        if type(size) is not int or not 8 <= size <= 32:
            raise ValueError("размер поля должен быть целым числом от 8 до 32")
        if type(starvation_factor) is not int or starvation_factor < 1:
            raise ValueError("лимит голода должен быть положительным целым числом")
        self.size = size
        self.starvation_factor = starvation_factor
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, (11,), dtype=np.float32)
        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] | None = None
        self.direction = 0
        self.score = self.steps = self.since_food = 0
        self.done = True
        self.reason = "ready"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        middle = self.size // 2
        self.snake = [(middle - i, middle) for i in range(3)]
        self.direction = 0
        self.score = self.steps = self.since_food = 0
        self.done = False
        self.reason = "playing"
        self._place_food()
        return self.observation(), self._info()

    def _place_food(self):
        occupied = set(self.snake)
        free = [
            (x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied
        ]
        self.food = free[int(self.np_random.integers(len(free)))] if free else None

    def _next(self, direction):
        x, y = self.snake[0]
        dx, dy = DIRECTIONS[direction]
        return x + dx, y + dy

    def _danger(self, position):
        x, y = position
        body = self.snake if position == self.food else self.snake[:-1]
        return not (0 <= x < self.size and 0 <= y < self.size) or position in body

    def observation(self):
        if not self.snake:
            return np.zeros(11, dtype=np.float32)
        x, y = self.snake[0]
        fx, fy = self.food if self.food is not None else (x, y)
        return np.asarray(
            [
                *(self._danger(self._next((self.direction + turn) % 4)) for turn in TURNS),
                *(self.direction == d for d in range(4)),
                fx < x,
                fx > x,
                fy < y,
                fy > y,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("действие должно быть целым числом от 0 до 2")
        if action not in range(3):
            raise ValueError("действие должно быть от 0 до 2")
        if self.done:
            raise RuntimeError("эпизод завершён; начните новую игру")
        self.direction = (self.direction + TURNS[action]) % 4
        head = self._next(self.direction)
        self.steps += 1
        self.since_food += 1
        terminated = self._danger(head)
        reward = -0.01
        if terminated:
            reward = -10.0
            self.reason = "collision"
        else:
            self.snake.insert(0, head)
            if head == self.food:
                reward = 10.0
                self.score += 1
                self.since_food = 0
                self._place_food()
                if self.food is None:
                    terminated = True
                    self.reason = "win"
            else:
                self.snake.pop()
        truncated = not terminated and self.since_food >= self.starvation_factor * self.size**2
        if truncated:
            self.reason = "starvation"
        self.done = terminated or truncated
        return self.observation(), reward, terminated, truncated, self._info()

    def _info(self):
        return {"score": self.score, "steps": self.steps, "reason": self.reason}

    def snapshot(self):
        return {
            "size": self.size,
            "snake": [list(cell) for cell in self.snake],
            "food": list(self.food) if self.food is not None else None,
            "direction": self.direction,
            "done": self.done,
            **self._info(),
        }

    def state_dict(self):
        return {
            **self.snapshot(),
            "since_food": self.since_food,
            "rng": self.np_random.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("несовместимый размер игрового поля")
        self.reset(seed=0)
        self.snake = [tuple(cell) for cell in state["snake"]]
        self.food = tuple(state["food"]) if state["food"] else None
        for key in ("direction", "score", "steps", "since_food", "done", "reason"):
            setattr(self, key, state[key])
        self.np_random.bit_generator.state = state["rng"]
