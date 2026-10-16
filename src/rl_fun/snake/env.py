"""Authoritative seeded Snake rules shared by learning and interactive play."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_fun.snake.gpu.cycle import hamiltonian_cycle

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

    def __init__(
        self,
        size: int = 16,
        starvation_factor: int = 100,
        *,
        food_reward: float = 10.0,
        death_penalty: float = 10.0,
        step_penalty: float = 0.01,
        distance_reward: float = 0.0,
        idle_steps: int = 0,
        idle_penalty: float = 0.0,
        observation_mode: str = "local",
        win_reward: float = 0.0,
    ):
        if type(size) is not int or not 8 <= size <= 32:
            raise ValueError("размер поля должен быть целым числом от 8 до 32")
        if type(starvation_factor) is not int or starvation_factor < 1:
            raise ValueError("лимит голода должен быть положительным целым числом")
        self.size = size
        if observation_mode not in ("local", "board", "egocentric", "egocentric_cycle"):
            raise ValueError("наблюдение: local, board, egocentric или egocentric_cycle")
        if observation_mode == "egocentric_cycle" and size % 2:
            raise ValueError("наблюдение egocentric_cycle требует чётной стороны поля")
        self.observation_mode = observation_mode
        self._cycle = hamiltonian_cycle(size) if observation_mode == "egocentric_cycle" else None
        self.win_reward = win_reward
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.distance_reward = distance_reward
        self.idle_steps = idle_steps
        self.idle_penalty = idle_penalty
        self.starvation_factor = starvation_factor
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            0,
            1,
            (
                (6 if observation_mode == "egocentric_cycle" else 5) * (2 * size - 1) ** 2
                if observation_mode in ("egocentric", "egocentric_cycle")
                else 11 + (2 * size * size if observation_mode == "board" else 0),
            ),
            dtype=np.float32,
        )
        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] | None = None
        self.direction = 0
        self.score = self.steps = self.since_food = 0
        self.done = True
        self.reason = "ready"
        self.initial_length = 3
        self._view_y, self._view_x = np.indices((2 * size - 1, 2 * size - 1)) - (size - 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        middle = self.size // 2
        self.snake = [(middle - i, middle) for i in range(3)]
        self.direction = 0
        self.initial_length = (options or {}).get("initial_length", 3)
        if type(self.initial_length) is not int or not 3 <= self.initial_length < self.size**2:
            raise ValueError("начальная длина: от 3 до числа клеток минус одна")
        if self.initial_length != 3:
            path = [
                (x, y)
                for y in range(self.size)
                for x in (range(self.size) if y % 2 == 0 else reversed(range(self.size)))
            ]
            offset = int(self.np_random.integers(self.size**2 - self.initial_length))
            self.snake = list(reversed(path[offset : offset + self.initial_length]))
            if self.np_random.random() < 0.5:
                self.snake = [(self.size - 1 - x, y) for x, y in self.snake]
            for _ in range(int(self.np_random.integers(4))):
                self.snake = [(y, self.size - 1 - x) for x, y in self.snake]
            head, neck = self.snake[:2]
            self.direction = DIRECTIONS.index((head[0] - neck[0], head[1] - neck[1]))
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
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.observation_mode in ("egocentric", "egocentric_cycle"):
            width, center = 2 * self.size - 1, self.size - 1
            with_cycle = self._cycle is not None
            planes = np.zeros((6 if with_cycle else 5, width, width), dtype=np.float32)
            x, y = self.snake[0]
            planes[0, center, center] = 1
            for rank, (sx, sy) in enumerate(reversed(self.snake), start=1):
                planes[1, sy - y + center, sx - x + center] = rank / len(self.snake)
            if self.food is not None:
                fx, fy = self.food
                planes[2, fy - y + center, fx - x + center] = 1
            planes[3].fill(len(self.snake) / self.size**2)
            xs, ys = self._view_x + x, self._view_y + y
            outside = (xs < 0) | (xs >= self.size) | (ys < 0) | (ys >= self.size)
            planes[4] = outside
            if with_cycle:
                # Distance ahead along the Hamiltonian cycle from the head, 0 outside the board.
                ahead = (
                    self._cycle[ys.clip(0, self.size - 1), xs.clip(0, self.size - 1)]
                    - self._cycle[y, x]
                ) % self.size**2
                planes[5] = np.where(outside, 0, ahead / self.size**2)
            return np.rot90(planes, (self.direction + 1) % 4, axes=(1, 2)).copy().ravel()
        x, y = self.snake[0]
        fx, fy = self.food if self.food is not None else (x, y)
        local = np.asarray(
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
        if self.observation_mode == "local":
            return local
        body = np.zeros((self.size, self.size), dtype=np.float32)
        food = np.zeros_like(body)
        for rank, (sx, sy) in enumerate(reversed(self.snake), start=1):
            body[sy, sx] = rank / len(self.snake)
        if self.food is not None:
            food[self.food[1], self.food[0]] = 1
        return np.concatenate((local, body.ravel(), food.ravel()))

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
        reward = -self.step_penalty
        if self.idle_steps and self.since_food > self.idle_steps:
            reward -= self.idle_penalty
        if terminated:
            reward = -self.death_penalty
            self.reason = "collision"
        else:
            progress_reward = 0.0
            if self.distance_reward and self.food is not None:
                x, y = self.snake[0]
                fx, fy = self.food
                before = abs(x - fx) + abs(y - fy)
                after = abs(head[0] - fx) + abs(head[1] - fy)
                progress_reward = self.distance_reward * (before - after)
            self.snake.insert(0, head)
            if head == self.food:
                reward = self.food_reward
                self.score += 1
                self.since_food = 0
                self._place_food()
                if self.food is None:
                    terminated = True
                    self.reason = "win"
                    reward += self.win_reward
            else:
                self.snake.pop()
            reward += progress_reward
        truncated = not terminated and self.since_food >= self.starvation_factor * self.size**2
        if truncated:
            self.reason = "starvation"
        self.done = terminated or truncated
        return self.observation(), reward, terminated, truncated, self._info()

    def _info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "reason": self.reason,
            "occupancy": len(self.snake) / self.size**2,
        }

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
            "initial_length": self.initial_length,
            "rng": self.np_random.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("несовместимый размер игрового поля")
        self.reset(seed=0)
        self.snake = [tuple(cell) for cell in state["snake"]]
        self.food = tuple(state["food"]) if state["food"] else None
        self.initial_length = state.get("initial_length", 3)
        for key in ("direction", "score", "steps", "since_food", "done", "reason"):
            setattr(self, key, state[key])
        self.np_random.bit_generator.state = state["rng"]
