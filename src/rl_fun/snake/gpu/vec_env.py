"""Batched Snake on tensors: many games stepped at once with the rules of ``SnakeEnv``."""

from __future__ import annotations

import torch

from rl_fun.snake.gpu.cycle import hamiltonian_cycle

REASONS = ("playing", "collision", "win", "starvation")
START_MODES = ("standard", "cycle_segment", "mixed")
_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_TURNS = (0, -1, 1)


class VecSnake:
    """``num_envs`` games on a ``size`` x ``size`` board with relative actions 0/1/2.

    The ``body`` grid stores, for every occupied cell, how many more steps it stays
    occupied: the head holds the snake length and the tail holds 1. Moving decrements
    the grid unless the snake eats, so a move is a few batched tensor operations.
    Finished games restart at once; ``step`` reports their final outcome in ``info``.
    """

    def __init__(
        self,
        size: int,
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        *,
        starvation_factor: int = 100,
        food_reward: float = 10.0,
        death_penalty: float = 10.0,
        step_penalty: float = 0.01,
        win_reward: float = 0.0,
        start_mode: str = "standard",
        max_start_length: int | None = None,
    ):
        if type(size) is not int or not 4 <= size <= 32:
            raise ValueError("размер поля должен быть целым числом от 4 до 32")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("число игр должно быть положительным целым числом")
        if type(starvation_factor) is not int or starvation_factor < 1:
            raise ValueError("лимит голода должен быть положительным целым числом")
        if start_mode not in START_MODES:
            raise ValueError("старт: standard, cycle_segment или mixed")
        if start_mode != "standard" and size % 2:
            raise ValueError("старт по циклу требует чётной стороны поля")
        self.size = size
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.starvation_factor = starvation_factor
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.win_reward = win_reward
        self.start_mode = start_mode
        self.max_start_length = (
            max(3, int(0.95 * size**2)) if max_start_length is None else max_start_length
        )
        if not 3 <= self.max_start_length < size**2:
            raise ValueError("максимальная начальная длина: от 3 до числа клеток минус одна")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        long = {"dtype": torch.long, "device": self.device}
        self.body = torch.zeros(num_envs, size, size, dtype=torch.int32, device=self.device)
        self.head = torch.zeros(num_envs, 2, **long)
        self.food = torch.zeros(num_envs, 2, **long)
        self.direction = torch.zeros(num_envs, **long)
        self.length = torch.zeros(num_envs, **long)
        self.score = torch.zeros(num_envs, **long)
        self.steps = torch.zeros(num_envs, **long)
        self.since_food = torch.zeros(num_envs, **long)
        self._directions = torch.tensor(_DIRECTIONS, **long)
        self._turns = torch.tensor(_TURNS, **long)
        if size % 2 == 0:
            order = torch.as_tensor(hamiltonian_cycle(size), device=self.device)
            cells = torch.zeros(size * size, 2, **long)
            y, x = torch.meshgrid(
                torch.arange(size, device=self.device),
                torch.arange(size, device=self.device),
                indexing="ij",
            )
            cells[order.reshape(-1)] = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
            self.cycle_order = order
            self.cycle_cells = cells
        self.reset()

    # -- starting positions -------------------------------------------------------------

    def reset(self, mask: torch.Tensor | None = None, mode: str | None = None) -> None:
        """Restart the masked games (all by default) from standard or cycle-segment starts."""
        mode = self.start_mode if mode is None else mode
        if mode not in START_MODES:
            raise ValueError("старт: standard, cycle_segment или mixed")
        everyone = torch.arange(self.num_envs, device=self.device)
        index = everyone if mask is None else everyone[mask]
        if index.numel() == 0:
            return
        if mode == "standard":
            self._standard_start(index)
        elif mode == "cycle_segment":
            self._segment_start(index)
        else:
            coin = torch.rand(index.numel(), generator=self.generator, device=self.device) < 0.5
            self._standard_start(index[coin])
            self._segment_start(index[~coin])
        self.score[index] = 0
        self.steps[index] = 0
        self.since_food[index] = 0
        self._place_food(index)

    def _standard_start(self, index: torch.Tensor) -> None:
        if index.numel() == 0:
            return
        middle = self.size // 2
        self.body[index] = 0
        for rank, offset in enumerate((2, 1, 0), start=1):
            self.body[index, middle, middle - offset] = rank
        self.head[index] = torch.tensor([middle, middle], device=self.device)
        self.direction[index] = 0
        self.length[index] = 3

    def _segment_start(self, index: torch.Tensor) -> None:
        count = index.numel()
        if count == 0:
            return
        area = self.size**2
        longest = self.max_start_length
        first = torch.randint(area, (count,), generator=self.generator, device=self.device)
        length = torch.randint(
            3, longest + 1, (count,), generator=self.generator, device=self.device
        )
        rank = torch.arange(longest, device=self.device)
        cells = self.cycle_cells[(first[:, None] + rank) % area]  # tail first
        flat = cells[..., 1] * self.size + cells[..., 0]
        values = torch.where(rank < length[:, None], rank + 1, 0).to(torch.int32)
        body = torch.zeros(count, area, dtype=torch.int32, device=self.device)
        body.scatter_(1, flat, values)
        self.body[index] = body.view(count, self.size, self.size)
        rows = torch.arange(count, device=self.device)
        head, neck = cells[rows, length - 1], cells[rows, length - 2]
        step = head - neck
        self.head[index] = head
        self.direction[index] = torch.where(
            step[:, 0] == 1,
            0,
            torch.where(step[:, 1] == 1, 1, torch.where(step[:, 0] == -1, 2, 3)),
        )
        self.length[index] = length

    def _place_food(self, index: torch.Tensor) -> None:
        if index.numel() == 0:
            return
        free = (self.body[index] == 0).view(index.numel(), -1).float()
        choice = torch.multinomial(free, 1, generator=self.generator)[:, 0]
        self.food[index] = torch.stack((choice % self.size, choice // self.size), dim=1)

    # -- inspection and test hooks ----------------------------------------------------------

    def snake_cells(self, index: int) -> list[tuple[int, int]]:
        """Cells of one snake as ``(x, y)``, head first."""
        ys, xs = torch.nonzero(self.body[index] > 0, as_tuple=True)
        ranks = self.body[index][ys, xs]
        order = torch.argsort(ranks, descending=True)
        return [(int(xs[i]), int(ys[i])) for i in order]

    def set_food(self, index: int, cell: tuple[int, int]) -> None:
        self.food[index] = torch.tensor(cell, device=self.device)

    def load_snapshot(
        self,
        index: int,
        snake: list[tuple[int, int]],
        food: tuple[int, int] | None,
        direction: int,
        since_food: int = 0,
    ) -> None:
        """Replace one game with the given snake (head first), food and heading."""
        self.body[index] = 0
        for rank, (x, y) in enumerate(reversed(snake), start=1):
            self.body[index, y, x] = rank
        self.head[index] = torch.tensor(snake[0], device=self.device)
        self.food[index] = torch.tensor(food if food is not None else (-1, -1), device=self.device)
        self.direction[index] = direction
        self.length[index] = len(snake)
        self.score[index] = 0
        self.steps[index] = 0
        self.since_food[index] = since_food

    # -- dynamics -----------------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        """Advance every game; return ``(reward, terminated, truncated, info)``.

        ``info`` holds the final ``score``, ``length``, ``steps`` and ``reason`` code of
        games that just finished (marked by ``info["done"]``); those games have already
        been restarted when this returns.
        """
        batch, size = self.num_envs, self.size
        direction = (self.direction + self._turns[actions]) % 4
        target = self.head + self._directions[direction]
        inside = ((target >= 0) & (target < size)).all(dim=1)
        flat = target[:, 1].clamp(0, size - 1) * size + target[:, 0].clamp(0, size - 1)
        body = self.body.view(batch, -1)
        occupied = body.gather(1, flat[:, None])[:, 0]
        fatal = ~inside | (occupied > 1)
        alive = ~fatal
        eat = alive & (target == self.food).all(dim=1)
        length = self.length + eat

        self.steps += 1
        self.since_food += 1
        body -= ((body > 0) & (alive & ~eat)[:, None]).to(torch.int32)
        body.scatter_(1, flat[:, None], torch.where(alive, length, occupied.long())[:, None].int())
        self.head = torch.where(alive[:, None], target, self.head)
        self.direction = direction
        self.length = length
        self.score += eat
        self.since_food = torch.where(eat, torch.zeros_like(self.since_food), self.since_food)

        won = eat & (length == size * size)
        terminated = fatal | won
        truncated = ~terminated & (self.since_food >= self.starvation_factor * size * size)
        reward = torch.full((batch,), -self.step_penalty, device=self.device)
        reward = torch.where(eat, reward.new_tensor(self.food_reward), reward)
        reward = reward + won * self.win_reward
        reward = torch.where(fatal, reward.new_tensor(-self.death_penalty), reward)

        done = terminated | truncated
        reason = torch.where(fatal, 1, torch.where(won, 2, torch.where(truncated, 3, 0)))
        info = {
            "done": done,
            "score": self.score.clone(),
            "length": self.length.clone(),
            "steps": self.steps.clone(),
            "reason": reason,
        }
        self._place_food(torch.nonzero(eat & ~won)[:, 0])
        self.reset(done)
        return reward, terminated, truncated, info
