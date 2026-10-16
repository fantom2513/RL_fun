"""Hamiltonian-cycle expert with shortcuts, vectorized over a ``VecSnake`` batch.

The body of a snake that started on the cycle always occupies cells in increasing cycle
order, so every cell ahead of the head up to the tail is free. Stepping to the next cycle
cell is therefore always safe, and a shortcut to a cell further ahead stays safe as long
as it does not jump past the food or reach the tail. Each move shortens the cycle distance
to the food, so every food is eaten and a full board is reached.
"""

import torch

from rl_fun.snake.gpu.vec_env import VecSnake


def teacher_actions(
    vec: VecSnake, shortcuts: bool = True, shortcut_fraction: float = 1.0
) -> torch.Tensor:
    """Relative actions of the expert for every game; valid for cycle-aligned snakes.

    ``shortcuts=False`` follows the cycle exactly. Shortcuts apply only while the snake is
    shorter than ``shortcut_fraction`` of the board.
    """
    batch, size = vec.num_envs, vec.size
    area = size * size
    cycle = vec.cycle_order.view(-1)
    ahead = vec.cycle_order[vec.head[:, 1], vec.head[:, 0]]
    food_distance = (vec.cycle_order[vec.food[:, 1], vec.food[:, 0]] - ahead) % area
    tail = (vec.body.view(batch, -1) == 1).int().argmax(dim=1)
    tail_distance = (cycle[tail] - ahead) % area

    directions = (vec.direction[:, None] + vec._turns[None, :]) % 4
    target = vec.head[:, None, :] + vec._directions[directions]
    inside = ((target >= 0) & (target < size)).all(dim=2)
    flat = target[..., 1].clamp(0, size - 1) * size + target[..., 0].clamp(0, size - 1)
    safe = inside & (vec.body.view(batch, -1).gather(1, flat) <= 1)
    distance = (cycle[flat] - ahead[:, None]) % area

    allowed = safe & (distance == 1)
    if shortcuts:
        reach = torch.minimum(food_distance, tail_distance - 1)[:, None]
        short_enough = (vec.length < shortcut_fraction * area)[:, None]
        allowed |= safe & (distance >= 2) & (distance <= reach) & short_enough
    return torch.where(allowed, distance, -1).argmax(dim=1)
