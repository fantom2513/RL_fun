"""Greedy evaluation on the batched environment and on the authoritative reference."""

from collections import Counter
from collections.abc import Callable

import numpy as np
import torch

from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.vec_env import REASONS, VecSnake


def greedy_policy(
    model: ConvActorCritic, masked: bool = False
) -> Callable[[VecSnake], torch.Tensor]:
    """Policy over a ``VecSnake``; ``masked`` (a shield) is for secondary reporting only."""

    def policy(vec: VecSnake) -> torch.Tensor:
        return model.greedy(vec.observation(), vec.action_mask() if masked else None)

    return policy


def evaluate_vec(
    policy: Callable[[VecSnake], torch.Tensor],
    size: int,
    games: int,
    seed: int,
    device: str | torch.device = "cpu",
    max_steps: int | None = None,
    start_mode: str = "standard",
    **env_options,
) -> dict:
    """Play ``games`` games at once and report the first outcome of each."""
    max_steps = size**4 if max_steps is None else max_steps
    vec = VecSnake(size, games, device, seed, start_mode=start_mode, **env_options)
    area = size * size
    finished = torch.zeros(games, dtype=torch.bool, device=vec.device)
    reason = torch.zeros(games, dtype=torch.long, device=vec.device)
    steps = torch.zeros(games, dtype=torch.long, device=vec.device)
    length = torch.zeros(games, dtype=torch.long, device=vec.device)
    for _ in range(max_steps):
        _, _, _, info = vec.step(policy(vec))
        fresh = info["done"] & ~finished
        reason[fresh] = info["reason"][fresh]
        steps[fresh] = info["steps"][fresh]
        length[fresh] = info["length"][fresh]
        finished |= fresh
        if bool(finished.all()):
            break
    steps = torch.where(finished, steps, vec.steps)
    length = torch.where(finished, length, vec.length)
    names = [
        REASONS[code] if done else "max_steps"
        for code, done in zip(reason.tolist(), finished.tolist(), strict=True)
    ]
    reasons = dict(Counter(names))
    wins = reasons.get("win", 0)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "occupancy": float((length.float() / area).mean()),
        "steps": float(steps.float().mean()),
        "reasons": reasons,
    }


def evaluate_reference(
    model: ConvActorCritic,
    size: int,
    seeds: list[int],
    max_steps: int,
    env_options: dict | None = None,
) -> dict:
    """Greedy, unmasked play on the CPU ``SnakeEnv``, one game per seed; the acceptance path."""
    if not seeds:
        raise ValueError("для проверки нужен хотя бы один seed")
    window = 2 * size - 1
    outcomes = []
    model = model.eval()
    for seed in seeds:
        env = SnakeEnv(size, observation_mode="egocentric_cycle", **(env_options or {}))
        observation, _ = env.reset(seed=int(seed))
        reason = "max_steps"
        for _ in range(max_steps):
            planes = torch.as_tensor(observation).view(1, model.channels, window, window)
            observation, _, terminated, truncated, _ = env.step(int(model.greedy(planes)[0]))
            if terminated or truncated:
                reason = env.reason
                break
        outcomes.append(
            {
                "seed": int(seed),
                "score": env.score,
                "steps": env.steps,
                "occupancy": len(env.snake) / size**2,
                "reason": reason,
                "win": reason == "win",
            }
        )
    reasons = dict(Counter(row["reason"] for row in outcomes))
    wins = reasons.get("win", 0)
    return {
        "games": len(outcomes),
        "wins": wins,
        "win_rate": wins / len(outcomes),
        "occupancy": float(np.mean([row["occupancy"] for row in outcomes])),
        "steps": float(np.mean([row["steps"] for row in outcomes])),
        "reasons": reasons,
        "outcomes": outcomes,
    }
