"""Behaviour cloning of the Hamiltonian-cycle teacher from streamed rollouts."""

from collections.abc import Callable

import torch
import torch.nn.functional as functional

from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.teacher import allowed_distances
from rl_fun.snake.gpu.vec_env import VecSnake


def clone(
    model: ConvActorCritic,
    vec: VecSnake,
    iterations: int,
    *,
    rollout: int = 4,
    learning_rate: float = 1e-3,
    explore: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Train the policy to pick the teacher's move; the value head is left untouched.

    Every iteration plays ``rollout`` steps in all games of ``vec`` and fits one batch.
    With probability ``explore`` a game takes a random move that is still safe for the
    teacher's invariant, so the data covers more states; labels stay the teacher's choice.
    """
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("value_head")
    ]
    optimizer = optimizer or torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    model.train()
    for iteration in range(iterations):
        observations, labels = [], []
        for _ in range(rollout):
            distances = allowed_distances(vec)
            best = distances.argmax(dim=1)
            observations.append(vec.observation())
            labels.append(best)
            actions = best
            if explore:
                noise = torch.rand(distances.shape, generator=vec.generator, device=vec.device)
                random_move = torch.where(distances >= 0, noise, -1.0).argmax(dim=1)
                coin = torch.rand(best.shape, generator=vec.generator, device=vec.device)
                actions = torch.where(coin < explore, random_move, best)
            vec.step(actions)
        observation, target = torch.cat(observations), torch.cat(labels)
        logits, _ = model(observation)
        loss = functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "iteration": iteration,
            "loss": float(loss.detach()),
            "accuracy": float((logits.argmax(dim=1) == target).float().mean()),
        }
        history.append(row)
        if on_step is not None:
            on_step(iteration, row)
    return history
