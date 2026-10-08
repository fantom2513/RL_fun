"""Torch part of the PPO learner: networks, the clipped loss and the update step (CPU only).

Only `rl_fun.learners.ppo` imports this module, and only when a `PPOLearner` is created, so runs
of other learners never load torch. Everything facing the learner takes and returns numpy arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from rl_fun.lab.config import PPOParams

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
HIDDEN_GAIN = math.sqrt(2.0)
"""Orthogonal-init gain of the hidden layers."""
POLICY_OUTPUT_GAIN = 0.01
"""Small output gain: the initial mean action is close to zero for every observation."""
VALUE_OUTPUT_GAIN = 0.01
"""Small output gain: the initial value estimate is close to zero everywhere. Per-step rewards are
tiny (about 1e-3), so with gain 1 the random slope of an untrained value net (for example along
the speed input) dominated the advantages and, for some seeds, taught the cars to brake and stand
still before any real signal arrived (see the lab iteration journal, iteration 2, D-E)."""


def _orthogonal(rng: np.random.Generator, rows: int, cols: int, gain: float) -> np.ndarray:
    """A (rows, cols) matrix with orthonormal rows or columns, scaled by `gain`."""
    sample = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(sample)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return gain * q[:rows, :cols]


class _MLP(nn.Module):
    """Linear layers with a hidden activation; the output layer is linear."""

    def __init__(
        self, sizes: Sequence[int], activation: str, rng: np.random.Generator, output_gain: float
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; supported: {tuple(_ACTIVATIONS)}")
        self._activation = _ACTIVATIONS[activation]
        pairs = zip(sizes[:-1], sizes[1:], strict=True)
        self.layers = nn.ModuleList(nn.Linear(fan_in, fan_out) for fan_in, fan_out in pairs)
        with torch.no_grad():
            last = len(self.layers) - 1
            for index, layer in enumerate(self.layers):
                gain = output_gain if index == last else HIDDEN_GAIN
                matrix = _orthogonal(rng, layer.out_features, layer.in_features, gain)
                layer.weight.copy_(torch.from_numpy(matrix))
                layer.bias.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index != last:
                values = self._activation(values)
        return values


class PolicyNet(nn.Module):
    """Gaussian policy: a tanh mean from an MLP over `sizes`, and a state-independent log std.

    The mean has exactly the layout of `rl_fun.racing.reference` (hidden `activation`, tanh output),
    so `export_weights` gives a flat vector that `reference.forward` evaluates to the same mean.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str,
        initial_std: float,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.body = _MLP(sizes, activation, rng, POLICY_OUTPUT_GAIN)
        self.log_std = nn.Parameter(torch.full((sizes[-1],), math.log(initial_std)))

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.body(observations))

    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def log_prob_of(self, mean: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log density of the (unclipped) actions under N(mean, std), summed over action dims."""
        z = (actions - mean) / self.std()
        return (-0.5 * z * z - self.log_std - _LOG_SQRT_2PI).sum(dim=-1)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob_of(self.mean(observations), actions)

    def entropy(self) -> torch.Tensor:
        """Entropy of the action distribution (the same for every state)."""
        return (0.5 + _LOG_SQRT_2PI + self.log_std).sum()


class ValueNet(nn.Module):
    """State-value estimate: an MLP with the policy's hidden layers and one linear output."""

    def __init__(self, sizes: Sequence[int], activation: str, rng: np.random.Generator) -> None:
        super().__init__()
        self.body = _MLP([*sizes[:-1], 1], activation, rng, VALUE_OUTPUT_GAIN)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.body(observations).squeeze(-1)


def export_weights(policy: PolicyNet) -> np.ndarray:
    """The policy mean as a flat float64 vector: per layer the (out, in) matrix, then the bias."""
    parts: list[np.ndarray] = []
    for layer in policy.body.layers:
        parts.append(layer.weight.detach().double().numpy().reshape(-1))
        parts.append(layer.bias.detach().double().numpy().reshape(-1))
    return np.concatenate(parts)


def clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """PPO clipped surrogate, negated for minimization: -mean(min(r A, clip(r) A))."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.min(unclipped, clipped).mean()


def ppo_update(
    policy: PolicyNet,
    value: ValueNet,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    params: PPOParams,
    rng: np.random.Generator,
) -> dict[str, float]:
    """`params.epochs` passes over shuffled minibatches of `batch`; returns mean loss metrics.

    `batch` holds flat arrays `observations`, `actions`, `log_probs`, `advantages`, `returns`.
    Advantages are normalized over the whole batch. The learning rate is taken from `params` on
    every call, so it can change between updates.
    """
    for group in optimizer.param_groups:
        group["lr"] = params.learning_rate
    observations = torch.as_tensor(batch["observations"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    parameters = [*policy.parameters(), *value.parameters()]
    count = observations.shape[0]
    size = min(params.minibatch_size, count)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
    steps = 0
    for _ in range(params.epochs):
        order = torch.from_numpy(rng.permutation(count))
        for start in range(0, count, size):
            index = order[start : start + size]
            new_log_probs = policy.log_prob(observations[index], actions[index])
            policy_loss = clipped_policy_loss(
                new_log_probs, old_log_probs[index], advantages[index], params.clip_epsilon
            )
            value_loss = ((value(observations[index]) - returns[index]) ** 2).mean()
            entropy = policy.entropy()
            loss = policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, params.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs[index]
                kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["kl"] += kl.item()
            steps += 1
    metrics = {name: total / steps for name, total in totals.items()}
    metrics["entropy"] = policy.entropy().item()
    return metrics


class Agent:
    """Policy, value function and optimizer behind a numpy interface for the learner."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str,
        params: PPOParams,
        rng: np.random.Generator,
    ) -> None:
        torch.set_num_threads(1)
        self.policy = PolicyNet(sizes, activation, params.initial_std, rng)
        self.value = ValueNet(sizes, activation, rng)
        self.optimizer = torch.optim.Adam(
            [*self.policy.parameters(), *self.value.parameters()],
            lr=params.learning_rate,
            eps=1e-5,
        )

    @torch.no_grad()
    def act(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sampled actions (unclipped), their log probabilities and the state values."""
        obs = torch.as_tensor(observations, dtype=torch.float32)
        mean = self.policy.mean(obs)
        noise = torch.from_numpy(rng.standard_normal(tuple(mean.shape)).astype(np.float32))
        actions = mean + self.policy.std() * noise
        log_probs = self.policy.log_prob_of(mean, actions)
        values = self.value(obs)
        return actions.numpy(), log_probs.numpy(), values.numpy()

    @torch.no_grad()
    def values(self, observations: np.ndarray) -> np.ndarray:
        return self.value(torch.as_tensor(observations, dtype=torch.float32)).numpy()

    def std(self) -> np.ndarray:
        return self.policy.std().detach().numpy().copy()

    def export(self) -> np.ndarray:
        return export_weights(self.policy)

    def update(
        self, batch: Mapping[str, np.ndarray], params: PPOParams, rng: np.random.Generator
    ) -> dict[str, float]:
        return ppo_update(self.policy, self.value, self.optimizer, batch, params, rng)
