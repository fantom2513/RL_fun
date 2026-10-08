"""Explicit algorithm registry for environment-neutral experiment runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.types import BanditOutcome
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.experiments.config import JSONValue
from rl_fun.policies import Policy, RandomPolicy
from rl_fun.rollouts import rollout_episode
from rl_fun.tracking.metrics import MetricSink

AlgorithmFunction = Callable[
    [gym.Env, int, int, np.random.Generator, MetricSink, Mapping[str, JSONValue]],
    "AlgorithmResult",
]
AlgorithmValidator = Callable[[gym.Env, str], None]


@dataclass(frozen=True, slots=True)
class AlgorithmResult:
    """Metrics and optional reusable policy produced by one algorithm run."""

    metrics: dict[str, float]
    policy: Policy | None = None


@dataclass(frozen=True, slots=True)
class AlgorithmDefinition:
    """Runtime function and environment compatibility check for an algorithm ID."""

    run: AlgorithmFunction
    validate: AlgorithmValidator


def get_algorithm(algorithm_id: str) -> AlgorithmDefinition:
    """Return the definition for a supported local algorithm identifier."""
    try:
        return _ALGORITHMS[algorithm_id]
    except KeyError as error:
        supported = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(
            f"unknown algorithm {algorithm_id!r}; supported IDs: {supported}"
        ) from error


def _run_random(
    env: gym.Env,
    total_steps: int,
    seed: int,
    rng: np.random.Generator,
    metrics: MetricSink,
    parameters: Mapping[str, JSONValue],
) -> AlgorithmResult:
    _reject_unknown_parameters(parameters, set())
    policy = RandomPolicy()
    completed_steps = 0
    completed_episodes = 0
    cumulative_reward = 0.0

    while completed_steps < total_steps:
        episode_seed = seed if completed_episodes == 0 else _next_seed(rng)
        episode = rollout_episode(
            env,
            policy,
            episode_seed,
            max_episode_steps=total_steps - completed_steps,
        )
        completed_steps += episode.length
        cumulative_reward += episode.reward
        if episode.reached_safety_limit:
            continue
        completed_episodes += 1
        metrics.log(
            completed_steps,
            {
                "episode/reward": episode.reward,
                "episode/length": float(episode.length),
                "train/steps": float(completed_steps),
                "train/episodes": float(completed_episodes),
                "train/cumulative_reward": cumulative_reward,
            },
        )

    return AlgorithmResult(
        metrics={
            "train/steps": float(completed_steps),
            "train/episodes": float(completed_episodes),
            "train/cumulative_reward": cumulative_reward,
        },
        policy=policy,
    )


def _run_bandit_greedy(
    env: gym.Env,
    total_steps: int,
    seed: int,
    rng: np.random.Generator,
    metrics: MetricSink,
    parameters: Mapping[str, JSONValue],
) -> AlgorithmResult:
    bandit = _require_bandit_environment(env, "bandit_greedy")
    _reject_unknown_parameters(parameters, set())
    env.reset(seed=seed)
    return _bandit_result(run_greedy(bandit, total_steps, rng, metrics))


def _run_bandit_epsilon(
    env: gym.Env,
    total_steps: int,
    seed: int,
    rng: np.random.Generator,
    metrics: MetricSink,
    parameters: Mapping[str, JSONValue],
) -> AlgorithmResult:
    bandit = _require_bandit_environment(env, "bandit_epsilon")
    epsilon = _require_epsilon(parameters)
    env.reset(seed=seed)
    return _bandit_result(run_epsilon_greedy(bandit, total_steps, rng, metrics, epsilon))


def _bandit_result(outcome: BanditOutcome) -> AlgorithmResult:
    return AlgorithmResult(
        metrics={
            "train/steps": float(outcome.counts.sum()),
            "train/cumulative_reward": outcome.cumulative_reward,
            "train/cumulative_regret": outcome.cumulative_regret,
        }
    )


def _require_bandit_environment(
    env: gym.Env, algorithm_id: str, environment_id: str | None = None,
) -> gym.Env:
    unwrapped = env.unwrapped
    identifier = environment_id or (
        env.spec.id if env.spec is not None else type(unwrapped).__name__
    )
    incompatible = not isinstance(unwrapped, StationaryBanditEnv)
    current = env
    while isinstance(current, gym.Wrapper):
        incompatible |= (
            isinstance(current, gym.ActionWrapper)
            or current.action_space != unwrapped.action_space
        )
        current = current.env
    if incompatible:
        raise ValueError(
            f"algorithm {algorithm_id!r} is incompatible with environment {identifier!r}; "
            "requires a StationaryBanditEnv with unchanged actions"
        )
    return env


def _validate_bandit(algorithm_id: str) -> AlgorithmValidator:
    def validate(env: gym.Env, environment_id: str) -> None:
        _require_bandit_environment(env, algorithm_id, environment_id)

    return validate


def _validate_random(env: gym.Env, environment_id: str) -> None:
    del environment_id
    if not callable(getattr(env.action_space, "sample", None)):
        raise ValueError("random algorithm requires an action space with sample()")


def _require_epsilon(parameters: Mapping[str, JSONValue]) -> float:
    _reject_unknown_parameters(parameters, {"epsilon"})
    if "epsilon" not in parameters:
        raise ValueError("bandit_epsilon requires an epsilon parameter")
    epsilon = parameters["epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be a number")
    return float(epsilon)


def _reject_unknown_parameters(parameters: Mapping[str, JSONValue], allowed: set[str]) -> None:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(unknown)}")


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))


_ALGORITHMS: dict[str, AlgorithmDefinition] = {
    "random": AlgorithmDefinition(run=_run_random, validate=_validate_random),
    "bandit_greedy": AlgorithmDefinition(
        run=_run_bandit_greedy,
        validate=_validate_bandit("bandit_greedy"),
    ),
    "bandit_epsilon": AlgorithmDefinition(
        run=_run_bandit_epsilon,
        validate=_validate_bandit("bandit_epsilon"),
    ),
}
