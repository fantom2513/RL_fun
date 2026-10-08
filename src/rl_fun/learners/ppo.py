"""PPO learner: on-policy reinforcement learning with a clipped surrogate objective.

One iteration collects `rollout_steps` steps from every car of the fleet (cars that crash, finish
or run out of time are respawned at once), then updates the policy and value networks. The torch
code lives in `rl_fun.learners.ppo_core`, imported only when a `PPOLearner` is created, so that
importing this module (or the lab) never loads torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from rl_fun.lab.config import RunConfig
from rl_fun.learners.base import IterationResult
from rl_fun.learners.rewards import StepReward
from rl_fun.learners.schemas import PPO_SCHEMA, copy_schema
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import FleetDrawable

HOT_PARAMS = (
    "learning_rate",
    "entropy_coef",
    "clip_epsilon",
    "gamma",
    "gae_lambda",
    "value_coef",
    "crash_penalty",
)
"""PPO parameters that may change while a run is going."""
STRUCTURAL_PARAMS = ("rollout_steps", "epochs", "minibatch_size", "initial_std", "max_grad_norm")
"""PPO parameters fixed when the run starts."""
BEST_WINDOW = 5
"""The best snapshot is taken at the best moving average of `mean_return` over this many
iterations."""


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminals: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation over a rollout; arrays of shape (steps, cars).

    `next_values[t]` is the value of the state reached after step t within the same episode (for a
    step cut by the time limit, the value of the last observation before the respawn). `terminals`
    marks real ends (crash, finish) with no bootstrap; `dones` marks every episode end, terminal or
    cut, where the advantage does not carry over to the next step. Returns (advantages, returns).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    keep_value = 1.0 - np.asarray(terminals, dtype=np.float64)
    carry = 1.0 - np.asarray(dones, dtype=np.float64)
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1:])
    for step in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[step] + gamma * next_values[step] * keep_value[step] - values[step]
        running = delta + gamma * lam * carry[step] * running
        advantages[step] = running
    return advantages, advantages + values


class PPOLearner:
    """Proximal policy optimization over a fleet of cars sharing one policy."""

    def __init__(self) -> None:
        from rl_fun.learners import ppo_core

        self._core = ppo_core
        self._agent: Any = None

    @staticmethod
    def schema() -> dict[str, Any]:
        """Parameter description of the PPO learner."""
        return copy_schema(PPO_SCHEMA)

    def setup(
        self,
        config: RunConfig,
        fleet: RacingFleet,
        rng: np.random.Generator,
        view: FleetDrawable | None,
    ) -> None:
        self.config = config
        self._fleet = fleet
        self._rng = rng
        self._view = view
        self._sizes = config.model.layer_sizes
        self._activation = config.model.activation
        self._agent = self._core.Agent(self._sizes, self._activation, config.ppo, rng)
        self._reward = self._make_reward()
        self._observation = fleet.reset()
        self._episode_return = np.zeros(fleet.n_cars)
        self._recent_returns: list[float] = []
        self._best_score = -np.inf
        self._best: dict[str, Any] | None = None

    # ---- iteration ---------------------------------------------------------------------------

    def run_iteration(self, iteration: int) -> IterationResult:
        config, fleet, agent = self.config, self._fleet, self._agent
        steps, cars = config.ppo.rollout_steps, fleet.n_cars
        weights = agent.export()
        observations = np.zeros((steps, cars, fleet.observation_size), dtype=np.float32)
        actions = np.zeros((steps, cars, fleet.action_size), dtype=np.float32)
        log_probs = np.zeros((steps, cars), dtype=np.float32)
        rewards = np.zeros((steps, cars))
        values = np.zeros((steps, cars))
        cut_values = np.zeros((steps, cars))
        terminals = np.zeros((steps, cars), dtype=bool)
        truncations = np.zeros((steps, cars), dtype=bool)
        episodes: list[tuple[float, bool, float, float]] = []

        observation = self._observation
        for step in range(steps):
            action, log_prob, value = agent.act(observation, self._rng)
            next_observation, _ = fleet.step(action)
            reward = self._reward.compute(fleet)
            self._episode_return += reward
            terminal = fleet.just_crashed | fleet.just_finished
            truncated = ~terminal & (fleet.steps_alive >= config.max_steps)
            done = terminal | truncated

            observations[step] = observation
            actions[step] = action
            log_probs[step] = log_prob
            rewards[step] = reward
            values[step] = value
            terminals[step] = terminal
            truncations[step] = truncated
            if truncated.any():
                cut_values[step, truncated] = agent.values(next_observation[truncated])
            for index in np.flatnonzero(done):
                episodes.append(
                    (
                        float(fleet.progress[index]),
                        bool(fleet.just_finished[index]),
                        float(fleet.lap_steps[index]),
                        float(self._episode_return[index]),
                    )
                )
            self._episode_return[done] = 0.0

            self._draw(iteration, next_observation, weights)
            observation = fleet.respawn(done) if done.any() else next_observation
        self._observation = observation

        bootstrap = np.concatenate([values[1:], agent.values(observation)[None, :]], axis=0)
        next_values = np.where(truncations, cut_values, bootstrap)
        ppo = config.ppo
        advantages, returns = compute_gae(
            rewards, values, next_values, terminals, terminals | truncations, ppo.gamma,
            ppo.gae_lambda,
        )
        batch = {
            "observations": observations.reshape(steps * cars, -1),
            "actions": actions.reshape(steps * cars, -1),
            "log_probs": log_probs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        metrics = agent.update(batch, ppo, self._rng)
        return self._summarize(iteration, episodes, metrics, weights)

    def _summarize(
        self,
        iteration: int,
        episodes: list[tuple[float, bool, float, float]],
        metrics: dict[str, float],
        weights: np.ndarray,
    ) -> IterationResult:
        fleet = self._fleet
        if episodes:
            progress = np.array([episode[0] for episode in episodes])
            finished = np.array([episode[1] for episode in episodes])
            lap_steps = np.array([episode[2] for episode in episodes])
            episode_returns = np.array([episode[3] for episode in episodes])
        else:
            # No episode ended in this iteration: report the episodes still under way.
            progress = fleet.progress.copy()
            finished = np.zeros(fleet.n_cars, dtype=bool)
            lap_steps = np.full(fleet.n_cars, np.nan)
            episode_returns = self._episode_return.copy()
        finished_laps = lap_steps[finished & np.isfinite(lap_steps)]
        mean_return = float(np.mean(episode_returns))

        self._recent_returns.append(mean_return)
        score = float(np.mean(self._recent_returns[-BEST_WINDOW:]))
        if score > self._best_score:
            self._best_score = score
            self._best = {
                "sizes": list(self._sizes),
                "activation": self._activation,
                "generation": int(iteration),
                "fitness": score,
                "weights": weights.tolist(),
            }

        extra = dict(metrics)
        extra["std"] = float(np.mean(self._agent.std()))
        extra["mean_return"] = mean_return
        extra["episodes"] = float(len(episodes))
        return IterationResult(
            best=float(np.max(progress)),
            mean=float(np.mean(progress)),
            finished=int(np.count_nonzero(finished)),
            best_fitness=float(np.max(episode_returns)),
            best_lap_steps=int(finished_laps.min()) if finished_laps.size > 0 else None,
            params={f"ppo.{name}": getattr(self.config.ppo, name) for name in HOT_PARAMS},
            extra=extra,
        )

    def _draw(self, iteration: int, observation: np.ndarray, weights: np.ndarray) -> None:
        if self._view is None:
            return
        fleet = self._fleet
        living = np.flatnonzero(fleet.alive)
        pool = living if living.size > 0 else np.arange(fleet.n_cars)
        leader = int(pool[np.argmax(fleet.progress[pool])])
        network = reference.inspect(weights, observation[leader], self._sizes, self._activation)
        lines = [
            f"Лаборатория, PPO, итерация {iteration + 1}",
            f"Живых: {living.size}/{fleet.n_cars}",
            f"Шаг: {fleet.steps}",
        ]
        self._view.draw(fleet, lines, network)

    # ---- parameters and snapshot -------------------------------------------------------------

    def apply_update(self, params: dict[str, Any]) -> list[str]:
        notices: list[str] = []
        for key, value in params.items():
            if key == "ppo":
                notices.extend(self._update_ppo(value))
            elif key == "fitness":
                try:
                    self.config = replace(self.config, fitness=value)
                except ValueError as error:
                    notices.append(f"параметр fitness не применён: {error}")
            else:
                notices.append(f"параметр {key} нельзя менять во время запуска")
        self._reward = self._make_reward()
        return notices

    def _update_ppo(self, values: Any) -> list[str]:
        if not isinstance(values, Mapping):
            return ["параметр ppo: ожидался объект с параметрами PPO"]
        notices: list[str] = []
        for name, value in values.items():
            path = f"ppo.{name}"
            if name in STRUCTURAL_PARAMS:
                notices.append(f"параметр {path} нельзя менять во время запуска")
            elif name not in HOT_PARAMS:
                notices.append(f"неизвестный параметр {path}")
            else:
                try:
                    ppo = replace(self.config.ppo, **{name: value})
                except (TypeError, ValueError) as error:
                    notices.append(f"параметр {path} не применён: {error}")
                else:
                    self.config = replace(self.config, ppo=ppo)
        return notices

    def best_snapshot(self) -> dict[str, Any] | None:
        return None if self._best is None else dict(self._best)

    def _make_reward(self) -> StepReward:
        config = self.config
        return StepReward.from_fitness(
            config.fitness, self._fleet.dt, config.max_steps, config.ppo.crash_penalty
        )
