"""Categorical REINFORCE, A2C and PPO with resumable on-policy trajectories."""

import copy
from collections import deque
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.frames import emit_frame
from rl_fun.snake.policy import Policy, architecture, evaluate, evaluation_rank

ON_POLICY = ("reinforce", "a2c", "ppo")
ROLLOUT_KEYS = (
    "observations",
    "actions",
    "rewards",
    "values",
    "next_values",
    "terminated",
    "ended",
    "log_probs",
)


def make_critic(config):
    dimensions = architecture(config.size, config.observation_mode)
    return nn.Sequential(
        nn.Linear(dimensions[0], dimensions[1]),
        nn.ReLU(),
        nn.Linear(dimensions[1], dimensions[2]),
        nn.ReLU(),
        nn.Linear(dimensions[2], 1),
    )


def generalized_advantages(rewards, values, next_values, terminated, ended, gamma, gae_lambda):
    """Bootstrap truncations; stop GAE carry at every reset boundary."""
    deltas = rewards + gamma * (~terminated).float() * next_values - values
    advantages = torch.empty_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype)
    for index in range(len(rewards) - 1, -1, -1):
        carry = deltas[index] + gamma * gae_lambda * (~ended[index]).float() * carry
        advantages[index] = carry
    return advantages, advantages + values


def ppo_objective(log_probs, old_log_probs, advantages, clip):
    ratio = (log_probs - old_log_probs).exp()
    return torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)


class OnPolicyTrainer:
    def __init__(self, config):
        if config.algorithm not in ON_POLICY:
            raise ValueError("нужен REINFORCE, A2C или PPO")
        self.config = config
        self.training_seed = config.seed
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = Policy(config.size, config.observation_mode)
        self.critic = make_critic(config)
        self.optimizer = torch.optim.Adam(
            chain(self.model.parameters(), self.critic.parameters()), lr=config.learning_rate
        )
        self.best_model = copy.deepcopy(self.model.state_dict())
        self.best_rank = (-1.0, -1.0, -1.0)
        self.last_evaluation = None
        self.env = SnakeEnv(config.size, **config.env_options())
        self.completed = self.steps = self.updates = 0
        self.scores = deque(maxlen=100)
        self.loss = self.entropy = None
        self.episode = None
        self.rollout = []
        self.optimization = None

    def learn_rollout(self, control=lambda: True):
        if not self.rollout:
            return True
        columns = self.rollout_state()
        observations, actions = columns["observations"], columns["actions"]
        if self.config.algorithm == "reinforce":
            returns = torch.empty_like(columns["rewards"])
            carry = 0.0
            for index in range(len(returns) - 1, -1, -1):
                carry = columns["rewards"][index] + self.config.gamma * carry
                returns[index] = carry
            advantages = returns
        else:
            advantages, returns = generalized_advantages(
                columns["rewards"],
                columns["values"],
                columns["next_values"],
                columns["terminated"],
                columns["ended"],
                self.config.gamma,
                self.config.gae_lambda if self.config.algorithm == "ppo" else 1.0,
            )
        ppo = self.config.algorithm == "ppo"
        if ppo and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        if self.optimization is None:
            self.optimization = {
                "epoch": 0,
                "offset": 0,
                "indices": None,
                "epochs": self.config.ppo_epochs if ppo else 1,
                "batch_size": self.config.batch_size if ppo else len(actions),
            }
        cursor = self.optimization
        while cursor["epoch"] < cursor["epochs"]:
            if cursor["indices"] is None:
                if not control():
                    return False
                cursor["indices"] = (
                    torch.randperm(len(actions)) if ppo else torch.arange(len(actions))
                )
            while cursor["offset"] < len(actions):
                if not control():
                    return False
                batch = cursor["indices"][
                    cursor["offset"] : cursor["offset"] + cursor["batch_size"]
                ]
                distribution = Categorical(logits=self.model(observations[batch]))
                log_probs = distribution.log_prob(actions[batch])
                objective = (
                    ppo_objective(
                        log_probs,
                        columns["log_probs"][batch],
                        advantages[batch],
                        self.config.ppo_clip,
                    )
                    if ppo
                    else log_probs * advantages[batch]
                )
                entropy = distribution.entropy().mean()
                loss = -objective.mean() - self.config.entropy_coefficient * entropy
                if self.config.algorithm != "reinforce":
                    values = self.critic(observations[batch]).squeeze(-1)
                    loss = loss + self.config.value_coefficient * nn.functional.mse_loss(
                        values, returns[batch]
                    )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    chain(self.model.parameters(), self.critic.parameters()), 0.5
                )
                self.optimizer.step()
                self.updates += 1
                self.loss, self.entropy = float(loss.detach()), float(entropy.detach())
                cursor["offset"] += len(batch)
            cursor["epoch"] += 1
            cursor["offset"], cursor["indices"] = 0, None
        self.rollout.clear()
        self.optimization = None
        return True

    def run(self, control, emit, on_frame=None):
        while self.completed < self.config.episodes or self.episode is not None:
            if self.optimization is not None and not self.learn_rollout(control):
                return
            if self.episode is None:
                self.env.reset(
                    seed=int(self.rng.integers(1000000)),
                    options=self.config.training_reset_options(self.steps, self.rng),
                )
                self.episode = {"index": 0, "total": 0.0, "finished": False}
                emit_frame(on_frame, self.env, episode=self.completed + 1, ended=False)
            while not self.episode["finished"]:
                if not control():
                    return
                observation = self.env.observation()
                with torch.no_grad():
                    tensor = torch.as_tensor(observation)
                    distribution = Categorical(logits=self.model(tensor))
                    action = distribution.sample()
                    value = float(self.critic(tensor).squeeze())
                    log_prob = float(distribution.log_prob(action))
                nxt, reward, terminated, truncated, _ = self.env.step(int(action))
                self.episode["index"] += 1
                ended = terminated or truncated or self.episode["index"] >= self.config.max_steps
                with torch.no_grad():
                    next_value = float(self.critic(torch.as_tensor(nxt)).squeeze())
                self.rollout.append(
                    (
                        observation,
                        int(action),
                        reward,
                        value,
                        next_value,
                        terminated,
                        ended,
                        log_prob,
                    )
                )
                self.steps += 1
                self.episode["total"] += reward
                if ended:
                    if not (terminated or truncated):
                        self.env.done, self.env.reason = True, "max_steps"
                    self.episode["finished"] = True
                    self.completed += 1
                    self.scores.append(self.env.score)
                update = (
                    ended
                    if self.config.algorithm == "reinforce"
                    else len(self.rollout) >= self.config.rollout_steps
                )
                if update or (ended and self.completed >= self.config.episodes):
                    if not self.learn_rollout(control):
                        return
                emit_frame(
                    on_frame,
                    self.env,
                    episode=self.completed if ended else self.completed + 1,
                    ended=bool(ended),
                )
            evaluation_iteration = None
            if (
                self.last_evaluation is None
                or self.completed % self.config.evaluation_interval == 0
                or self.completed >= self.config.episodes
            ):
                result = evaluate(
                    self.model,
                    self.config.size,
                    range(1000000, 1000000 + self.config.evaluation_episodes),
                    self.config.max_steps,
                    control,
                    env_options=self.config.env_options(),
                )
                if result is None:
                    return
                self.last_evaluation = result
                evaluation_iteration = self.completed
                if evaluation_rank(result) > self.best_rank:
                    self.best_rank = evaluation_rank(result)
                    self.best_model = copy.deepcopy(self.model.state_dict())
            result = self.last_evaluation
            total = self.episode["total"]
            self.episode = None
            emit(
                {
                    "iteration": self.completed,
                    "score": self.env.score,
                    "reward": total,
                    "mean_score": float(np.mean(self.scores)),
                    "loss": self.loss,
                    "entropy": self.entropy,
                    "updates": self.updates,
                    "occupancy": len(self.env.snake) / self.config.size**2,
                    "reason": self.env.reason,
                    "evaluation_score": result["score"],
                    "evaluation_occupancy": result["occupancy"],
                    "evaluation_win_rate": result["win_rate"],
                    "evaluation_reasons": result["reasons"],
                    "evaluation_iteration": evaluation_iteration,
                    "frame": result["frame"],
                }
            )

    def rollout_state(self):
        types = (
            torch.float32,
            torch.long,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.bool,
            torch.bool,
            torch.float32,
        )
        result = {
            key: torch.tensor(np.asarray([row[index] for row in self.rollout]), dtype=dtype)
            for index, (key, dtype) in enumerate(zip(ROLLOUT_KEYS, types, strict=True))
        }
        if not self.rollout:
            result["observations"] = torch.empty(
                (0, architecture(self.config.size, self.config.observation_mode)[0])
            )
        return result

    def state_dict(self):
        return {
            "version": 2,
            "algorithm": self.config.algorithm,
            "architecture": architecture(self.config.size, self.config.observation_mode),
            "config": self.config.to_dict(),
            "training_seed": self.training_seed,
            "model": copy.deepcopy(self.model.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "best_model": copy.deepcopy(self.best_model),
            "best_rank": list(self.best_rank),
            "last_evaluation": copy.deepcopy(self.last_evaluation),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "rollout": self.rollout_state(),
            "optimization": copy.deepcopy(self.optimization),
            "completed": self.completed,
            "steps": self.steps,
            "updates": self.updates,
            "scores": list(self.scores),
            "loss": self.loss,
            "entropy": self.entropy,
            "episode": copy.deepcopy(self.episode),
            "environment": self.env.state_dict() if self.episode is not None else None,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        previous = type(self.config).from_dict(state["config"])
        if any(
            getattr(previous, key) != getattr(self.config, key)
            for key in ("algorithm", "size", "observation_mode")
        ):
            raise ValueError("для продолжения нужны прежний алгоритм, поле и наблюдение")
        self.model.load_state_dict(state["model"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.training_seed = state.get("training_seed", previous.seed)
        self.best_model, self.best_rank = (
            copy.deepcopy(state["best_model"]),
            tuple(state["best_rank"]),
        )
        self.last_evaluation = copy.deepcopy(state["last_evaluation"])
        for key in ("completed", "steps", "updates", "loss", "entropy"):
            setattr(self, key, state[key])
        self.scores = deque(state["scores"], maxlen=100)
        self.episode = copy.deepcopy(state["episode"])
        self.optimization = copy.deepcopy(state.get("optimization"))
        if self.episode is not None:
            self.env.load_state_dict(state["environment"])
        self.rollout = [
            (
                obs.numpy().copy(),
                int(action),
                float(reward),
                float(value),
                float(nxt),
                bool(terminal),
                bool(ended),
                float(logp),
            )
            for obs, action, reward, value, nxt, terminal, ended, logp in zip(
                *(state["rollout"][key] for key in ROLLOUT_KEYS), strict=True
            )
        ]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        torch.set_rng_state(state["torch_rng"])
