"""Replay-based DQN with target network and resumable training state."""

import copy
from collections import deque

import numpy as np
import torch
from torch import nn

from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.policy import ARCHITECTURE, Policy, evaluate


def td_targets(rewards, done, next_values, gamma):
    return rewards + gamma * (~done).float() * next_values


class DQNTrainer:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = Policy()
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_capacity)
        self.completed = self.steps = self.updates = 0
        self.scores = deque(maxlen=100)
        self.loss = None
        self.env = SnakeEnv(config.size)
        self.episode = None

    @property
    def epsilon(self):
        fraction = max(0.0, 1 - self.completed / self.config.epsilon_decay)
        return self.config.epsilon_end + fraction * (
            self.config.epsilon_start - self.config.epsilon_end
        )

    def learn_batch(self):
        indices = self.rng.choice(len(self.replay), self.config.batch_size, replace=False)
        batch = [self.replay[int(i)] for i in indices]
        observations = torch.tensor(np.asarray([row[0] for row in batch]))
        actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
        rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
        next_observations = torch.tensor(np.asarray([row[3] for row in batch]))
        done = torch.tensor([row[4] for row in batch], dtype=torch.bool)
        with torch.no_grad():
            target = td_targets(
                rewards, done, self.target(next_observations).max(1).values, self.config.gamma
            )
        prediction = self.model(observations).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.loss = float(loss.detach())

    def run(self, control, emit):
        env = self.env
        while self.completed < self.config.episodes or self.episode is not None:
            if self.episode is None:
                env.reset(seed=int(self.rng.integers(2**31)))
                self.episode = {
                    "index": 0,
                    "total": 0.0,
                    "finished": False,
                    "horizon": self.config.max_steps,
                }
            while not self.episode["finished"]:
                if not control():
                    return
                observation = env.observation()
                action = (
                    int(self.rng.integers(3))
                    if self.rng.random() < self.epsilon
                    else self.model.act(observation)
                )
                next_observation, reward, terminated, truncated, _ = env.step(action)
                self.episode["index"] += 1
                done = terminated or truncated or self.episode["index"] >= self.episode["horizon"]
                self.replay.append((observation, action, reward, next_observation, done))
                self.steps += 1
                self.episode["total"] += reward
                if (
                    len(self.replay) >= max(self.config.warmup, self.config.batch_size)
                    and self.steps % self.config.train_interval == 0
                ):
                    self.learn_batch()
                if done:
                    self.episode["finished"] = True
                    self.completed += 1
                    self.scores.append(env.score)
            evaluation = evaluate(
                self.model,
                self.config.size,
                [10001, 10002],
                min(self.config.max_steps, 500),
                control,
            )
            if evaluation is None:
                return
            total = self.episode["total"]
            self.episode = None
            emit(
                {
                    "iteration": self.completed,
                    "score": env.score,
                    "reward": total,
                    "mean_score": float(np.mean(self.scores)),
                    "epsilon": self.epsilon,
                    "loss": self.loss,
                    "evaluation_score": evaluation["score"],
                    "frame": evaluation["frame"],
                }
            )

    def state_dict(self):
        rows = list(self.replay)
        replay = {
            "observations": torch.tensor(np.asarray([r[0] for r in rows]), dtype=torch.float32),
            "actions": torch.tensor([r[1] for r in rows], dtype=torch.long),
            "rewards": torch.tensor([r[2] for r in rows], dtype=torch.float32),
            "next": torch.tensor(np.asarray([r[3] for r in rows]), dtype=torch.float32),
            "done": torch.tensor([r[4] for r in rows], dtype=torch.bool),
        }
        return {
            "version": 1,
            "architecture": ARCHITECTURE,
            "algorithm": "dqn",
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": replay,
            "completed": self.completed,
            "steps": self.steps,
            "updates": self.updates,
            "scores": list(self.scores),
            "loss": self.loss,
            "episode": copy.deepcopy(self.episode),
            "environment": self.env.state_dict() if self.episode is not None else None,
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.completed, self.steps, self.updates = (
            state["completed"],
            state["steps"],
            state["updates"],
        )
        self.loss = state["loss"]
        self.episode = copy.deepcopy(state.get("episode"))
        if self.episode is not None:
            self.env.load_state_dict(state["environment"])
        self.scores = deque(state["scores"], maxlen=100)
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        replay = state["replay"]
        self.replay.clear()
        for obs, action, reward, nxt, done in zip(
            replay["observations"],
            replay["actions"],
            replay["rewards"],
            replay["next"],
            replay["done"],
            strict=True,
        ):
            self.replay.append((obs.numpy(), int(action), float(reward), nxt.numpy(), bool(done)))
