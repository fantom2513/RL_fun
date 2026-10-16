"""Replay-based DQN with target network and resumable training state."""

import copy
from collections import deque

import numpy as np
import torch
from torch import nn

from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.frames import emit_frame
from rl_fun.snake.policy import Policy, architecture, evaluate, evaluation_rank


def td_targets(rewards, done, next_values, gamma):
    return rewards + gamma * (~done).float() * next_values


def next_q_values(online, target, double=False):
    return (
        target.gather(1, online.argmax(1, keepdim=True)).squeeze(1)
        if double
        else target.max(1).values
    )


class DQNTrainer:
    def __init__(self, config):
        self.config = config
        self.training_seed = config.seed
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = Policy(config.size, config.observation_mode)
        self.best_model = copy.deepcopy(self.model.state_dict())
        self.best_rank = (-1.0, -1.0, -1.0)
        self.last_evaluation = None
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_capacity)
        self.completed = self.steps = self.updates = 0
        self.scores = deque(maxlen=100)
        self.loss = None
        self.env = SnakeEnv(config.size, **config.env_options())
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
                rewards,
                done,
                next_q_values(
                    self.model(next_observations),
                    self.target(next_observations),
                    self.config.algorithm == "double_dqn",
                ),
                self.config.gamma,
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

    def choose_action(self, observation):
        action = (
            int(self.rng.integers(3))
            if self.rng.random() < self.epsilon
            else self.model.act(observation)
        )
        return action, action

    def run(self, control, emit, on_frame=None):
        env = self.env
        while self.completed < self.config.episodes or self.episode is not None:
            episode_number = self.completed + 1
            if self.episode is None:
                seed_limit = 1000000 if self.config.observation_mode != "local" else 2**31
                env.reset(
                    seed=int(self.rng.integers(seed_limit)),
                    options=self.config.training_reset_options(self.steps, self.rng),
                )
                self.episode = {
                    "index": 0,
                    "total": 0.0,
                    "finished": False,
                    "horizon": self.config.max_steps,
                }
                emit_frame(on_frame, env, episode=episode_number, epsilon=self.epsilon, ended=False)
            while not self.episode["finished"]:
                if not control():
                    return
                observation = env.observation()
                action, replay_action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                self.episode["index"] += 1
                done = terminated or truncated or self.episode["index"] >= self.episode["horizon"]
                self.replay.append(
                    (observation, replay_action, reward, next_observation, terminated)
                )
                self.steps += 1
                self.episode["total"] += reward
                if (
                    len(self.replay) >= max(self.config.warmup, self.config.batch_size)
                    and self.steps % self.config.train_interval == 0
                ):
                    self.learn_batch()
                if done:
                    if not (terminated or truncated):
                        env.done, env.reason = True, "max_steps"
                    self.episode["finished"] = True
                    self.completed += 1
                    self.scores.append(env.score)
                emit_frame(
                    on_frame, env, episode=episode_number, epsilon=self.epsilon, ended=bool(done)
                )
            evaluation = self.last_evaluation
            if (
                evaluation is None
                or self.completed % self.config.evaluation_interval == 0
                or self.completed >= self.config.episodes
            ):
                evaluation = evaluate(
                    self.model,
                    self.config.size,
                    range(1000000, 1000000 + self.config.evaluation_episodes),
                    self.config.max_steps,
                    control,
                    env_options=self.config.env_options(),
                )
                if evaluation is not None:
                    self.last_evaluation = evaluation
                    if evaluation_rank(evaluation) > self.best_rank:
                        self.best_rank = evaluation_rank(evaluation)
                        self.best_model = copy.deepcopy(self.model.state_dict())
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
                    "occupancy": len(env.snake) / self.config.size**2,
                    "reason": env.reason,
                    "evaluation_occupancy": evaluation["occupancy"],
                    "evaluation_win_rate": evaluation["win_rate"],
                    "evaluation_reasons": evaluation["reasons"],
                    "evaluation_iteration": self.completed
                    if self.completed % self.config.evaluation_interval == 0
                    or self.completed == 1
                    or self.completed == self.config.episodes
                    else None,
                    "frame": evaluation["frame"],
                }
            )

    def state_dict(self):
        rows = list(self.replay)
        replay = {
            "observations": torch.tensor(np.asarray([r[0] for r in rows]), dtype=torch.float32),
            "actions": torch.tensor(
                np.asarray([r[1] for r in rows]),
                dtype=torch.float32 if self.config.algorithm == "td3" else torch.long,
            ).reshape(-1, 3)
            if self.config.algorithm == "td3"
            else torch.tensor([r[1] for r in rows], dtype=torch.long),
            "rewards": torch.tensor([r[2] for r in rows], dtype=torch.float32),
            "next": torch.tensor(np.asarray([r[3] for r in rows]), dtype=torch.float32),
            "done": torch.tensor([r[4] for r in rows], dtype=torch.bool),
        }
        return {
            "version": 2,
            "architecture": architecture(self.config.size, self.config.observation_mode),
            "algorithm": self.config.algorithm,
            "config": self.config.to_dict(),
            "training_seed": self.training_seed,
            "model": self.model.state_dict(),
            "best_model": self.best_model,
            "best_rank": list(self.best_rank),
            "last_evaluation": self.last_evaluation,
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
        previous = type(self.config).from_dict(state["config"])
        if any(
            getattr(previous, key) != getattr(self.config, key)
            for key in ("algorithm", "size", "observation_mode")
        ):
            raise ValueError("для продолжения нужны прежний алгоритм, поле и наблюдение")
        self.model.load_state_dict(state["model"])
        self.training_seed = state.get("training_seed", previous.seed)
        self.best_model = copy.deepcopy(state.get("best_model", state["model"]))
        self.best_rank = tuple(state.get("best_rank", (-1.0, -1.0, -1.0)))
        self.last_evaluation = state.get("last_evaluation")
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
            stored_action = action.numpy().copy() if self.config.algorithm == "td3" else int(action)
            self.replay.append((obs.numpy(), stored_action, float(reward), nxt.numpy(), bool(done)))
