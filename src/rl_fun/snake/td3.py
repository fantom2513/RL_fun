"""TD3 on continuous preference scores, mapped to Snake turns by argmax.

The adapter creates a piecewise-constant continuous-action environment. This is
an experimental use of TD3, whose original action space is continuous.
"""

import copy
from itertools import chain

import numpy as np
import torch
from torch import nn

from rl_fun.snake.dqn import DQNTrainer, td_targets
from rl_fun.snake.policy import architecture


def make_q_critic(config):
    dimensions = architecture(config.size, config.observation_mode)
    return nn.Sequential(
        nn.Linear(dimensions[0] + 3, dimensions[1]),
        nn.ReLU(),
        nn.Linear(dimensions[1], dimensions[2]),
        nn.ReLU(),
        nn.Linear(dimensions[2], 1),
    )


class TD3Trainer(DQNTrainer):
    def __init__(self, config):
        if config.algorithm != "td3":
            raise ValueError("нужен алгоритм td3")
        super().__init__(config)
        self.model.action_adapter = self.target.action_adapter = "continuous_scores_argmax"
        self.critic1, self.critic2 = make_q_critic(config), make_q_critic(config)
        self.target_critic1, self.target_critic2 = (
            copy.deepcopy(self.critic1),
            copy.deepcopy(self.critic2),
        )
        self.critic_optimizer = torch.optim.Adam(
            chain(self.critic1.parameters(), self.critic2.parameters()), lr=config.learning_rate
        )
        self.actor_updates = 0

    def choose_action(self, observation):
        if self.steps < self.config.warmup:
            scores = self.rng.uniform(-1, 1, 3)
        else:
            with torch.no_grad():
                scores = self.model(torch.as_tensor(observation)).tanh().numpy()
            scores = scores + self.rng.normal(0, self.config.action_noise, 3)
        scores = np.clip(scores, -1, 1).astype(np.float32)
        return int(scores.argmax()), scores

    def learn_batch(self):
        indices = self.rng.choice(len(self.replay), self.config.batch_size, replace=False)
        batch = [self.replay[int(i)] for i in indices]
        observations = torch.tensor(np.asarray([row[0] for row in batch]))
        actions = torch.tensor(np.asarray([row[1] for row in batch]))
        rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
        nxt = torch.tensor(np.asarray([row[3] for row in batch]))
        done = torch.tensor([row[4] for row in batch], dtype=torch.bool)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.config.target_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            target_actions = (self.target(nxt).tanh() + noise).clamp(-1, 1)
            target_inputs = torch.cat((nxt, target_actions), dim=1)
            value = torch.minimum(
                self.target_critic1(target_inputs), self.target_critic2(target_inputs)
            ).squeeze(-1)
            target = td_targets(rewards, done, value, self.config.gamma)
        inputs = torch.cat((observations, actions), dim=1)
        loss = nn.functional.mse_loss(
            self.critic1(inputs).squeeze(-1), target
        ) + nn.functional.mse_loss(self.critic2(inputs).squeeze(-1), target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(chain(self.critic1.parameters(), self.critic2.parameters()), 10.0)
        self.critic_optimizer.step()
        self.updates += 1
        self.loss = float(loss.detach())
        if self.updates % self.config.policy_delay == 0:
            for parameter in self.critic1.parameters():
                parameter.requires_grad_(False)
            actor_inputs = torch.cat((observations, self.model(observations).tanh()), dim=1)
            actor_loss = -self.critic1(actor_inputs).mean()
            self.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            for parameter in self.critic1.parameters():
                parameter.requires_grad_(True)
            self.actor_updates += 1
            with torch.no_grad():
                for online, target_model in (
                    (self.model, self.target),
                    (self.critic1, self.target_critic1),
                    (self.critic2, self.target_critic2),
                ):
                    for parameter, target_parameter in zip(
                        online.parameters(), target_model.parameters(), strict=True
                    ):
                        target_parameter.lerp_(parameter, self.config.tau)

    def run(self, control, emit, on_frame=None):
        def metric(row):
            row.pop("epsilon", None)
            emit(
                {
                    **row,
                    "action_noise": self.config.action_noise,
                    "updates": self.updates,
                    "actor_updates": self.actor_updates,
                }
            )

        def frame(row):
            row.pop("epsilon", None)
            on_frame({**row, "action_noise": self.config.action_noise})

        if on_frame is not None:
            frame.enabled = getattr(on_frame, "enabled", lambda: True)
            if hasattr(on_frame, "should_emit"):
                frame.should_emit = on_frame.should_emit
        super().run(control, metric, frame if on_frame is not None else None)

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_updates": self.actor_updates,
                "action_adapter": "continuous_scores_argmax",
            }
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for key in ("critic1", "critic2", "target_critic1", "target_critic2"):
            getattr(self, key).load_state_dict(state[key])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.actor_updates = state["actor_updates"]
