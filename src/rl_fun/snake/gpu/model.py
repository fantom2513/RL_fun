"""Convolutional actor-critic over the head-centred Snake window."""

import torch
from torch import nn


class ConvActorCritic(nn.Module):
    """Policy logits for the three relative actions and a state value."""

    def __init__(self, size: int, channels: int = 6, width: int = 48, hidden: int = 512):
        super().__init__()
        self.size, self.channels, self.width, self.hidden = size, channels, width, hidden
        window = 2 * size - 1
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, channels, window, window)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, 3)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.trunk(self.features(observation))
        return self.policy_head(shared), self.value_head(shared)[:, 0]

    @torch.no_grad()
    def greedy(self, observation: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Most likely action; ``mask`` (True = allowed) removes moves from consideration."""
        logits, _ = self(observation)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits.argmax(dim=1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
