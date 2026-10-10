"""A2C: advantage actor-critic, the plain policy gradient that PPO improves on.

It is PPO with its two safeguards removed: one gradient step per rollout over the whole batch
(`epochs=1`, a minibatch as big as the batch), so the policy never moves away from the one that
collected the data and the clip has nothing to clip. Everything else is shared with PPO.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rl_fun.lab.config import RunConfig
from rl_fun.learners.ppo import PPOLearner
from rl_fun.learners.schemas import A2C_SCHEMA, copy_schema
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import FleetDrawable

WHOLE_BATCH = 8192
"""The largest minibatch the parameters allow: one step over the batch for any usual fleet."""


class A2CLearner(PPOLearner):
    @staticmethod
    def schema() -> dict[str, Any]:
        """Parameter description of the A2C learner."""
        return copy_schema(A2C_SCHEMA)

    def setup(
        self,
        config: RunConfig,
        fleet: RacingFleet,
        rng: np.random.Generator,
        view: FleetDrawable | None,
    ) -> None:
        single_step = replace(config.ppo, epochs=1, minibatch_size=WHOLE_BATCH)
        super().setup(replace(config, ppo=single_step), fleet, rng, view)
