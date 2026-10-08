"""Per-step reward for reinforcement learners, built from the lab's fitness profile.

The fitness weights are the same ones evolution uses on whole episodes; here every term is turned
into a per-step amount so that the sum over an episode is comparable to the episode-level term.
The scales are module constants (see the plan, section 4); tune them here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_fun.racing.fitness import FitnessSpec
from rl_fun.racing.fleet import RacingFleet

PROGRESS_SCALE = 10.0
"""Reward per lap of progress (a lap is about 10 units)."""
STEP_TERM_SCALE = 0.1
"""Per-second amount of the speed, centering, smoothness and survival terms (times dt per step)."""
LAP_BONUS_SCALE = 5.0
"""Bonus for a lap finished with no time left over: `(1 - steps / max_steps) * scale`."""


@dataclass(frozen=True)
class StepReward:
    """Turns the state of a fleet after one step into a reward per car.

    `weights` are the fitness weights (absent terms count as zero), `dt` is the simulation step,
    `max_steps` the episode length used by the lap bonus, `crash_penalty` the amount subtracted on
    the step a car leaves the road.
    """

    weights: dict[str, float]
    dt: float
    max_steps: int
    crash_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not self.dt > 0:
            raise ValueError("dt must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not self.crash_penalty >= 0:
            raise ValueError("crash_penalty must not be negative")

    @classmethod
    def from_fitness(
        cls, fitness: FitnessSpec, dt: float, max_steps: int, crash_penalty: float
    ) -> StepReward:
        """Build from a fitness profile; its weights are copied."""
        return cls(dict(fitness.weights), float(dt), int(max_steps), float(crash_penalty))

    def compute(self, fleet: RacingFleet) -> np.ndarray:
        """Rewards of shape (n_cars,) for the step the fleet just made.

        Only cars that were moving during the step (alive, or just crashed or finished) are paid;
        the others get exactly zero.
        """
        moving = fleet.alive | fleet.just_crashed | fleet.just_finished
        step_amount = self.dt * STEP_TERM_SCALE
        half_width = fleet.track.width / 2
        weights = self.weights

        total = np.zeros(fleet.n_cars)
        if weights.get("progress"):
            total += weights["progress"] * PROGRESS_SCALE * fleet.progress_delta
        if weights.get("speed"):
            total += weights["speed"] * step_amount * fleet.v_long / fleet.dynamics.max_speed
        if weights.get("centering"):
            centering = 1.0 - np.clip(fleet.lateral_offset / half_width, 0.0, 1.0)
            total += weights["centering"] * step_amount * centering
        if weights.get("smoothness"):
            total += weights["smoothness"] * step_amount * (1.0 - fleet.steer_change / 2.0)
        if weights.get("survival"):
            total += weights["survival"] * step_amount
        if weights.get("lap_bonus"):
            lap_steps = np.nan_to_num(fleet.lap_steps, nan=float(self.max_steps))
            bonus = (1.0 - lap_steps / self.max_steps) * LAP_BONUS_SCALE
            total += weights["lap_bonus"] * np.where(fleet.just_finished, bonus, 0.0)
        total = np.where(moving, total, 0.0)
        return total - self.crash_penalty * fleet.just_crashed
