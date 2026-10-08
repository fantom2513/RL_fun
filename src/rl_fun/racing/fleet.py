"""Vectorized fleet: N cars with ray sensors racing on the same closed track.

One car behaves exactly like `RacingEnv` with the same settings; the fleet only evaluates many
cars at once. Cars that crash or finish freeze in place and stop receiving rewards.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rl_fun.racing.dynamics import make_dynamics
from rl_fun.racing.sensors import cast_rays_many
from rl_fun.racing.track import Track, load_track


class RacingFleet:
    """A batch of cars driving the same track with independent controls.

    Actions: array of shape (n_cars, 2) with [steer, throttle] in [-1, 1].
    Observation per car: normalized ray distances, then longitudinal speed, lateral speed,
    yaw rate, as in `RacingEnv`. Reward: lap progress gained this step as a fraction of the lap.
    """

    def __init__(
        self,
        n_cars: int,
        track: str = "oval",
        dynamics: str = "kinematic",
        ray_angles_deg: Sequence[float] = (-90.0, -30.0, 0.0, 30.0, 90.0),
        ray_range: float = 40.0,
        dt: float = 1 / 30,
        max_steps: int = 1500,
        laps: int = 1,
    ) -> None:
        if n_cars < 1:
            raise ValueError("n_cars must be at least 1")
        if len(ray_angles_deg) < 1:
            raise ValueError("ray_angles_deg must contain at least one angle")
        if ray_range <= 0 or dt <= 0:
            raise ValueError("ray_range and dt must be positive")
        if max_steps < 1 or laps < 1:
            raise ValueError("max_steps and laps must be at least 1")

        self.track: Track = load_track(track)
        self.dynamics = make_dynamics(dynamics)
        self.n_cars = int(n_cars)
        self.ray_range = float(ray_range)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.laps = int(laps)
        self.ray_angles = np.radians(np.asarray(ray_angles_deg, dtype=np.float64))
        self.observation_size = len(self.ray_angles) + 3

        self._low = np.concatenate([np.zeros(len(self.ray_angles)), [0.0, -1.0, -1.0]])
        self._started = False
        self.steps = 0

    def reset(self) -> np.ndarray:
        """Place every car on the start line and return observations of shape (n_cars, obs)."""
        position, heading = self.track.start_pose()
        arclength, _ = self.track.project(position)
        count = self.n_cars
        self.x = np.full(count, float(position[0]))
        self.y = np.full(count, float(position[1]))
        self.heading = np.full(count, heading)
        self.v_long = np.zeros(count)
        self.v_lat = np.zeros(count)
        self.yaw_rate = np.zeros(count)
        self.alive = np.ones(count, dtype=bool)
        self.finished = np.zeros(count, dtype=bool)
        self.steps_alive = np.zeros(count, dtype=int)
        self.lap_steps = np.full(count, np.nan)
        self._travelled = np.zeros(count)
        self._arclength = np.full(count, arclength)
        self.steps = 0
        self._started = True
        self._distances = self._cast()
        return self._observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance every living car by one step; returns (observations, rewards)."""
        if not self._started:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n_cars, 2):
            raise ValueError(f"actions must have shape ({self.n_cars}, 2), got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)

        moving = self.alive.copy()
        x, y, heading, v_long, yaw_rate = self.dynamics.step_arrays(
            self.x, self.y, self.heading, self.v_long, actions[:, 0], actions[:, 1], self.dt
        )
        self.x = np.where(moving, x, self.x)
        self.y = np.where(moving, y, self.y)
        self.heading = np.where(moving, heading, self.heading)
        self.v_long = np.where(moving, v_long, self.v_long)
        self.yaw_rate = np.where(moving, yaw_rate, self.yaw_rate)

        self.steps += 1
        self.steps_alive += moving.astype(int)
        positions = np.column_stack([self.x, self.y])
        arclength, _ = self.track.project_many(positions)
        delta = np.where(moving, self.track.progress_delta_many(self._arclength, arclength), 0.0)
        self._arclength = np.where(moving, arclength, self._arclength)
        self._travelled = self._travelled + delta
        rewards = delta / self.track.length

        on_road = self.track.contains_many(positions)
        finished_now = moving & on_road & (self._travelled >= self.laps * self.track.length)
        self.finished |= finished_now
        self.lap_steps = np.where(finished_now, float(self.steps), self.lap_steps)
        self.alive = moving & on_road & ~finished_now

        self._distances = self._cast()
        return self._observe(), rewards

    @property
    def progress(self) -> np.ndarray:
        """Laps completed per car, accumulated along the track (may be negative when reversing)."""
        return self._travelled / self.track.length

    @property
    def distances(self) -> np.ndarray:
        """Ray distances in meters per car, shape (n_cars, n_rays)."""
        return self._distances

    @property
    def done(self) -> bool:
        """True when every car has stopped or the step budget is used up."""
        if not self._started:
            return False
        return not self.alive.any() or self.steps >= self.max_steps

    def _cast(self) -> np.ndarray:
        return cast_rays_many(
            self.track.boundary_segments,
            np.column_stack([self.x, self.y]),
            self.heading,
            self.ray_angles,
            self.ray_range,
        )

    def _observe(self) -> np.ndarray:
        motion = np.stack(
            [
                self.v_long / self.dynamics.max_speed,
                self.v_lat / self.dynamics.max_speed,
                self.yaw_rate / self.dynamics.max_yaw_rate,
            ],
            axis=1,
        )
        observation = np.concatenate([self._distances / self.ray_range, motion], axis=1)
        return np.clip(observation, self._low, 1.0).astype(np.float32)
