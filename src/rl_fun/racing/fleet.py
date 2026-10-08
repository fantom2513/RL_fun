"""Vectorized fleet: N cars with ray sensors racing on the same closed track.

One car behaves exactly like `RacingEnv` with the same settings; the fleet only evaluates many
cars at once. Cars that crash or finish freeze in place and stop receiving rewards.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rl_fun.racing.controls import map_controls
from rl_fun.racing.dynamics import make_dynamics
from rl_fun.racing.model_spec import ModelSpec
from rl_fun.racing.sensors import cast_rays_many
from rl_fun.racing.track import Track, load_track

_MOTION_INPUTS = ("speed", "lateral_speed", "yaw_rate")


class RacingFleet:
    """A batch of cars driving the same track with independent controls.

    Actions: array of shape (n_cars, action_size) of raw network outputs, in the order of
    `model.outputs` (default: [steer, throttle], each in [-1, 1]).
    Observation per car: one column per entry of `model.inputs`, in that order. The default model
    gives normalized ray distances, then longitudinal speed, lateral speed and yaw rate, as in
    `RacingEnv`. Reward: lap progress gained this step as a fraction of the lap.

    Pass either `model` (a `ModelSpec`) or `ray_angles_deg` (legacy: rays plus the three motion
    inputs), not both.
    """

    def __init__(
        self,
        n_cars: int,
        track: str = "oval",
        dynamics: str = "kinematic",
        ray_angles_deg: Sequence[float] | None = None,
        ray_range: float = 40.0,
        dt: float = 1 / 30,
        max_steps: int = 1500,
        laps: int = 1,
        model: ModelSpec | None = None,
    ) -> None:
        if n_cars < 1:
            raise ValueError("n_cars must be at least 1")
        if ray_range <= 0 or dt <= 0:
            raise ValueError("ray_range and dt must be positive")
        if max_steps < 1 or laps < 1:
            raise ValueError("max_steps and laps must be at least 1")
        if model is not None and ray_angles_deg is not None:
            raise ValueError("pass either model or ray_angles_deg, not both")
        if ray_angles_deg is not None:
            if len(ray_angles_deg) < 1:
                raise ValueError("ray_angles_deg must contain at least one angle")
            rays = tuple(f"ray:{float(angle)!r}" for angle in ray_angles_deg)
            model = ModelSpec(inputs=(*rays, *_MOTION_INPUTS))
        elif model is None:
            model = ModelSpec()

        self.model = model
        self.action_size = len(model.outputs)
        self.track: Track = load_track(track)
        self.dynamics = make_dynamics(dynamics)
        self.n_cars = int(n_cars)
        self.ray_range = float(ray_range)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.laps = int(laps)
        self.ray_angles = np.radians(np.asarray(model.ray_angles_deg, dtype=np.float64))
        self.observation_size = len(model.inputs)

        ray_names = [name for name in model.inputs if name.startswith("ray:")]
        self._ray_index = {name: index for index, name in enumerate(ray_names)}
        self._low = np.array(
            [
                0.0 if name.startswith("ray:") or name == "speed" else -1.0
                for name in model.inputs
            ]
        )
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
        self._acceleration = np.zeros(count)
        self._steering = np.zeros(count)
        self._speed_total = np.zeros(count)
        self._centering_total = np.zeros(count)
        self._steer_change_total = np.zeros(count)
        self.progress_delta = np.zeros(count)
        self.lateral_offset = np.zeros(count)
        self.steer_change = np.zeros(count)
        self.just_crashed = np.zeros(count, dtype=bool)
        self.just_finished = np.zeros(count, dtype=bool)
        self.steps = 0
        self._started = True
        self._distances = self._cast()
        return self._observe()

    def respawn(self, mask: np.ndarray) -> np.ndarray:
        """Return the cars selected by the boolean `mask` (shape (n_cars,)) to the start line.

        Their state, counters, statistics and flags are reset as in `reset`; the other cars and the
        fleet step counter are left alone. Returns the observations of all cars.
        """
        if not self._started:
            raise RuntimeError("call reset() before respawn()")
        mask = np.asarray(mask)
        if mask.shape != (self.n_cars,) or mask.dtype != bool:
            raise ValueError(
                f"mask must be a boolean array of shape ({self.n_cars},), "
                f"got {mask.dtype} {mask.shape}"
            )
        if not mask.any():
            return self._observe()
        position, heading = self.track.start_pose()
        arclength, _ = self.track.project(position)
        self.x[mask] = position[0]
        self.y[mask] = position[1]
        self.heading[mask] = heading
        for array in (
            self.v_long,
            self.v_lat,
            self.yaw_rate,
            self._travelled,
            self._acceleration,
            self._steering,
            self._speed_total,
            self._centering_total,
            self._steer_change_total,
            self.progress_delta,
            self.lateral_offset,
            self.steer_change,
        ):
            array[mask] = 0.0
        self._arclength[mask] = arclength
        self.alive[mask] = True
        self.finished[mask] = False
        self.just_crashed[mask] = False
        self.just_finished[mask] = False
        self.steps_alive[mask] = 0
        self.lap_steps[mask] = np.nan
        self._distances[mask] = self._cast(mask)
        return self._observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance every living car by one step; returns (observations, rewards)."""
        if not self._started:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n_cars, self.action_size):
            raise ValueError(
                f"actions must have shape ({self.n_cars}, {self.action_size}), got {actions.shape}"
            )
        actions = np.clip(actions, -1.0, 1.0)
        steer, throttle, boost = map_controls(self.model.outputs, actions)

        moving = self.alive.copy()
        previous_speed = self.v_long
        x, y, heading, v_long, yaw_rate = self.dynamics.step_arrays(
            self.x, self.y, self.heading, self.v_long, steer, throttle, self.dt, boost=boost
        )
        self.x = np.where(moving, x, self.x)
        self.y = np.where(moving, y, self.y)
        self.heading = np.where(moving, heading, self.heading)
        self.v_long = np.where(moving, v_long, self.v_long)
        self.yaw_rate = np.where(moving, yaw_rate, self.yaw_rate)
        accelerating = (self.v_long - previous_speed) / self.dt / self.dynamics.acceleration
        self._acceleration = np.where(moving, np.clip(accelerating, -1.0, 1.0), 0.0)

        self.steps += 1
        self.steps_alive += moving.astype(int)
        positions = np.column_stack([self.x, self.y])
        arclength, offset = self.track.project_many(positions)
        delta = np.where(moving, self.track.progress_delta_many(self._arclength, arclength), 0.0)
        self._arclength = np.where(moving, arclength, self._arclength)
        self._travelled = self._travelled + delta
        rewards = delta / self.track.length
        self.progress_delta = rewards

        centering = 1.0 - np.clip(offset / (self.track.width / 2), 0.0, 1.0)
        raw_change = np.abs(steer - self._steering)
        self._speed_total += np.where(moving, self.v_long / self.dynamics.max_speed, 0.0)
        self._centering_total += np.where(moving, centering, 0.0)
        self._steer_change_total += np.where(moving, raw_change / 2.0, 0.0)
        self._steering = np.where(moving, steer, self._steering)

        on_road = offset <= self.track.width / 2
        finished_now = moving & on_road & (self._travelled >= self.laps * self.track.length)
        self.just_crashed = moving & ~on_road
        self.just_finished = finished_now
        self.lateral_offset = offset
        self.steer_change = np.where(moving, raw_change, 0.0)
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
    def mean_speed(self) -> np.ndarray:
        """Mean speed over living steps as a fraction of max_speed, per car (0 before any step)."""
        return self._per_living_step(self._speed_total)

    @property
    def centering(self) -> np.ndarray:
        """Mean centering over living steps in [0, 1]: 1 on the centerline, 0 at the edge."""
        return self._per_living_step(self._centering_total)

    @property
    def smoothness(self) -> np.ndarray:
        """1 minus the mean steer change per living step (|change| / 2); 0 before any step."""
        mean_change = self._per_living_step(self._steer_change_total)
        return np.where(self.steps_alive > 0, 1.0 - mean_change, 0.0)

    @property
    def done(self) -> bool:
        """True when every car has stopped or the step budget is used up."""
        if not self._started:
            return False
        return not self.alive.any() or self.steps >= self.max_steps

    def _per_living_step(self, total: np.ndarray) -> np.ndarray:
        counts = self.steps_alive
        return np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)

    def _cast(self, mask: np.ndarray | None = None) -> np.ndarray:
        selected = slice(None) if mask is None else mask
        count = self.n_cars if mask is None else int(np.count_nonzero(mask))
        if self.ray_angles.size == 0:
            return np.zeros((count, 0))
        return cast_rays_many(
            self.track.boundary_segments,
            np.column_stack([self.x[selected], self.y[selected]]),
            self.heading[selected],
            self.ray_angles,
            self.ray_range,
        )

    def _observe(self) -> np.ndarray:
        rays = self._distances / self.ray_range
        scalars = {
            "speed": self.v_long / self.dynamics.max_speed,
            "lateral_speed": self.v_lat / self.dynamics.max_speed,
            "yaw_rate": self.yaw_rate / self.dynamics.max_yaw_rate,
            "acceleration": self._acceleration,
            "steering_angle": self._steering,
        }
        columns = [
            rays[:, self._ray_index[name]] if name in self._ray_index else scalars[name]
            for name in self.model.inputs
        ]
        observation = np.column_stack(columns)
        return np.clip(observation, self._low, 1.0).astype(np.float32)
