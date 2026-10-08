"""Gymnasium environment: a car with ray sensors driving around a closed 2D track."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_fun.racing.dynamics import VehicleState, make_dynamics
from rl_fun.racing.sensors import cast_rays
from rl_fun.racing.track import load_track


class RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Drive one or more laps without touching the track boundary.

    Action: [steer, throttle] in [-1, 1]; positive steer turns left, negative throttle brakes.
    Observation: normalized ray distances, then longitudinal speed, lateral speed, yaw rate.
    Reward: lap progress gained this step as a fraction of the lap length.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        track: str = "oval",
        dynamics: str = "kinematic",
        ray_angles_deg: Sequence[float] = (-90.0, -30.0, 0.0, 30.0, 90.0),
        ray_range: float = 40.0,
        dt: float = 1 / 30,
        max_steps: int = 1500,
        laps: int = 1,
        render_mode: str | None = None,
    ) -> None:
        if len(ray_angles_deg) < 1:
            raise ValueError("ray_angles_deg must contain at least one angle")
        if ray_range <= 0 or dt <= 0:
            raise ValueError("ray_range and dt must be positive")
        if max_steps < 1 or laps < 1:
            raise ValueError("max_steps and laps must be at least 1")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.track = load_track(track)
        self.dynamics = make_dynamics(dynamics)
        self.ray_range = float(ray_range)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.laps = int(laps)
        self.render_mode = render_mode
        self.metadata = {**RacingEnv.metadata, "render_fps": round(1 / self.dt)}
        self.side_panel = None  # optional rl_fun.racing.render.SidePanel, set before first render

        self._angles = np.radians(np.asarray(ray_angles_deg, dtype=np.float64))
        n_rays = len(self._angles)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.concatenate([np.zeros(n_rays), [0.0, -1.0, -1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low, np.ones(n_rays + 3, np.float32), dtype=np.float32)

        self._state: VehicleState | None = None
        self._distances = np.zeros(n_rays)
        self._previous_arclength = 0.0
        self._progress = 0.0
        self._steps = 0
        self._done = True
        self._collided = False
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        position, heading = self.track.start_pose()
        self._state = VehicleState(float(position[0]), float(position[1]), heading)
        self._previous_arclength = self.track.project(position)[0]
        self._progress = 0.0
        self._steps = 0
        self._done = False
        self._collided = False
        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, self._info()

    def step(self, action: np.ndarray):
        if self._state is None or self._done:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        self._state = self.dynamics.step(
            self._state, float(action[0]), float(action[1]), self.dt
        )
        self._steps += 1
        arclength, _ = self.track.project((self._state.x, self._state.y))
        delta = self.track.progress_delta(self._previous_arclength, arclength)
        self._previous_arclength = arclength
        self._progress += delta
        self._collided = not self.track.contains((self._state.x, self._state.y))

        finished = self._progress >= self.laps * self.track.length
        terminated = self._collided or finished
        truncated = self._steps >= self.max_steps and not terminated
        self._done = terminated or truncated

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, delta / self.track.length, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("call reset() before render()")
        if self._renderer is None:
            from rl_fun.racing.render import Renderer

            self._renderer = Renderer(
                self.track, self.render_mode, self.metadata["render_fps"], self.side_panel
            )
        state = self._state
        directions = np.stack(
            [np.cos(state.heading + self._angles), np.sin(state.heading + self._angles)], axis=1
        )
        ray_points = np.array([state.x, state.y]) + directions * self._distances[:, None]
        lines = [
            f"Скорость: {state.v_long:4.1f} м/с",
            f"Прогресс: {self._progress / self.track.length:6.1%}",
            f"Шаг: {self._steps}",
        ]
        return self._renderer.draw(state.x, state.y, state.heading, ray_points, lines)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observe(self) -> np.ndarray:
        state = self._state
        assert state is not None
        self._distances = cast_rays(
            self.track.boundary_segments,
            np.array([state.x, state.y]),
            state.heading,
            self._angles,
            self.ray_range,
        )
        motion = np.array(
            [
                state.v_long / self.dynamics.max_speed,
                state.v_lat / self.dynamics.max_speed,
                state.yaw_rate / self.dynamics.max_yaw_rate,
            ]
        )
        observation = np.concatenate([self._distances / self.ray_range, motion])
        return np.clip(observation, self.observation_space.low, 1.0).astype(np.float32)

    def _info(self) -> dict:
        return {
            "progress": self._progress / self.track.length,
            "lap": int(max(self._progress, 0.0) // self.track.length),
            "collided": self._collided,
            "speed": 0.0 if self._state is None else self._state.v_long,
            "steps": self._steps,
        }
