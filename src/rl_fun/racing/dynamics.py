"""Vehicle dynamics models. A model maps (state, steer, throttle, dt) to the next state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Pose in world coordinates (meters, radians) and body-frame velocities."""

    x: float
    y: float
    heading: float
    v_long: float = 0.0
    v_lat: float = 0.0
    yaw_rate: float = 0.0


class VehicleDynamics(Protocol):
    """Interchangeable vehicle model. Steer and throttle are in [-1, 1]."""

    max_speed: float
    max_yaw_rate: float

    def step(
        self, state: VehicleState, steer: float, throttle: float, dt: float
    ) -> VehicleState:
        """Advance the vehicle by dt seconds."""

    def step_arrays(
        self,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        v_long: np.ndarray,
        steer: np.ndarray,
        throttle: np.ndarray,
        dt: float,
        boost: np.ndarray | float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized `step`: returns (x, y, heading, v_long, yaw_rate) arrays.

        `boost` in [0, 1] raises acceleration and top speed while throttling forward.
        """


BOOST_ACCELERATION_GAIN = 0.8
"""Extra forward acceleration at full boost, as a fraction of the base acceleration."""
BOOST_TOP_SPEED_GAIN = 0.3
"""Extra top speed at full boost, as a fraction of `max_speed`."""


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class KinematicBicycle:
    """Kinematic bicycle model: the car moves exactly where it points, with no sliding."""

    wheelbase: float = 2.5
    max_steer: float = 0.4
    max_speed: float = 20.0
    acceleration: float = 8.0
    braking: float = 14.0

    @property
    def max_yaw_rate(self) -> float:
        """Yaw rate at full steering lock and top speed."""
        return self.max_speed / self.wheelbase * math.tan(self.max_steer)

    def step(
        self, state: VehicleState, steer: float, throttle: float, dt: float
    ) -> VehicleState:
        """Advance the vehicle by dt seconds."""
        steer = _clip(steer, -1.0, 1.0)
        throttle = _clip(throttle, -1.0, 1.0)
        rate = throttle * self.acceleration if throttle >= 0 else throttle * self.braking
        speed = _clip(state.v_long + rate * dt, 0.0, self.max_speed)
        yaw_rate = speed / self.wheelbase * math.tan(steer * self.max_steer)
        heading = state.heading + yaw_rate * dt
        return VehicleState(
            x=state.x + speed * math.cos(heading) * dt,
            y=state.y + speed * math.sin(heading) * dt,
            heading=heading,
            v_long=speed,
            v_lat=0.0,
            yaw_rate=yaw_rate,
        )

    def step_arrays(
        self,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        v_long: np.ndarray,
        steer: np.ndarray,
        throttle: np.ndarray,
        dt: float,
        boost: np.ndarray | float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized `step` over arrays of cars: returns (x, y, heading, v_long, yaw_rate).

        Boost in [0, 1]: full boost gives 1.8x acceleration and 1.3x top speed.
        """
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 1.0)
        boost = np.clip(boost, 0.0, 1.0)
        forward_gain = 1.0 + BOOST_ACCELERATION_GAIN * boost
        top_speed = self.max_speed * (1.0 + BOOST_TOP_SPEED_GAIN * boost)
        rate = np.where(
            throttle >= 0, throttle * self.acceleration * forward_gain, throttle * self.braking
        )
        speed = np.clip(v_long + rate * dt, 0.0, top_speed)
        yaw_rate = speed / self.wheelbase * np.tan(steer * self.max_steer)
        new_heading = heading + yaw_rate * dt
        new_x = x + speed * np.cos(new_heading) * dt
        new_y = y + speed * np.sin(new_heading) * dt
        return new_x, new_y, new_heading, speed, yaw_rate


_DYNAMICS = {"kinematic": KinematicBicycle}


def make_dynamics(name: str) -> VehicleDynamics:
    """Create a dynamics model by name."""
    try:
        return _DYNAMICS[name]()
    except KeyError as error:
        supported = ", ".join(sorted(_DYNAMICS))
        raise ValueError(f"unknown dynamics {name!r}; supported: {supported}") from error
