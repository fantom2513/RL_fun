import numpy as np
import pytest

from rl_fun.racing.controls import map_controls
from rl_fun.racing.dynamics import KinematicBicycle


def test_default_outputs_pass_steer_and_throttle_through():
    raw = np.array([[0.5, -0.25], [-1.0, 1.0]])

    steer, throttle, boost = map_controls(("steer", "throttle"), raw)

    assert steer == pytest.approx([0.5, -1.0])
    assert throttle == pytest.approx([-0.25, 1.0])
    assert boost == pytest.approx([0.0, 0.0])


def test_accelerate_and_brake_are_rescaled_and_combined():
    raw = np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])

    _, throttle, _ = map_controls(("steer", "accelerate", "brake"), raw)

    assert throttle == pytest.approx([1.0, -1.0, 0.0])


def test_boost_is_rescaled_to_unit_interval():
    raw = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    _, _, boost = map_controls(("steer", "throttle", "boost"), raw)

    assert boost == pytest.approx([1.0, 0.0])


def test_values_are_clipped():
    steer, throttle, _ = map_controls(("steer", "throttle"), np.array([[5.0, -5.0]]))

    assert steer == pytest.approx([1.0])
    assert throttle == pytest.approx([-1.0])


def test_wrong_width_raises():
    with pytest.raises(ValueError):
        map_controls(("steer", "throttle"), np.zeros((2, 3)))


def step(boost: float, speed: float = 0.0) -> tuple[float, float]:
    model = KinematicBicycle()
    x = np.zeros(1)
    result = model.step_arrays(
        x, x, x, np.array([speed]), x, np.ones(1), 0.1, boost=np.array([boost])
    )
    return float(result[3][0]), model.max_speed


def test_boost_accelerates_faster():
    boosted, _ = step(1.0)
    plain, _ = step(0.0)

    assert boosted > plain


def test_boost_raises_top_speed():
    top = KinematicBicycle().max_speed

    boosted, _ = step(1.0, speed=top)

    assert boosted > top


def test_zero_boost_keeps_old_behaviour():
    model = KinematicBicycle()
    arrays = [np.array([1.0]), np.array([2.0]), np.array([0.3]), np.array([10.0])]
    steer, throttle = np.array([0.4]), np.array([0.7])

    with_default = model.step_arrays(*arrays, steer, throttle, 0.05)
    explicit = model.step_arrays(*arrays, steer, throttle, 0.05, boost=np.zeros(1))

    for a, b in zip(with_default, explicit, strict=True):
        assert a == pytest.approx(b)
