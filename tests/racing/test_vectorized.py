import numpy as np
import pytest

from rl_fun.racing.dynamics import KinematicBicycle, VehicleState
from rl_fun.racing.sensors import cast_rays, cast_rays_many
from rl_fun.racing.track import load_track

POINTS = np.array([[60.0, 0.0], [0.0, 35.0], [10.0, 3.0], [100.0, 100.0]])


def test_project_many_matches_scalar_projection():
    track = load_track("oval")

    progress, distance = track.project_many(POINTS)

    for index, point in enumerate(POINTS):
        expected_progress, expected_distance = track.project(point)
        assert progress[index] == pytest.approx(expected_progress)
        assert distance[index] == pytest.approx(expected_distance)


def test_contains_many_matches_scalar_check():
    track = load_track("oval")

    inside = track.contains_many(POINTS)

    assert inside.dtype == bool
    assert inside.tolist() == [track.contains(point) for point in POINTS]


def test_progress_delta_many_wraps_around_start():
    track = load_track("oval")
    length = track.length

    delta = track.progress_delta_many(
        np.array([length - 1.0, 5.0, 10.0]), np.array([1.0, length - 1.0, 20.0])
    )

    assert delta == pytest.approx([2.0, -6.0, 10.0])


def test_cast_rays_many_matches_single_origin_casts():
    track = load_track("wavy")
    origins = np.array([track.centerline[0], track.centerline[10], track.centerline[30]])
    headings = np.array([0.3, 1.0, -2.0])
    angles = np.radians([-90.0, -30.0, 0.0, 30.0, 90.0])

    many = cast_rays_many(track.boundary_segments, origins, headings, angles, 40.0)

    assert many.shape == (3, 5)
    for index in range(3):
        single = cast_rays(track.boundary_segments, origins[index], headings[index], angles, 40.0)
        assert many[index] == pytest.approx(single)


def test_step_arrays_matches_scalar_step():
    model = KinematicBicycle()
    x = np.array([0.0, 5.0, -3.0])
    y = np.array([0.0, 1.0, 2.0])
    heading = np.array([0.0, 1.0, -2.0])
    speed = np.array([0.0, 10.0, 19.9])
    steer = np.array([1.0, -0.5, 0.0])
    throttle = np.array([1.0, -1.0, 0.3])

    nx, ny, nheading, nspeed, nyaw = model.step_arrays(x, y, heading, speed, steer, throttle, 0.05)

    for i in range(3):
        after = model.step(
            VehicleState(x[i], y[i], heading[i], v_long=speed[i]), steer[i], throttle[i], 0.05
        )
        assert nx[i] == pytest.approx(after.x)
        assert ny[i] == pytest.approx(after.y)
        assert nheading[i] == pytest.approx(after.heading)
        assert nspeed[i] == pytest.approx(after.v_long)
        assert nyaw[i] == pytest.approx(after.yaw_rate)
