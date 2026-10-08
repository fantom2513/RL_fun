import math

import numpy as np
import pytest

from rl_fun.racing.sensors import cast_rays, ray_angles
from rl_fun.racing.track import Track

WALL = np.array([[[10.0, -5.0], [10.0, 5.0]]])


def test_ray_angles_span_field_of_view():
    angles = ray_angles(3, fov_degrees=180.0)

    assert angles == pytest.approx([-math.pi / 2, 0.0, math.pi / 2])


def test_single_ray_points_forward():
    assert ray_angles(1) == pytest.approx([0.0])


def test_ray_angles_reject_empty_count():
    with pytest.raises(ValueError):
        ray_angles(0)


def test_ray_hits_wall_ahead():
    distances = cast_rays(WALL, np.array([0.0, 0.0]), 0.0, np.array([0.0]), max_range=50.0)

    assert distances == pytest.approx([10.0])


def test_ray_is_clipped_to_max_range():
    distances = cast_rays(WALL, np.array([0.0, 0.0]), 0.0, np.array([0.0]), max_range=6.0)

    assert distances == pytest.approx([6.0])


def test_parallel_and_backward_rays_miss():
    angles = np.array([math.pi / 2, math.pi])

    distances = cast_rays(WALL, np.array([0.0, 0.0]), 0.0, angles, max_range=50.0)

    assert distances == pytest.approx([50.0, 50.0])


def test_angle_is_relative_to_heading():
    distances = cast_rays(
        WALL, np.array([0.0, 0.0]), math.pi / 2, np.array([-math.pi / 2]), max_range=50.0
    )

    assert distances == pytest.approx([10.0])


def test_side_rays_see_track_edges_at_half_width():
    track = Track("square", [[0, 0], [100, 0], [100, 100], [0, 100]], width=10.0)
    angles = np.array([-math.pi / 2, math.pi / 2])

    distances = cast_rays(
        track.boundary_segments, np.array([50.0, 0.0]), 0.0, angles, max_range=40.0
    )

    assert distances == pytest.approx([5.0, 5.0], abs=1e-6)
