import numpy as np

from rl_fun.fourier.shapes import circles_as_beziers, segments_as_beziers


def _bezier_midpoints(points: np.ndarray) -> np.ndarray:
    curves = points.reshape(-1, 4, 3)
    return (curves[:, 0] + 3 * curves[:, 1] + 3 * curves[:, 2] + curves[:, 3]) / 8


def test_circle_has_sixteen_points_and_lies_on_the_circle():
    points = circles_as_beziers(np.array([2 + 1j]), np.array([3.0]))

    assert points.shape == (16, 3)
    flat = points[:, 0] + 1j * points[:, 1]
    anchors = flat.reshape(4, 4)[:, [0, 3]]
    assert np.allclose(np.abs(anchors - (2 + 1j)), 3.0)
    mid = _bezier_midpoints(points)
    assert np.allclose(np.hypot(mid[:, 0] - 2, mid[:, 1] - 1), 3.0, rtol=1e-3)
    assert np.all(points[:, 2] == 0)


def test_many_circles_are_stacked():
    points = circles_as_beziers(np.array([0j, 5 + 0j]), np.array([1.0, 2.0]))

    assert points.shape == (32, 3)


def test_no_circles_gives_an_empty_array():
    points = circles_as_beziers(np.array([], dtype=complex), np.array([]))

    assert points.shape == (0, 3)


def test_segments_are_straight_cubic_curves():
    points = segments_as_beziers(np.array([0j, 1 + 1j]), np.array([3 + 0j, 1 + 4j]))

    assert points.shape == (8, 3)
    first = points[:4, 0] + 1j * points[:4, 1]
    assert np.allclose(first, [0, 1, 2, 3])
    last = points[4:, 0] + 1j * points[4:, 1]
    assert np.allclose(last, [1 + 1j, 1 + 2j, 1 + 3j, 1 + 4j])
