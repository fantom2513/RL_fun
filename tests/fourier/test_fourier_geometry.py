import numpy as np
import pytest

from rl_fun.fourier.contour import join_subpaths, normalize, resample


def _unit_square_corners() -> np.ndarray:
    return np.array([0 + 0j, 4 + 0j, 4 + 4j, 0 + 4j])


def test_resample_returns_requested_count_with_equal_steps():
    points = resample(_unit_square_corners(), 64)

    closed = np.append(points, points[0])
    steps = np.abs(np.diff(closed))

    assert len(points) == 64
    assert np.allclose(steps, 16 / 64, atol=1e-9)


def test_resample_starts_at_the_first_point():
    points = resample(_unit_square_corners(), 32)

    assert points[0] == 0


def test_normalize_centres_and_fits_the_longer_side():
    points = np.array([10 + 0j, 30 + 0j, 30 + 10j, 10 + 10j])

    result = normalize(points)

    assert np.isclose(result.real.min(), -1) and np.isclose(result.real.max(), 1)
    assert np.isclose(result.imag.min(), -0.5) and np.isclose(result.imag.max(), 0.5)


def test_join_subpaths_visits_the_nearest_path_next():
    near = np.array([10 + 0j, 11 + 0j, 11 + 1j])
    far = np.array([100 + 0j, 101 + 0j, 101 + 1j])
    first = np.array([0 + 0j, 1 + 0j, 1 + 1j])

    joined = join_subpaths([first, far, near])

    assert len(joined) == 9
    assert np.array_equal(joined[3:6], near)
    assert np.array_equal(joined[6:], far)


def test_join_subpaths_rejects_an_empty_list():
    with pytest.raises(ValueError):
        join_subpaths([])


def test_normalize_rejects_a_zero_extent_shape():
    with pytest.raises(ValueError, match="zero extent"):
        normalize(np.array([1 + 1j] * 3))


def test_resample_rejects_fewer_than_two_points():
    with pytest.raises(ValueError, match="at least 2 points"):
        resample(np.array([1 + 1j]), 10)
