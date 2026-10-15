import numpy as np

from rl_fun.fourier.series import chain_positions, coefficients, partial_sum


def _square(count: int = 256) -> np.ndarray:
    side = count // 4
    ramp = np.linspace(-1.0, 1.0, side, endpoint=False)
    return np.concatenate(
        [
            ramp - 1j,
            1 + 1j * ramp,
            -ramp + 1j,
            -1 - 1j * ramp,
        ]
    )


def _mean_squared_error(points: np.ndarray, n: int) -> float:
    freqs, coeffs = coefficients(points)
    t = np.arange(len(points)) / len(points)
    return float(np.mean(np.abs(partial_sum(freqs, coeffs, n, t) - points) ** 2))


def test_coefficients_are_sorted_by_descending_magnitude():
    freqs, coeffs = coefficients(_square())

    magnitudes = np.abs(coeffs)

    assert np.all(np.diff(magnitudes) <= 1e-12)
    assert len(freqs) == len(coeffs)


def test_full_series_reproduces_the_points():
    points = _square()
    freqs, coeffs = coefficients(points)
    t = np.arange(len(points)) / len(points)

    rebuilt = partial_sum(freqs, coeffs, len(points), t)

    assert np.allclose(rebuilt, points, atol=1e-9)


def test_error_never_grows_with_more_terms():
    points = _square()

    errors = [_mean_squared_error(points, n) for n in (1, 3, 10, 50, 200)]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < errors[0] / 100


def test_pure_circle_needs_one_term():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = 2.0 * np.exp(1j * angles)

    assert _mean_squared_error(points, 1) < 1e-12


def test_chain_ends_at_the_partial_sum():
    points = _square()
    freqs, coeffs = coefficients(points)

    positions = chain_positions(freqs, coeffs, 7, 0.3)

    assert positions[0] == 0
    assert len(positions) == 8
    assert np.isclose(positions[-1], partial_sum(freqs, coeffs, 7, 0.3)[0])
