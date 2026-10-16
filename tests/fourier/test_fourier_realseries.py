import numpy as np
import pytest

from rl_fun.fourier.series import RealSeries, approximation_error, real_series


def _signal(count: int) -> np.ndarray:
    t = np.arange(count) / count
    return 0.5 + np.cos(2 * np.pi * 3 * t + 0.4) - 0.7 * np.sin(2 * np.pi * 8 * t) + 0.1 * np.cos(
        2 * np.pi * 21 * t
    )


@pytest.mark.parametrize("count", [64, 65, 100])
def test_all_terms_reproduce_the_samples(count):
    values = np.random.default_rng(0).normal(size=count)
    series = real_series(values)
    t = np.arange(count) / count

    rebuilt = series.value(len(series.freqs), t)

    assert np.allclose(rebuilt, values, atol=1e-9)


def test_terms_are_sorted_by_descending_amplitude():
    series = real_series(_signal(256))

    assert np.all(np.diff(series.amplitudes) <= 1e-12)


def test_single_cosine_is_recovered():
    count = 128
    t = np.arange(count) / count
    series = real_series(2.0 + 3.0 * np.cos(2 * np.pi * 5 * t + 0.7))

    assert np.isclose(series.offset, 2.0)
    assert series.freqs[0] == 5
    assert np.isclose(series.amplitudes[0], 3.0)
    assert np.isclose(series.phases[0], 0.7)
    assert np.allclose(series.value(1, t), 2.0 + 3.0 * np.cos(2 * np.pi * 5 * t + 0.7))


def test_chain_starts_at_the_offset_and_ends_at_the_value():
    series = real_series(_signal(256))

    chain = series.chain(6, 0.37)

    assert len(chain) == 7
    assert np.isclose(chain[0], series.offset)
    assert np.isclose(chain[-1], series.value(6, 0.37)[0])


def test_error_never_grows_with_more_terms_and_vanishes_at_the_end():
    side = 64
    ramp = np.linspace(-1.0, 1.0, side, endpoint=False)
    square = np.concatenate([ramp - 1j, 1 + 1j * ramp, -ramp + 1j, -1 - 1j * ramp])
    x, y = real_series(square.real), real_series(square.imag)

    errors = [approximation_error(x, y, n, square) for n in (1, 3, 10, 50, len(x.freqs))]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:], strict=False))
    assert errors[0] > 0.1
    assert errors[-1] < 1e-9


def test_circle_needs_one_term_per_coordinate():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.exp(1j * angles)

    error = approximation_error(real_series(circle.real), real_series(circle.imag), 1, circle)

    assert error < 1e-9


def test_real_series_is_a_frozen_dataclass():
    series = real_series(_signal(64))

    assert isinstance(series, RealSeries)
    with pytest.raises(AttributeError):
        series.offset = 1.0  # type: ignore[misc]
