from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("contourpy")
Image = pytest.importorskip("PIL.Image")

from rl_fun.fourier.contour import contour_from_image  # noqa: E402


def _save(path: Path, gray: np.ndarray) -> Path:
    Image.fromarray(gray.astype(np.uint8)).save(path)
    return path


def test_dark_disc_on_white_gives_a_round_contour(tmp_path):
    yy, xx = np.mgrid[0:100, 0:100]
    gray = np.where((xx - 50) ** 2 + (yy - 50) ** 2 < 30**2, 0, 255)
    path = _save(tmp_path / "disc.png", gray)

    points = contour_from_image(path, 200)

    assert points.shape == (200,)
    assert np.allclose(np.abs(points), 1.0, atol=0.06)


def test_longest_contour_wins_over_small_specks(tmp_path):
    gray = np.full((100, 100), 255)
    gray[40:60, 20:80] = 0  # 60 x 20 bar
    gray[2:6, 2:6] = 0  # 4 x 4 speck
    path = _save(tmp_path / "bar.png", gray)

    points = contour_from_image(path, 200)

    assert np.isclose(points.real.max(), 1.0, atol=0.02)
    assert np.isclose(points.imag.max(), 1 / 3, atol=0.05)


def test_image_is_not_upside_down(tmp_path):
    gray = np.full((100, 100), 255)
    gray[10:30, 40:60] = 0  # blob in the upper part of the picture
    gray[30:90, 48:52] = 0  # thin tail going down
    path = _save(tmp_path / "t.png", gray)

    points = contour_from_image(path, 400)

    # after normalising, imag spans [-1, 1] and real [-0.5, 0.5]
    assert np.abs(points.real[points.imag > 0.5]).max() > 0.3  # wide blob on top
    assert np.abs(points.real[points.imag < -0.5]).max() < 0.1  # thin tail at the bottom


def test_invert_flips_which_side_is_the_shape(tmp_path):
    gray = np.full((100, 100), 0)
    gray[40:60, 20:80] = 255  # white bar on black
    path = _save(tmp_path / "inv.png", gray)

    points = contour_from_image(path, 200, invert=True)

    assert np.isclose(points.imag.max(), 1 / 3, atol=0.05)


def test_blank_image_is_rejected(tmp_path):
    path = _save(tmp_path / "blank.png", np.full((50, 50), 255))

    with pytest.raises(ValueError):
        contour_from_image(path, 100)
