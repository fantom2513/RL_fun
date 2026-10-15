import numpy as np
import pytest

pytest.importorskip("manim")

from rl_fun.fourier.contour import (  # noqa: E402
    available_presets,
    contour_from_svg,
    load_contour,
)

SQUARE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <path d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/>
</svg>"""

TWO_SQUARES = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 10">
  <path d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/>
  <path d="M 20 0 L 30 0 L 30 10 L 20 10 Z"/>
</svg>"""


def test_square_svg_lies_on_the_normalised_square(tmp_path):
    path = tmp_path / "square.svg"
    path.write_text(SQUARE)

    points = contour_from_svg(path, 400)

    assert points.shape == (400,)
    assert np.allclose(np.maximum(np.abs(points.real), np.abs(points.imag)), 1.0, atol=0.02)


def test_svg_with_two_parts_becomes_one_curve_through_both(tmp_path):
    path = tmp_path / "two.svg"
    path.write_text(TWO_SQUARES)

    points = contour_from_svg(path, 800)

    assert np.isclose(points.real.min(), -1) and np.isclose(points.real.max(), 1)
    left = np.any(points.real < -0.9)
    right = np.any(points.real > 0.9)
    assert left and right


def test_svg_is_upright(tmp_path):
    path = tmp_path / "tri.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M 5 0 L 10 10 L 0 10 Z"/></svg>'
    )

    points = contour_from_svg(path, 600)

    apex = points[np.argmax(points.imag)]
    assert abs(apex.real) < 0.1  # the apex (svg y=0) must be at the top


def test_presets_are_listed_and_load():
    names = available_presets()

    assert {"pi", "heart", "star"} <= set(names)
    for name in names:
        points = load_contour(name, 300)
        assert points.shape == (300,)
        assert np.isclose(max(points.real.max(), points.imag.max()), 1.0, atol=0.02)


def test_load_contour_dispatches_on_extension(tmp_path):
    svg = tmp_path / "square.svg"
    svg.write_text(SQUARE)

    assert load_contour(svg, 100).shape == (100,)


def test_unknown_source_is_a_clear_error():
    with pytest.raises(ValueError, match="no-such-shape"):
        load_contour("no-such-shape", 100)


def test_svg_without_paths_names_the_file(tmp_path):
    path = tmp_path / "empty.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>')

    with pytest.raises(ValueError, match="no drawable paths"):
        contour_from_svg(path, 100)
