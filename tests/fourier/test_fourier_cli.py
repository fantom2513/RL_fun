import argparse

import pytest

from rl_fun.fourier import cli


def test_parse_stages_reads_a_comma_list():
    assert cli.parse_stages("3,10,50") == (3, 10, 50)


@pytest.mark.parametrize("bad", ["", "a,b", "0,5", "5,3", "3,3", "-1,4"])
def test_parse_stages_rejects_bad_input(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_stages(bad)


def test_defaults():
    args = cli.build_parser().parse_args(["--source", "pi"])

    assert args.stages == (3, 10, 50, 200)
    assert args.quality == "high"
    assert args.points == 2048


def test_stages_larger_than_points_exit_with_usage_error():
    with pytest.raises(SystemExit) as error:
        cli.main(["--source", "pi", "--stages", "3,500", "--points", "256"])

    assert error.value.code == 2


def test_main_builds_the_spec_and_hands_it_to_the_renderer(monkeypatch, tmp_path):
    pytest.importorskip("manim")
    captured = {}

    def fake_render(spec, out, quality):
        captured.update(spec=spec, out=out, quality=quality)
        return out

    monkeypatch.setattr(cli, "_render", fake_render)
    out = tmp_path / "pi.mp4"

    code = cli.main(["--source", "pi", "--stages", "3,10", "--quality", "low", "--out", str(out)])

    assert code == 0
    assert captured["out"] == out
    assert captured["quality"] == "low"
    assert captured["spec"].stages == (3, 10)
    assert len(captured["spec"].coeffs) == 2048


def test_raster_option_defaults():
    args = cli.build_parser().parse_args(["--source", "pi"])

    assert args.threshold == 0.5
    assert args.invert is False


def test_raster_option_flags():
    args = cli.build_parser().parse_args(["--source", "a.png", "--threshold", "0.3", "--invert"])

    assert args.threshold == 0.3
    assert args.invert is True


@pytest.mark.parametrize("bad", ["0", "1", "-0.2", "1.5", "abc"])
def test_threshold_outside_open_unit_interval_is_a_usage_error(bad):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["--source", "pi", "--threshold", bad])

    assert error.value.code == 2


def test_main_passes_raster_options_to_load_contour(monkeypatch):
    captured = {}

    def fake_load(source, count, threshold=0.5, invert=False):
        captured.update(threshold=threshold, invert=invert)
        raise ValueError("stop here")

    monkeypatch.setattr(cli, "load_contour", fake_load)

    with pytest.raises(SystemExit):
        cli.main(["--source", "x.png", "--threshold", "0.3", "--invert"])

    assert captured == {"threshold": 0.3, "invert": True}


def test_missing_animation_dependencies_are_a_usage_error(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'manim'")

    monkeypatch.setattr(cli, "load_contour", missing)

    with pytest.raises(SystemExit) as error:
        cli.main(["--source", "pi"])

    assert error.value.code == 2
    assert "optional animation dependencies" in capsys.readouterr().err


def test_missing_dependencies_at_render_time_are_a_usage_error(monkeypatch, capsys):
    import numpy as np

    monkeypatch.setattr(cli, "load_contour", lambda *a, **k: np.exp(2j * np.pi * np.arange(64) / 64))

    def missing(*args, **kwargs):
        raise ImportError("no manim")

    monkeypatch.setattr(cli, "_render", missing)

    with pytest.raises(SystemExit) as error:
        cli.main(["--source", "pi", "--stages", "3", "--points", "64"])

    assert error.value.code == 2
    assert "optional animation dependencies" in capsys.readouterr().err
