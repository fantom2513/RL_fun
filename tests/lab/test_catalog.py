"""Catalog and track geometry served by the lab API."""

from __future__ import annotations

import json
import re

import pytest
from rl_fun.lab.catalog import build_catalog, build_track

from rl_fun.lab.config import RunConfig
from rl_fun.racing.fitness import PRESETS, available_terms
from rl_fun.racing.model_spec import available_inputs, available_outputs
from rl_fun.racing.style import CAR_COLORS, car_polygons
from rl_fun.racing.track import available_tracks, load_track

HEX = re.compile(r"^#[0-9a-f]{6}$")


@pytest.fixture(scope="module")
def catalog() -> dict:
    return build_catalog(available_tracks())


def test_catalog_has_expected_keys(catalog: dict) -> None:
    expected = {"inputs", "ray", "outputs", "activations", "fitness", "defaults", "tracks", "style"}
    assert expected <= set(catalog)


def test_catalog_lists_every_input_and_output(catalog: dict) -> None:
    input_ids = [item["id"] for item in catalog["inputs"]]
    output_ids = [item["id"] for item in catalog["outputs"]]
    assert input_ids == list(available_inputs())
    assert output_ids == list(available_outputs())
    assert all(item["label"] for item in catalog["inputs"] + catalog["outputs"])


def test_catalog_describes_ray_inputs(catalog: dict) -> None:
    ray = catalog["ray"]
    assert ray["prefix"] == "ray:"
    assert (ray["min"], ray["max"]) == (-180.0, 180.0)


def test_catalog_activations(catalog: dict) -> None:
    assert [item["id"] for item in catalog["activations"]] == ["tanh", "relu", "sigmoid"]


def test_catalog_fitness_terms_and_presets(catalog: dict) -> None:
    assert [item["id"] for item in catalog["fitness"]["terms"]] == list(available_terms())
    assert all(item["label"] for item in catalog["fitness"]["terms"])
    assert catalog["fitness"]["presets"] == {
        name: spec.to_dict() for name, spec in PRESETS.items()
    }


def test_catalog_defaults_equal_run_config_defaults(catalog: dict) -> None:
    assert catalog["defaults"] == RunConfig().to_dict()


def test_catalog_tracks_are_the_given_names() -> None:
    assert build_catalog(["oval", "wavy"])["tracks"] == ["oval", "wavy"]


def test_catalog_style_colors_are_hex(catalog: dict) -> None:
    style = catalog["style"]
    flat = [
        style[key]
        for key in (
            "grass", "road", "road_line", "curb_red", "curb_white",
            "chequer_dark", "chequer_light", "leader_body", "ray",
        )
    ]
    for states in style["car_colors"].values():
        flat += [states["alive"], states["dead"]]
    assert set(style["car_colors"]) == set(CAR_COLORS)
    assert all(HEX.match(color) for color in flat)


def test_catalog_style_car_polygons(catalog: dict) -> None:
    parts = catalog["style"]["car"]
    assert [part["name"] for part in parts] == [part.name for part in car_polygons()]
    for part in parts:
        assert len(part["points"]) >= 3
        assert all(len(point) == 2 for point in part["points"])
        assert part["color"] in CAR_COLORS


def test_catalog_is_json_serializable(catalog: dict) -> None:
    assert json.loads(json.dumps(catalog)) == catalog


def test_build_track_matches_track() -> None:
    track = load_track("oval")

    data = build_track("oval")

    assert data["name"] == "oval"
    assert data["width"] == track.width
    assert data["length"] == pytest.approx(track.length)
    assert len(data["centerline"]) == len(track.centerline)
    assert len(data["left"]) == len(track.left)
    assert len(data["right"]) == len(track.right)
    assert data["centerline"][0] == pytest.approx(track.centerline[0].tolist())


def test_build_track_curbs_are_nonempty_and_valid() -> None:
    data = build_track("oval")

    assert data["curbs"]
    for curb in data["curbs"]:
        assert curb["side"] in ("left", "right")
        assert 0 <= curb["start"] < curb["stop"] <= len(data[curb["side"]])


def test_build_track_start_is_inside_track() -> None:
    track = load_track("oval")

    start = build_track("oval")["start"]

    assert track.contains([start["x"], start["y"]])
    _, expected_heading = track.start_pose()
    assert start["heading"] == pytest.approx(expected_heading)


def test_build_track_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="nope"):
        build_track("nope")


@pytest.mark.parametrize("name", ["../racing/tracks/oval", "C:\\x.json", "", "oval.json"])
def test_build_track_rejects_paths(name: str) -> None:
    with pytest.raises(ValueError):
        build_track(name)


def test_build_track_is_json_serializable() -> None:
    data = build_track("circuit")

    assert json.loads(json.dumps(data)) == data
