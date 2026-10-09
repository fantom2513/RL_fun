"""Custom tracks of the lab: validation, the on-disk store and the API around it."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from rl_fun.lab.tracks import (
    TrackExistsError,
    delete_track,
    list_tracks,
    save_track,
)
from rl_fun.racing.track import available_tracks, load_track


def ellipse(count: int = 24, rx: float = 60.0, ry: float = 35.0) -> list[list[float]]:
    return [
        [
            round(rx * math.cos(2 * math.pi * i / count), 3),
            round(ry * math.sin(2 * math.pi * i / count), 3),
        ]
        for i in range(count)
    ]


def definition(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"name": "Моя петля", "width": 10.0, "centerline": ellipse()}
    data.update(overrides)
    return data


@pytest.fixture
def tracks_dir(tmp_path: Path) -> Path:
    return tmp_path / "tracks"


def test_a_valid_track_is_saved_listed_and_loadable(tracks_dir: Path) -> None:
    saved = save_track(tracks_dir, definition())

    assert saved["name"] == "Моя петля" and saved["builtin"] is False
    assert (tracks_dir / "Моя петля.json").is_file()
    assert "Моя петля" in available_tracks(tracks_dir)
    assert load_track("Моя петля", tracks_dir).length == pytest.approx(saved["length"])


def test_listing_puts_the_built_in_tracks_first_and_flags_the_kind(tracks_dir: Path) -> None:
    save_track(tracks_dir, definition())

    rows = list_tracks(tracks_dir)

    assert [row["name"] for row in rows[:3]] == ["circuit", "oval", "wavy"]
    assert [row["builtin"] for row in rows] == [True, True, True, False]
    assert all(row["centerline"] and row["length"] > 0 and row["width"] > 0 for row in rows)


def test_the_store_survives_a_missing_directory(tmp_path: Path) -> None:
    assert available_tracks(tmp_path / "nowhere") == ["circuit", "oval", "wavy"]


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"name": ""}, "название"),
        ({"name": "a/b"}, "название"),
        ({"name": ".."}, "название"),
        ({"name": "x" * 41}, "название"),
        ({"name": "OVAL"}, "встроенной"),
        ({"width": 3.0}, "ширина"),
        ({"width": 40.0}, "ширина"),
        ({"width": "wide"}, "ширина"),
        ({"centerline": ellipse()[:3]}, "точек"),
        ({"centerline": "no"}, "точек"),
        ({"centerline": [[0, 0], [50, 50], [50, 0], [0, 50]]}, "пересека"),
        ({"centerline": [[0, 0], [100, 0], [100, 6], [0, 6]]}, "узк"),
        ({"centerline": [[0, 0], [20, 0], [20, 20], [0, 20]]}, "коротк"),
        ({"centerline": [[0, 0], [60, 0], [60, 0], [60, 40], [0, 40]]}, "точки"),
        ({"centerline": [[0, 0], [60, 0], [60, float("nan")], [0, 40]]}, "числа"),
        ({"centerline": [[0, 0], [60, 0], [60, 5000], [0, 40]]}, "числа"),
    ],
)
def test_invalid_definitions_are_rejected_with_a_russian_reason(
    tracks_dir: Path, overrides: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        save_track(tracks_dir, definition(**overrides))

    assert not tracks_dir.exists() or not list(tracks_dir.glob("*.json"))


def test_a_name_is_taken_unless_the_save_overwrites_it(tracks_dir: Path) -> None:
    save_track(tracks_dir, definition())

    with pytest.raises(TrackExistsError):
        save_track(tracks_dir, definition())
    changed = save_track(tracks_dir, definition(width=14.0), overwrite=True)

    assert changed["width"] == 14.0
    assert load_track("Моя петля", tracks_dir).width == 14.0


def test_only_custom_tracks_can_be_deleted(tracks_dir: Path) -> None:
    save_track(tracks_dir, definition())

    delete_track(tracks_dir, "Моя петля")

    assert "Моя петля" not in available_tracks(tracks_dir)
    with pytest.raises(ValueError, match="встроенн"):
        delete_track(tracks_dir, "oval")
    with pytest.raises(KeyError):
        delete_track(tracks_dir, "Моя петля")


def test_a_saved_track_can_host_a_fleet(tracks_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rl_fun.racing.fleet import RacingFleet

    save_track(tracks_dir, definition())
    monkeypatch.setenv("RL_FUN_TRACKS_DIR", str(tracks_dir))

    fleet = RacingFleet(4, track="Моя петля", max_steps=20)

    assert fleet.track.name == "Моя петля"
