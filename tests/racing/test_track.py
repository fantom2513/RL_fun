import json

import pytest

from rl_fun.racing.track import Track, available_tracks, load_track

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


@pytest.fixture
def square() -> Track:
    return Track("square", SQUARE, width=10.0)


def test_length_is_perimeter(square: Track):
    assert square.length == pytest.approx(400.0)


def test_project_returns_arclength_and_distance(square: Track):
    # Act
    progress, distance = square.project((50, 3))

    # Assert
    assert progress == pytest.approx(50.0)
    assert distance == pytest.approx(3.0)


def test_project_on_second_edge(square: Track):
    progress, distance = square.project((100, 50))

    assert progress == pytest.approx(150.0)
    assert distance == pytest.approx(0.0)


def test_contains_uses_half_width(square: Track):
    assert square.contains((50, 4.9))
    assert not square.contains((50, 5.1))


def test_start_pose_points_along_first_segment(square: Track):
    position, heading = square.start_pose()

    assert tuple(position) == (0.0, 0.0)
    assert heading == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("progress_from", "progress_to", "expected"),
    [(10.0, 20.0, 10.0), (395.0, 5.0, 10.0), (5.0, 395.0, -10.0)],
)
def test_progress_delta_wraps_around_start(
    square: Track, progress_from: float, progress_to: float, expected: float
):
    assert square.progress_delta(progress_from, progress_to) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("centerline", "width"),
    [
        ([[0, 0], [1, 0]], 10.0),
        (SQUARE, 0.0),
        (SQUARE, -1.0),
        ([[0, 0], [10, 10], [10, 0], [0, 10]], 5.0),
        ([[0, 0], [0, 0], [10, 0], [10, 10]], 5.0),
    ],
)
def test_invalid_track_raises(centerline: list[list[float]], width: float):
    with pytest.raises(ValueError):
        Track("bad", centerline, width)


def test_boundary_segments_cover_both_sides(square: Track):
    assert square.boundary_segments.shape == (8, 2, 2)


def test_available_tracks_lists_defaults():
    assert {"oval", "wavy"} <= set(available_tracks())


def test_builtin_tracks_load_and_start_inside():
    for name in available_tracks():
        track = load_track(name)
        position, _ = track.start_pose()
        assert track.contains(position)


def test_builtin_boundaries_are_half_width_from_centerline():
    track = load_track("oval")

    for point in track.left:
        _, distance = track.project(point)
        assert distance == pytest.approx(track.width / 2, abs=0.5)


def test_load_track_from_json_path(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps({"name": "custom", "width": 8, "centerline": SQUARE}), encoding="utf-8"
    )

    track = load_track(str(path))

    assert track.name == "custom"
    assert track.width == 8.0


def test_unknown_track_raises():
    with pytest.raises(ValueError, match="unknown track"):
        load_track("does-not-exist")


def test_from_dict_requires_all_keys():
    with pytest.raises(ValueError, match="centerline"):
        Track.from_dict({"name": "x", "width": 5})
