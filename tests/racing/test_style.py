import numpy as np
import pytest

from rl_fun.racing.style import CurbSpan, car_polygons, curb_spans
from rl_fun.racing.track import Track, load_track


def _on_circle(center: tuple[float, float], radius: float, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])


def rounded_rectangle(clockwise: bool = False) -> Track:
    """Counter-clockwise track: 10 m straight segments joined by quarter circles of radius 20 m."""
    radius = 20.0
    centers = [(40.0, -20.0), (40.0, 20.0), (-40.0, 20.0), (-40.0, -20.0)]
    points: list[list[float]] = []
    for index, center in enumerate(centers):
        first = -90.0 + 90.0 * index
        points += [list(_on_circle(center, radius, first + 15.0 * step)) for step in range(6)]
        start = _on_circle(center, radius, first + 90.0)
        end = _on_circle(centers[(index + 1) % 4], radius, first + 90.0)
        count = round(float(np.linalg.norm(end - start)) / 10.0)
        points += [list(start + (end - start) * step / count) for step in range(count)]
    if clockwise:
        points = points[::-1]
    return Track("rounded", points, width=10.0)


def segment_midpoints(track: Track, span: CurbSpan) -> np.ndarray:
    ring = track.left if span.side == "left" else track.right
    ends = np.roll(ring, -1, axis=0)
    return ((ring + ends) / 2)[span.start : span.stop]


def test_car_polygons_are_non_empty_polygons():
    parts = car_polygons()

    assert parts
    for part in parts:
        assert len(part.points) >= 3


def test_car_has_four_wheels():
    wheels = [part for part in car_polygons() if part.name.startswith("wheel")]

    assert len(wheels) == 4


def test_car_has_body_cockpit_and_wings():
    names = {part.name for part in car_polygons()}

    assert {"body", "cockpit", "front_wing", "rear_wing"} <= names


def test_every_part_has_a_color_key():
    assert all(part.color_key for part in car_polygons())


def test_nose_is_the_max_x_point_of_the_whole_car():
    points = np.concatenate([np.asarray(part.points) for part in car_polygons()])
    body = np.asarray(next(part for part in car_polygons() if part.name == "body").points)

    assert body[:, 0].max() == pytest.approx(points[:, 0].max())


def test_car_length_is_about_four_and_a_half_metres():
    points = np.concatenate([np.asarray(part.points) for part in car_polygons()])

    assert np.ptp(points[:, 0]) == pytest.approx(4.5, abs=0.3)


def test_body_is_symmetric_about_the_x_axis():
    body = np.asarray(next(part for part in car_polygons() if part.name == "body").points)
    mirrored = body * np.array([1.0, -1.0])

    as_set = {tuple(np.round(p, 6)) for p in body}
    assert {tuple(np.round(p, 6)) for p in mirrored} == as_set


def test_wheels_sit_on_both_sides_and_both_axles():
    centers = np.array(
        [np.mean(part.points, axis=0) for part in car_polygons() if part.name.startswith("wheel")]
    )

    assert (centers[:, 1] > 0).sum() == 2
    assert (centers[:, 1] < 0).sum() == 2
    assert (centers[:, 0] > 0).sum() == 2


def test_no_curbs_on_straight_edges():
    track = rounded_rectangle()
    spans = curb_spans(track)

    assert spans
    for span in spans:
        for x, y in segment_midpoints(track, span):
            # Inner (left) boundary of the straights: y = +-35 for |x| <= 40, x = +-55 for |y| <= 20.
            assert not (abs(abs(y) - 35.0) < 1.0 and abs(x) <= 35.0)
            assert not (abs(abs(x) - 55.0) < 1.0 and abs(y) <= 15.0)


def test_curbs_exist_on_every_corner_of_a_ccw_track():
    track = rounded_rectangle()
    spans = curb_spans(track)
    midpoints = np.concatenate([segment_midpoints(track, span) for span in spans])

    for corner in ([40, 20], [40, -20], [-40, 20], [-40, -20]):
        assert np.min(np.linalg.norm(midpoints - np.array(corner), axis=1)) < 40.0


def test_ccw_track_has_curbs_on_the_left_boundary_only():
    spans = curb_spans(rounded_rectangle())

    assert spans
    assert {span.side for span in spans} == {"left"}


def test_cw_track_has_curbs_on_the_right_boundary_only():
    spans = curb_spans(rounded_rectangle(clockwise=True))

    assert spans
    assert {span.side for span in spans} == {"right"}


def test_spans_stay_inside_the_boundary_index_range():
    track = load_track("wavy")

    for span in curb_spans(track):
        assert 0 <= span.start < span.stop <= len(track.left)


def test_spans_do_not_overlap_per_side():
    track = load_track("wavy")
    covered: set[tuple[str, int]] = set()

    for span in curb_spans(track):
        indices = {(span.side, i) for i in range(span.start, span.stop)}
        assert not indices & covered
        covered |= indices


def test_oval_has_curbs_at_both_ends():
    track = load_track("oval")
    spans = curb_spans(track)
    midpoints = np.concatenate([segment_midpoints(track, span) for span in spans])

    assert (midpoints[:, 0] > 30).any()
    assert (midpoints[:, 0] < -30).any()


def test_wavy_track_has_curbs_on_both_sides():
    assert {span.side for span in curb_spans(load_track("wavy"))} == {"left", "right"}
