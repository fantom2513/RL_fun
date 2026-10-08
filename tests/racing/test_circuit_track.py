import numpy as np

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.style import curb_spans
from rl_fun.racing.track import _self_intersects, available_tracks, load_track

MIN_RADIUS_M = 9.0
NEAR_COLLINEAR_SIN = 1e-3


def _min_turning_radius(centerline: np.ndarray) -> float:
    """Smallest circumradius over consecutive vertex triples, skipping near-collinear ones."""
    previous = np.roll(centerline, 1, axis=0)
    following = np.roll(centerline, -1, axis=0)
    ab = centerline - previous
    bc = following - centerline
    ca = previous - following
    len_ab = np.linalg.norm(ab, axis=1)
    len_bc = np.linalg.norm(bc, axis=1)
    len_ca = np.linalg.norm(ca, axis=1)
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    sin_turn = np.abs(cross) / (len_ab * len_bc)
    keep = sin_turn >= NEAR_COLLINEAR_SIN
    radii = len_ab[keep] * len_bc[keep] * len_ca[keep] / (2 * np.abs(cross[keep]))
    return float(radii.min())


def test_circuit_track_is_built_in():
    # Arrange / Act
    track = load_track("circuit")

    # Assert
    assert "circuit" in available_tracks()
    assert track.name == "circuit"


def test_start_position_is_inside_track():
    track = load_track("circuit")

    position, _ = track.start_pose()

    assert track.contains(position)


def test_length_is_grand_prix_scale():
    track = load_track("circuit")

    assert 400.0 <= track.length <= 700.0


def test_minimum_turning_radius_allows_offset_boundaries():
    track = load_track("circuit")

    assert _min_turning_radius(track.centerline) >= MIN_RADIUS_M


def test_boundary_rings_do_not_self_intersect():
    track = load_track("circuit")

    for ring in (track.left, track.right):
        assert not _self_intersects(ring, np.roll(ring, -1, axis=0))


def test_fleet_resets_and_steps_without_error():
    fleet = RacingFleet(2, track="circuit", max_steps=50)
    actions = np.zeros((2, 2))
    actions[:, 1] = 0.5

    observation = fleet.reset()
    next_observation, rewards = fleet.step(actions)

    assert observation.shape == (2, fleet.observation_size)
    assert next_observation.shape == (2, fleet.observation_size)
    assert rewards.shape == (2,)


def test_curb_spans_cover_several_corners():
    track = load_track("circuit")

    assert len(curb_spans(track)) >= 3
