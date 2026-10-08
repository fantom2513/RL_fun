import numpy as np
import pygame
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.fleet_view import FleetView, NetworkPanel
from rl_fun.racing.markers import CrashMarkers
from rl_fun.racing.render import GAME_HEIGHT, GAME_WIDTH, Renderer
from rl_fun.racing.sprites import build_car_sprite
from rl_fun.racing.style import edge_threshold, visible_edges
from rl_fun.racing.track import load_track

# -- crash marker bookkeeping -----------------------------------------------------------------


def test_new_deaths_are_recorded_once_at_their_position():
    markers = CrashMarkers()
    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    markers.update(x, y, np.array([True, True, True]), steps=1)

    markers.update(x, y, np.array([True, False, True]), steps=2)
    markers.update(x + 5, y + 5, np.array([True, False, True]), steps=3)

    assert markers.points == [(2.0, 20.0)]


def test_cars_that_are_already_dead_on_first_sight_are_recorded():
    markers = CrashMarkers()

    markers.update(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([True, False]))

    assert markers.points == [(2.0, 0.0)]


def test_finished_cars_are_not_crashes():
    markers = CrashMarkers()
    x, y = np.zeros(2), np.zeros(2)
    markers.update(x, y, np.array([True, True]), steps=1)

    markers.update(x, y, np.array([False, False]), steps=2, finished=np.array([True, False]))

    assert len(markers.points) == 1


def test_markers_reset_when_steps_return_to_zero():
    markers = CrashMarkers()
    x, y = np.zeros(2), np.zeros(2)
    markers.update(x, y, np.array([True, False]), steps=5)

    markers.update(x, y, np.array([True, True]), steps=0)

    assert markers.points == []


def test_markers_reset_when_a_car_comes_back_to_life():
    markers = CrashMarkers()
    x, y = np.zeros(2), np.zeros(2)
    markers.update(x, y, np.array([False, False]))

    markers.update(x, y, np.array([True, True]))

    assert markers.points == []


def test_markers_reset_when_the_fleet_size_changes():
    markers = CrashMarkers()
    markers.update(np.zeros(2), np.zeros(2), np.array([False, False]))

    markers.update(np.zeros(3), np.zeros(3), np.array([True, True, True]))

    assert markers.points == []


def test_marker_count_is_capped():
    markers = CrashMarkers(limit=5)
    n = 20
    markers.update(np.zeros(n), np.zeros(n), np.ones(n, dtype=bool), steps=1)

    markers.update(np.arange(n, dtype=float), np.zeros(n), np.zeros(n, dtype=bool), steps=2)

    assert len(markers.points) == 5


# -- network edge thresholds --------------------------------------------------------------------


def test_first_layer_hides_more_edges_than_later_layers():
    assert edge_threshold(0) >= 0.15
    assert edge_threshold(1) < edge_threshold(0)


def test_first_layer_threshold_hides_a_weak_edge_that_later_layers_keep():
    matrix = np.array([[1.0, 0.12]])

    assert visible_edges(matrix, edge_threshold(0)).tolist() == [[True, False]]
    assert visible_edges(matrix, edge_threshold(1)).tolist() == [[True, True]]


# -- sprites ---------------------------------------------------------------------------------


def test_dead_sprite_is_dimmer_but_still_clearly_visible():
    alive = pygame.surfarray.array_alpha(build_car_sprite("alive", 60))
    dead = pygame.surfarray.array_alpha(build_car_sprite("dead", 60))

    assert dead.max() >= 200
    assert dead.mean() >= 0.8 * alive.mean()


# -- network panel adapts to the rect ----------------------------------------------------------


def test_panel_legend_stays_at_the_bottom_of_a_short_rect():
    panel = NetworkPanel(["a"], ["x"], width=420)
    surface = pygame.Surface((420, 500))

    panel.draw(surface, surface.get_rect())

    assert (pygame.surfarray.array3d(surface)[:, 500 - 60 :, :] != 255).any()


def test_panel_draws_into_a_tall_rect_beyond_the_default_height():
    panel = NetworkPanel(["a", "b"], ["x", "y"], width=420)
    panel.update([np.ones((3, 2)), -np.ones((2, 3))], [np.ones(2), np.ones(3), np.ones(2)])
    surface = pygame.Surface((420, 1000))

    panel.draw(surface, surface.get_rect())

    pixels = pygame.surfarray.array3d(surface)
    assert (pixels[:, GAME_HEIGHT + 100 :, :] != 255).any()


def test_legend_text_fits_the_panel_width():
    panel = NetworkPanel(["a"], ["x"], width=420)

    for _kind, _y, lines in panel._legend_layout(420, 640):
        for line in lines:
            assert panel._legend_font.size(line)[0] <= 420 - 56 - 4


# -- renderer: resizing, camera, markers ---------------------------------------------------------


def renderer(**kwargs) -> Renderer:
    return Renderer(load_track("circuit"), "rgb_array", fps=30, **kwargs)


def fleet_args(n: int = 6):
    x = np.linspace(20.0, 60.0, n)
    y = np.full(n, 5.0)
    heading = np.zeros(n)
    return x, y, heading


def test_default_frame_size_is_unchanged():
    r = renderer()
    try:
        frame = r.draw(30.0, 5.0, 0.0, np.array([[40.0, 5.0]]), [])
    finally:
        r.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH, 3)


def test_explicit_size_gives_a_frame_of_that_size():
    r = renderer(size=(1600, 900))
    try:
        frame = r.draw(30.0, 5.0, 0.0, np.array([[40.0, 5.0]]), [])
    finally:
        r.close()

    assert frame.shape == (900, 1600, 3)


class SizedPanel:
    width = 300

    def __init__(self) -> None:
        self.rects: list[tuple[int, int]] = []

    def draw(self, surface, rect) -> None:
        self.rects.append(surface.get_size())
        surface.fill((255, 0, 255))


def test_resize_recomputes_viewport_and_panel_rects():
    panel = SizedPanel()
    r = renderer(side_panel=panel)
    try:
        r.resize((1400, 800))
        frame = r.draw(30.0, 5.0, 0.0, np.array([[40.0, 5.0]]), [])
    finally:
        r.close()

    assert frame.shape == (800, 1400, 3)
    assert panel.rects[-1] == (300, 800)
    assert tuple(frame[10, 1400 - 10]) == (255, 0, 255)
    assert tuple(frame[10, 1400 - 310]) != (255, 0, 255)


def test_videoresize_event_resizes_the_layout():
    r = renderer()
    try:
        consumed = r.handle_event(pygame.event.Event(pygame.VIDEORESIZE, size=(1000, 700), w=1000, h=700))
        frame = r.draw(30.0, 5.0, 0.0, np.array([[40.0, 5.0]]), [])
    finally:
        r.close()

    assert consumed
    assert frame.shape == (700, 1000, 3)


def test_unrelated_events_are_not_consumed():
    r = renderer()
    try:
        assert not r.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
    finally:
        r.close()


def test_c_key_toggles_camera_mode():
    r = renderer()
    try:
        assert r.camera_mode == "fit"
        assert r.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert r.camera_mode == "follow"
        r.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert r.camera_mode == "fit"
    finally:
        r.close()


def test_wheel_and_plus_minus_change_zoom_in_follow_mode():
    r = renderer(camera="follow", zoom=2.0)
    try:
        r.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
        zoomed_in = r.zoom
        r.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        r.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        zoomed_out = r.zoom
        r.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
    finally:
        r.close()

    assert zoomed_in > 2.0 > zoomed_out
    assert zoomed_out < r.zoom


def test_zooming_in_from_fit_mode_switches_to_follow():
    r = renderer()
    try:
        r.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
        assert r.camera_mode == "follow"
    finally:
        r.close()


def test_follow_mode_frame_differs_from_fit_and_cars_are_bigger():
    x, y, heading = fleet_args()
    alive = np.ones(len(x), dtype=bool)
    fit = renderer()
    follow = renderer(camera="follow", zoom=4.0)
    try:
        frame_fit = fit.draw_fleet(x, y, heading, alive, 3, None, [])
        frame_follow = follow.draw_fleet(x, y, heading, alive, 3, None, [])
    finally:
        fit.close()
        follow.close()

    assert not np.array_equal(frame_fit, frame_follow)
    assert follow._car_px > 2 * fit._car_px


def test_follow_camera_keeps_the_leader_in_the_middle_of_the_viewport():
    x, y, heading = fleet_args()
    alive = np.ones(len(x), dtype=bool)
    r = renderer(camera="follow", zoom=3.0)
    try:
        r.draw_fleet(x, y, heading, alive, 2, None, [])
        sx, sy = r._camera.world_to_screen((x[2], y[2]))
    finally:
        r.close()

    assert (sx, sy) == pytest.approx((GAME_WIDTH / 2, GAME_HEIGHT / 2))


def test_sprite_cache_stays_bounded_while_zooming():
    from rl_fun.racing import sprites

    x, y, heading = fleet_args()
    alive = np.ones(len(x), dtype=bool)
    r = renderer(camera="follow")
    try:
        before = len(sprites._sprites)
        for i in range(60):
            r.set_zoom(0.5 + i * 0.09)
            r.draw_fleet(x, y, heading, alive, 0, None, [])
        grown = len(sprites._sprites) - before
    finally:
        r.close()

    assert grown <= 3 * 70


def test_crash_marker_is_drawn_where_a_car_died_and_cleared_on_reset():
    x, y, heading = fleet_args(4)
    r = renderer()
    try:
        alive = np.ones(4, dtype=bool)
        r.draw_fleet(x, y, heading, alive, 0, None, [], steps=1)
        clean = r.draw_fleet(x, y, heading, alive, 0, None, [], steps=2)
        alive[3] = False
        marked = r.draw_fleet(x, y, heading, alive, 0, None, [], steps=3)
        revived = np.array([True, True, True, True])
        later = r.draw_fleet(x, y, heading, revived, 0, None, [], steps=0)
    finally:
        r.close()

    assert not np.array_equal(clean, marked)
    assert np.array_equal(clean, later)


def test_crash_marker_persists_after_the_dead_car_is_the_only_trace():
    x, y, heading = fleet_args(4)
    r = renderer()
    try:
        alive = np.ones(4, dtype=bool)
        r.draw_fleet(x, y, heading, alive, 0, None, [], steps=1)
        alive[3] = False
        r.draw_fleet(x, y, heading, alive, 0, None, [], steps=2)
        assert len(r._crashes.points) == 1
        r.draw_fleet(x, y, heading, alive, 0, None, [], steps=3)
        assert len(r._crashes.points) == 1
    finally:
        r.close()


def test_fleet_view_accepts_camera_options_and_forwards_steps():
    fleet = RacingFleet(5, track="circuit")
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=False, camera="follow", zoom=3.0, resizable=True)
    try:
        frame = view.draw(fleet, [], None)
    finally:
        view.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH, 3)
