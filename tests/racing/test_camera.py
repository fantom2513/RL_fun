import math

import pytest

from rl_fun.racing.camera import (
    MAX_ZOOM,
    MIN_ZOOM,
    Camera,
    car_visual_factor,
    compute_layout,
    quantize_length,
)

BOUNDS = ((-100.0, -50.0), (100.0, 50.0))  # 200 x 100 m
VIEWPORT = (900, 640)


def make_camera(**kwargs) -> Camera:
    return Camera(BOUNDS, VIEWPORT, margin=40, **kwargs)


def test_fit_mode_centres_the_bounds_and_flips_y():
    camera = make_camera()

    cx, cy = camera.world_to_screen((0.0, 0.0))
    _, top = camera.world_to_screen((0.0, 50.0))

    assert (cx, cy) == pytest.approx((450.0, 320.0))
    assert top < cy


def test_fit_mode_keeps_the_whole_track_inside_the_margin():
    camera = make_camera()

    for corner in ((-100.0, -50.0), (100.0, 50.0), (-100.0, 50.0), (100.0, -50.0)):
        x, y = camera.world_to_screen(corner)
        assert 40 - 1e-6 <= x <= 900 - 40 + 1e-6
        assert 40 - 1e-6 <= y <= 640 - 40 + 1e-6


def test_fit_scale_is_limited_by_the_tighter_axis():
    camera = make_camera()

    assert camera.fit_scale == pytest.approx(min(820 / 200, 560 / 100))


def test_fit_round_trip():
    camera = make_camera()

    back = camera.screen_to_world(camera.world_to_screen((12.5, -7.25)))

    assert back == pytest.approx((12.5, -7.25))


def test_follow_mode_puts_the_target_in_the_viewport_centre_after_first_update():
    camera = make_camera(mode="follow", zoom=3.0)

    camera.update((30.0, 10.0), dt=1 / 30)

    assert camera.world_to_screen((30.0, 10.0)) == pytest.approx((450.0, 320.0))


def test_follow_scale_is_fit_scale_times_zoom():
    camera = make_camera(mode="follow", zoom=3.0)

    assert camera.scale == pytest.approx(3.0 * camera.fit_scale)


def test_follow_round_trip_off_centre():
    camera = make_camera(mode="follow", zoom=2.0)
    camera.update((30.0, 10.0), dt=1 / 30)

    back = camera.screen_to_world(camera.world_to_screen((41.0, 3.5)))

    assert back == pytest.approx((41.0, 3.5))


def test_zoom_is_clamped_to_limits():
    camera = make_camera(mode="follow")

    camera.set_zoom(100.0)
    assert camera.zoom == MAX_ZOOM
    camera.set_zoom(0.001)
    assert camera.zoom == MIN_ZOOM


def test_initial_zoom_is_clamped_too():
    assert make_camera(zoom=50.0).zoom == MAX_ZOOM


def test_zoom_in_then_out_returns_to_the_same_zoom():
    camera = make_camera(mode="follow", zoom=2.0)

    camera.zoom_in()
    assert camera.zoom > 2.0
    camera.zoom_out()

    assert camera.zoom == pytest.approx(2.0)


def test_zoom_in_never_exceeds_the_maximum():
    camera = make_camera(mode="follow", zoom=MAX_ZOOM)

    camera.zoom_in()

    assert camera.zoom == MAX_ZOOM


def test_fit_mode_ignores_zoom_for_the_scale():
    camera = make_camera(zoom=4.0)

    assert camera.scale == pytest.approx(camera.fit_scale)


def test_toggle_mode_switches_between_fit_and_follow():
    camera = make_camera()

    camera.toggle_mode()
    assert camera.mode == "follow"
    camera.toggle_mode()
    assert camera.mode == "fit"


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_camera(mode="orbit")


def test_follow_smoothing_moves_part_way_then_converges():
    camera = make_camera(mode="follow", zoom=2.0)
    camera.update((0.0, 0.0), dt=1 / 30)  # snaps to the first target
    target = (40.0, 20.0)

    camera.update(target, dt=1 / 30)
    first = math.dist(camera.center, target)
    for _ in range(300):
        camera.update(target, dt=1 / 30)

    assert 0 < first < math.dist((0.0, 0.0), target)
    assert math.dist(camera.center, target) < 1e-3


def test_follow_smoothing_distance_never_increases():
    camera = make_camera(mode="follow")
    camera.update((0.0, 0.0), dt=1 / 30)
    target = (-30.0, 12.0)
    distances = []

    for _ in range(40):
        camera.update(target, dt=1 / 30)
        distances.append(math.dist(camera.center, target))

    assert all(b <= a + 1e-12 for a, b in zip(distances, distances[1:], strict=False))


def test_fit_mode_update_does_not_move_the_centre():
    camera = make_camera()

    camera.update((70.0, 30.0), dt=1 / 30)

    assert camera.center == pytest.approx((0.0, 0.0))


def test_set_viewport_recomputes_the_fit_scale():
    camera = make_camera()
    before = camera.fit_scale

    camera.set_viewport((1800, 1280))

    assert camera.fit_scale > before
    assert camera.world_to_screen((0.0, 0.0)) == pytest.approx((900.0, 640.0))


def test_visible_world_rect_in_follow_mode_shrinks_with_zoom():
    camera = make_camera(mode="follow", zoom=1.0)
    camera.update((0.0, 0.0), dt=1 / 30)
    wide = camera.visible_world_rect()
    camera.set_zoom(4.0)
    narrow = camera.visible_world_rect()

    assert (narrow[2] - narrow[0]) == pytest.approx((wide[2] - wide[0]) / 4)


# -- window layout ----------------------------------------------------------------------------


def test_layout_gives_the_panel_a_fixed_width_on_the_right():
    layout = compute_layout((1600, 900), panel_width=420)

    assert layout.panel == (1180, 0, 420, 900)
    assert layout.viewport == (0, 0, 1180, 900)


def test_layout_without_panel_uses_the_whole_window():
    layout = compute_layout((800, 600), panel_width=0)

    assert layout.viewport == (0, 0, 800, 600)
    assert layout.panel[2] == 0


def test_layout_shrinks_the_panel_in_a_tiny_window_to_keep_a_viewport():
    layout = compute_layout((500, 400), panel_width=420, min_viewport_width=240)

    assert layout.viewport[2] == 240
    assert layout.panel == (240, 0, 260, 400)


@pytest.mark.parametrize("size", [(1320, 640), (800, 600), (1600, 900), (1000, 1000)])
def test_layout_rects_tile_the_window(size):
    layout = compute_layout(size, panel_width=420)

    assert layout.viewport[2] + layout.panel[2] == size[0]
    assert layout.viewport[3] == layout.panel[3] == size[1]


# -- sprite size quantisation ---------------------------------------------------------------


def test_quantize_length_is_idempotent_and_close_to_the_input():
    for value in (3, 8, 20.4, 57, 130, 250):
        q = quantize_length(value)
        assert quantize_length(q) == q
        assert abs(q - max(value, 8)) <= 0.07 * max(value, 8) + 1


def test_quantize_length_keeps_the_number_of_distinct_sizes_bounded():
    sizes = {quantize_length(value / 4) for value in range(1, 4 * 600)}

    assert len(sizes) <= 70


def test_quantize_length_is_clamped():
    assert quantize_length(0.1) == 8
    assert quantize_length(1e6) <= 320


def test_cars_shrink_towards_true_size_when_zoomed_in():
    assert car_visual_factor(3.0) == pytest.approx(1.6)
    assert car_visual_factor(30.0) == pytest.approx(1.0)
    assert 1.0 < car_visual_factor(10.0) < 1.6


def test_zoom_limits_are_half_and_six():
    assert (MIN_ZOOM, MAX_ZOOM) == (0.5, 6.0)
