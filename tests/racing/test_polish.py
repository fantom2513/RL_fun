import numpy as np
import pygame
import pytest

from rl_fun.racing.fleet_view import NetworkPanel
from rl_fun.racing.render import GAME_HEIGHT
from rl_fun.racing.sprites import CAR_VARIANTS, build_car_sprite, rotated_car_sprite
from rl_fun.racing.style import LEGEND_ENTRIES, network_column_titles, visible_edges

LENGTH = 60


def alpha_of(surface: pygame.Surface) -> np.ndarray:
    return pygame.surfarray.array_alpha(surface)  # shape (width, height)


def test_sprite_is_translucent_surface_with_visible_pixels():
    sprite = build_car_sprite("alive", LENGTH)

    assert sprite.get_flags() & pygame.SRCALPHA
    assert alpha_of(sprite).max() > 200


def test_sprite_is_wider_than_tall_with_nose_on_the_right():
    sprite = build_car_sprite("alive", LENGTH)
    columns = np.flatnonzero((alpha_of(sprite) > 128).any(axis=1))

    assert sprite.get_width() > sprite.get_height()
    assert abs(columns.max() - (sprite.get_width() / 2 + LENGTH / 2)) <= 2
    assert abs(sprite.get_width() / 2 - LENGTH / 2 - columns.min()) <= 2


def test_nose_is_narrower_than_the_tail():
    alpha = alpha_of(build_car_sprite("alive", LENGTH)) > 128
    columns = np.flatnonzero(alpha.any(axis=1))

    nose = alpha[columns.max() - 1].sum()
    tail = alpha[columns.min() + 2].sum()

    assert nose < tail


def test_dead_sprite_is_fainter_than_alive_one():
    alive = alpha_of(build_car_sprite("alive", LENGTH)).mean()
    dead = alpha_of(build_car_sprite("dead", LENGTH)).mean()

    assert dead < alive


def test_leader_body_is_brighter_than_alive_body():
    def brightness(variant: str) -> float:
        sprite = build_car_sprite(variant, LENGTH)
        mask = alpha_of(sprite) > 250
        return float(pygame.surfarray.array3d(sprite)[mask].mean())

    assert brightness("leader") > brightness("alive")


def test_same_arguments_return_the_cached_surface():
    assert build_car_sprite("alive", LENGTH) is build_car_sprite("alive", LENGTH)
    assert build_car_sprite("alive", LENGTH) is not build_car_sprite("dead", LENGTH)


def test_longer_car_gives_larger_sprite():
    assert build_car_sprite("alive", 80).get_width() > build_car_sprite("alive", 40).get_width()


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_car_sprite("purple", LENGTH)


def test_all_variants_build():
    for variant in CAR_VARIANTS:
        assert build_car_sprite(variant, 30).get_width() > 0


def test_rotated_sprite_is_cached_per_three_degree_bucket():
    first = rotated_car_sprite("alive", LENGTH, np.radians(9.5))
    same_bucket = rotated_car_sprite("alive", LENGTH, np.radians(10.4))
    other = rotated_car_sprite("alive", LENGTH, np.radians(40.0))

    assert first is same_bucket
    assert first is not other


def test_rotated_sprite_points_up_for_heading_ninety_degrees():
    sprite = rotated_car_sprite("alive", LENGTH, np.pi / 2)

    assert sprite.get_height() > sprite.get_width()


def test_column_titles_for_four_layers():
    assert network_column_titles(4) == ["Вход", "Скрытый 1", "Скрытый 2", "Выход"]


def test_column_titles_for_three_layers():
    assert network_column_titles(3) == ["Вход", "Скрытый 1", "Выход"]


def test_column_titles_for_two_layers():
    assert network_column_titles(2) == ["Вход", "Выход"]


def test_visible_edges_hides_weak_weights_and_keeps_the_largest():
    matrix = np.array([[1.0, 0.05], [-0.5, -0.07]])

    mask = visible_edges(matrix, 0.08)

    assert mask.tolist() == [[True, False], [True, False]]


def test_visible_edges_of_all_zero_matrix_is_empty():
    assert not visible_edges(np.zeros((2, 3)), 0.08).any()


def test_legend_has_a_few_russian_entries():
    assert 3 <= len(LEGEND_ENTRIES) <= 4
    for _kind, text in LEGEND_ENTRIES:
        assert any("а" <= char.lower() <= "я" for char in text)


def test_panel_draws_the_legend_even_without_a_network():
    panel = NetworkPanel(["a"], ["x"], width=420)
    surface = pygame.Surface((420, GAME_HEIGHT))

    panel.draw(surface, surface.get_rect())

    bottom = pygame.surfarray.array3d(surface)[:, GAME_HEIGHT - 120 :, :]
    assert (bottom != 255).any()
