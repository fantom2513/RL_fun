import numpy as np
import pygame
import pytest

from rl_fun.racing.render import GAME_HEIGHT, GAME_WIDTH, Renderer
from rl_fun.racing.track import load_track


class StubPanel:
    width = 200

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.calls.append((rect.width, rect.height))
        surface.fill((255, 0, 255))


def draw_once(renderer: Renderer) -> np.ndarray | None:
    return renderer.draw(60.0, 0.0, np.pi / 2, np.array([[60.0, 10.0]]), ["Скорость: 0"])


def test_rgb_array_frame_has_game_size():
    renderer = Renderer(load_track("oval"), "rgb_array", fps=30)
    try:
        frame = draw_once(renderer)
    finally:
        renderer.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH, 3)
    assert frame.dtype == np.uint8


def test_frame_shows_road_and_grass():
    renderer = Renderer(load_track("oval"), "rgb_array", fps=30)
    try:
        frame = draw_once(renderer)
    finally:
        renderer.close()

    assert len({tuple(pixel) for pixel in frame.reshape(-1, 3)}) > 3


def test_side_panel_extends_frame_and_receives_its_area():
    panel = StubPanel()
    renderer = Renderer(load_track("oval"), "rgb_array", fps=30, side_panel=panel)
    try:
        frame = draw_once(renderer)
    finally:
        renderer.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH + 200, 3)
    assert panel.calls == [(200, GAME_HEIGHT)]
    assert tuple(frame[10, GAME_WIDTH + 10]) == (255, 0, 255)


def test_human_mode_draw_returns_none():
    renderer = Renderer(load_track("oval"), "human", fps=1000)
    try:
        assert draw_once(renderer) is None
    finally:
        renderer.close()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        Renderer(load_track("oval"), "ansi", fps=30)
