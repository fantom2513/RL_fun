"""pygame renderer for the racing environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pygame

from rl_fun.racing.track import Track

GAME_WIDTH = 900
GAME_HEIGHT = 640
MARGIN = 40
CAR_LENGTH = 4.5
CAR_WIDTH = 2.0

GRASS = (34, 110, 50)
ROAD = (70, 70, 75)
CURB_RED = (200, 40, 40)
CURB_WHITE = (235, 235, 235)
START_LINE = (240, 240, 240)
CAR = (240, 200, 40)
CAR_NOSE = (200, 30, 30)
CAR_DEAD = (95, 88, 60)
CAR_DEAD_NOSE = (95, 45, 45)
LEADER_OUTLINE = (255, 255, 255)
RAY = (255, 255, 0)
TEXT = (255, 255, 255)
PANEL_BACKGROUND = (18, 18, 24)


class SidePanel(Protocol):
    """Optional widget drawn to the right of the track, for example a neural network view."""

    width: int

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw into `surface`; `rect` is the panel area in the surface's own coordinates."""


class Renderer:
    """Draw the track, car, rays and HUD; show a window or return RGB frames."""

    def __init__(
        self, track: Track, mode: str, fps: int, side_panel: SidePanel | None = None
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unsupported render mode {mode!r}")
        self._track = track
        self._mode = mode
        self._fps = fps
        self._panel = side_panel
        panel_width = side_panel.width if side_panel is not None else 0
        self._size = (GAME_WIDTH + panel_width, GAME_HEIGHT)
        self._canvas = pygame.Surface(self._size)
        pygame.font.init()
        self._font = pygame.font.Font(None, 26)
        if mode == "human":
            pygame.display.init()
            self._window = pygame.display.set_mode(self._size)
            pygame.display.set_caption("RL Fun — гонки")
            self._clock = pygame.time.Clock()
        self._fit_track()
        self._background = self._draw_background()

    def draw(
        self,
        x: float,
        y: float,
        heading: float,
        ray_points: np.ndarray,
        lines: Sequence[str],
    ) -> np.ndarray | None:
        """Draw one frame. Returns an RGB array in `rgb_array` mode, otherwise None."""
        self._begin_frame()
        self._draw_rays((x, y), ray_points)
        self._draw_car(x, y, heading)
        self._draw_hud(lines)
        return self._end_frame()

    def draw_fleet(
        self,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        alive: np.ndarray,
        leader: int | None,
        ray_points: np.ndarray | None,
        lines: Sequence[str],
    ) -> np.ndarray | None:
        """Draw many cars: crashed ones muted, living ones bright, the leader outlined with rays.

        Rays are drawn for the leader only; `ray_points` has shape (n_rays, 2) in world units.
        """
        self._begin_frame()
        if leader is not None:
            if ray_points is not None:
                self._draw_rays((x[leader], y[leader]), ray_points)
            self._draw_car(x[leader], y[leader], heading[leader], *_car_colors(alive[leader]))
        # Crashed cars first so living ones end up on top of them.
        for index in np.argsort(alive, kind="stable"):
            if index != leader:
                self._draw_car(x[index], y[index], heading[index], *_car_colors(alive[index]))
        if leader is not None:
            # The contour is drawn last so the leader stays marked even under overlapping cars.
            corners = self._car_corners(x[leader], y[leader], heading[leader])
            pygame.draw.polygon(self._canvas, LEADER_OUTLINE, corners, 2)
        self._draw_hud(lines)
        return self._end_frame()

    def close(self) -> None:
        """Release pygame resources."""
        pygame.display.quit()
        pygame.font.quit()

    def _fit_track(self) -> None:
        points = np.concatenate([self._track.left, self._track.right])
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high - low, 1e-9)
        self._center = (low + high) / 2
        self._scale = min((GAME_WIDTH - 2 * MARGIN) / span[0], (GAME_HEIGHT - 2 * MARGIN) / span[1])

    def _to_screen(self, point: Sequence[float]) -> tuple[float, float]:
        return (
            (point[0] - self._center[0]) * self._scale + GAME_WIDTH / 2,
            GAME_HEIGHT / 2 - (point[1] - self._center[1]) * self._scale,
        )

    def _draw_background(self) -> pygame.Surface:
        surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        surface.fill(GRASS)
        left, right = self._track.left, self._track.right
        count = len(left)
        for i in range(count):
            j = (i + 1) % count
            quad = [
                self._to_screen(left[i]),
                self._to_screen(left[j]),
                self._to_screen(right[j]),
                self._to_screen(right[i]),
            ]
            pygame.draw.polygon(surface, ROAD, quad)
            pygame.draw.polygon(surface, ROAD, quad, 1)
        for i in range(count):
            j = (i + 1) % count
            color = CURB_RED if i % 2 == 0 else CURB_WHITE
            for ring in (left, right):
                pygame.draw.line(
                    surface, color, self._to_screen(ring[i]), self._to_screen(ring[j]), 5
                )
        pygame.draw.line(
            surface, START_LINE, self._to_screen(left[0]), self._to_screen(right[0]), 4
        )
        return surface

    def _begin_frame(self) -> None:
        self._canvas.fill(PANEL_BACKGROUND)
        self._canvas.blit(self._background, (0, 0))

    def _draw_rays(self, origin_world: tuple[float, float], ray_points: np.ndarray) -> None:
        origin = self._to_screen(origin_world)
        for point in ray_points:
            end = self._to_screen(point)
            pygame.draw.line(self._canvas, RAY, origin, end, 1)
            pygame.draw.circle(self._canvas, RAY, end, 3)

    def _draw_hud(self, lines: Sequence[str]) -> None:
        for index, text in enumerate(lines):
            self._canvas.blit(self._font.render(text, True, TEXT), (12, 10 + index * 24))

    def _end_frame(self) -> np.ndarray | None:
        self._draw_side_panel()
        if self._mode == "human":
            self._window.blit(self._canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self._fps)
            return None
        return np.transpose(pygame.surfarray.array3d(self._canvas), (1, 0, 2))

    def _draw_side_panel(self) -> None:
        if self._panel is not None:
            area = pygame.Rect(GAME_WIDTH, 0, self._panel.width, GAME_HEIGHT)
            self._panel.draw(
                self._canvas.subsurface(area), pygame.Rect(0, 0, area.width, area.height)
            )

    def _car_corners(self, x: float, y: float, heading: float) -> list[tuple[float, float]]:
        """Screen-space corners of a car body; the first two are the nose edge."""
        cos, sin = math.cos(heading), math.sin(heading)
        half_length, half_width = CAR_LENGTH / 2, CAR_WIDTH / 2
        corners = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]
        return [
            self._to_screen((x + cx * cos - cy * sin, y + cx * sin + cy * cos))
            for cx, cy in corners
        ]

    def _draw_car(
        self,
        x: float,
        y: float,
        heading: float,
        body_color: tuple[int, int, int] = CAR,
        nose_color: tuple[int, int, int] = CAR_NOSE,
    ) -> None:
        body = self._car_corners(x, y, heading)
        pygame.draw.polygon(self._canvas, body_color, body)
        pygame.draw.line(self._canvas, nose_color, body[0], body[1], 3)


def _car_colors(alive: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Body and nose colors: bright for living cars, muted for crashed or finished ones."""
    if alive:
        return CAR, CAR_NOSE
    return CAR_DEAD, CAR_DEAD_NOSE
