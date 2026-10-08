"""pygame renderer for the racing environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pygame

from rl_fun.racing.sprites import rotated_car_sprite
from rl_fun.racing.style import (
    CHEQUER_DARK,
    CHEQUER_LIGHT,
    CURB_RED,
    CURB_STRIPE,
    CURB_WHITE,
    GRASS,
    LEADER_HALO,
    PANEL_BACKGROUND,
    RAY,
    ROAD,
    ROAD_LINE,
    TEXT,
    TEXT_SHADOW,
    curb_spans,
)
from rl_fun.racing.track import Track

GAME_WIDTH = 900
GAME_HEIGHT = 640
MARGIN = 40
CAR_LENGTH = 4.5
CAR_WIDTH = 2.0
SUPERSAMPLE = 3  # the static background is drawn this much larger, then smoothed down
LINE_WIDTH = 2.6  # px, road edge lines
CURB_WIDTH = 1.8  # m, how far a curb reaches onto the road
CHEQUER_SIZE = 1.0  # m, size of one start-line square
CAR_VISUAL_SCALE = 1.6  # cars are drawn larger than their physical size to stay readable
HALO_RADIUS = 3.2  # m, soft glow under the leader
HALO_ALPHA = 70


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
        self._car_px = max(8, round(CAR_LENGTH * CAR_VISUAL_SCALE * self._scale))
        self._halo = self._make_halo()

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
        self._draw_car(x, y, heading, alive=True)
        self._draw_rays((x, y), ray_points)
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
        """Draw many cars: crashed ones dimmed, living ones bright, the leader glowing with rays.

        Rays are drawn for the leader only; `ray_points` has shape (n_rays, 2) in world units.
        """
        self._begin_frame()
        if leader is not None:
            self._draw_halo((x[leader], y[leader]))
        # Crashed cars first so living ones end up on top of them; the leader is on top of all.
        for index in np.argsort(alive, kind="stable"):
            if index != leader:
                self._draw_car(x[index], y[index], heading[index], alive=bool(alive[index]))
        if leader is not None:
            self._draw_car(
                x[leader], y[leader], heading[leader], alive=bool(alive[leader]), leader=True
            )
            if ray_points is not None:
                self._draw_rays((x[leader], y[leader]), ray_points)
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
        """Static scene, drawn once at higher resolution and smoothed down for clean edges."""
        k = SUPERSAMPLE
        surface = pygame.Surface((GAME_WIDTH * k, GAME_HEIGHT * k))
        surface.fill(GRASS)
        left, right = self._track.left, self._track.right
        count = len(left)
        to_big = self._to_big
        for i in range(count):
            j = (i + 1) % count
            quad = [to_big(left[i]), to_big(left[j]), to_big(right[j]), to_big(right[i])]
            pygame.draw.polygon(surface, ROAD, quad)
            pygame.draw.polygon(surface, ROAD, quad, 1)
        self._draw_curbs(surface)
        self._draw_start_line(surface)
        width = max(1, round(LINE_WIDTH * k))
        for ring in (left, right):
            points = [to_big(point) for point in ring]
            pygame.draw.lines(surface, ROAD_LINE, True, points, width)
            for point in points:  # round joins
                pygame.draw.circle(surface, ROAD_LINE, point, width / 2)
        return pygame.transform.smoothscale(surface, (GAME_WIDTH, GAME_HEIGHT))

    def _to_big(self, point: Sequence[float]) -> tuple[float, float]:
        sx, sy = self._to_screen(point)
        return sx * SUPERSAMPLE, sy * SUPERSAMPLE

    def _draw_curbs(self, surface: pygame.Surface) -> None:
        """Red and white stripes on the inside of corners, `CURB_STRIPE` metres each."""
        track = self._track
        for span in curb_spans(track):
            ring = track.left if span.side == "left" else track.right
            toward_center = track.centerline - ring
            inner = ring + toward_center / np.linalg.norm(toward_center, axis=1, keepdims=True) * (
                CURB_WIDTH
            )
            distance = 0.0
            for i in range(span.start, span.stop):
                j = (i + 1) % len(ring)
                length = float(np.linalg.norm(ring[j] - ring[i]))
                pieces = max(1, math.ceil(length / (CURB_STRIPE / 2)))
                for piece in range(pieces):
                    t0, t1 = piece / pieces, (piece + 1) / pieces
                    middle = distance + (piece + 0.5) / pieces * length
                    color = CURB_RED if int(middle / CURB_STRIPE) % 2 == 0 else CURB_WHITE
                    quad = [
                        self._to_big(ring[i] + (ring[j] - ring[i]) * t0),
                        self._to_big(ring[i] + (ring[j] - ring[i]) * t1),
                        self._to_big(inner[i] + (inner[j] - inner[i]) * t1),
                        self._to_big(inner[i] + (inner[j] - inner[i]) * t0),
                    ]
                    pygame.draw.polygon(surface, color, quad)
                    pygame.draw.polygon(surface, color, quad, 1)
                distance += length

    def _draw_start_line(self, surface: pygame.Surface) -> None:
        """Two rows of small black and white squares across the road at the start."""
        track = self._track
        left, right = track.left[0], track.right[0]
        forward = track.centerline[1] - track.centerline[0]
        forward = forward / np.linalg.norm(forward) * CHEQUER_SIZE
        columns = max(1, round(float(np.linalg.norm(left - right)) / CHEQUER_SIZE))
        step = (right - left) / columns
        for row in range(2):
            origin = -forward + forward * row
            for column in range(columns):
                a = left + step * column + origin
                b = left + step * (column + 1) + origin
                color = CHEQUER_DARK if (row + column) % 2 == 0 else CHEQUER_LIGHT
                quad = [self._to_big(p) for p in (a, b, b + forward, a + forward)]
                pygame.draw.polygon(surface, color, quad)
                pygame.draw.polygon(surface, color, quad, 1)

    def _begin_frame(self) -> None:
        self._canvas.fill(PANEL_BACKGROUND)
        self._canvas.blit(self._background, (0, 0))

    def _draw_rays(self, origin_world: tuple[float, float], ray_points: np.ndarray) -> None:
        origin = self._to_screen(origin_world)
        for point in ray_points:
            pygame.draw.aaline(self._canvas, RAY, origin, self._to_screen(point))

    def _draw_hud(self, lines: Sequence[str]) -> None:
        for index, text in enumerate(lines):
            position = (12, 10 + index * 24)
            self._canvas.blit(self._font.render(text, True, TEXT_SHADOW), (13, position[1] + 1))
            self._canvas.blit(self._font.render(text, True, TEXT), position)

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

    def _make_halo(self) -> pygame.Surface:
        radius = max(2, round(HALO_RADIUS * self._scale))
        halo = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
        for ring in range(radius, 0, -1):  # concentric discs build a soft falloff
            alpha = round(HALO_ALPHA * (1 - ring / radius) ** 1.5)
            pygame.draw.circle(halo, (*LEADER_HALO, alpha), (radius, radius), ring)
        return halo

    def _draw_halo(self, center_world: tuple[float, float]) -> None:
        cx, cy = self._to_screen(center_world)
        self._canvas.blit(
            self._halo, (cx - self._halo.get_width() / 2, cy - self._halo.get_height() / 2)
        )

    def _draw_car(
        self, x: float, y: float, heading: float, alive: bool = True, leader: bool = False
    ) -> None:
        variant = "leader" if leader else ("alive" if alive else "dead")
        sprite = rotated_car_sprite(variant, self._car_px, heading)
        self._canvas.blit(sprite, sprite.get_rect(center=self._to_screen((x, y))))