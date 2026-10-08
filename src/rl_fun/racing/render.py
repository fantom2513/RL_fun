"""pygame renderer for the racing environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pygame

from rl_fun.racing.background import TrackBackground, track_bounds
from rl_fun.racing.camera import (
    Camera,
    car_visual_factor,
    compute_layout,
    quantize_length,
)
from rl_fun.racing.markers import CrashMarkers
from rl_fun.racing.sprites import rotated_car_sprite
from rl_fun.racing.style import (
    CURB_RED,
    LEADER_HALO,
    PANEL_BACKGROUND,
    RAY,
    TEXT,
    TEXT_SHADOW,
)
from rl_fun.racing.track import Track

GAME_WIDTH = 900  # initial width of the game viewport (the window may be resized later)
GAME_HEIGHT = 640
MARGIN = 40
CAR_LENGTH = 4.5
CAR_WIDTH = 2.0
MIN_WINDOW = (320, 240)
HALO_RADIUS = 3.2  # m, soft glow under the leader
HALO_ALPHA = 70
LEADER_RING = (255, 196, 28)
LEADER_RING_DARK = (60, 44, 0)
CRASH_DARK = (50, 24, 30)
HINT = "Колесо/+/-: масштаб, C: камера"
KEYS_IN = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
KEYS_OUT = (pygame.K_MINUS, pygame.K_KP_MINUS)


class SidePanel(Protocol):
    """Optional widget drawn to the right of the track, for example a neural network view."""

    width: int

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw into `surface`; `rect` is the panel area in the surface's own coordinates."""


class Renderer:
    """Draw the track, car, rays and HUD; show a window or return RGB frames.

    The window (human mode) can be resized: the side panel keeps its width at the right edge and
    the game viewport takes the rest. `camera` is "fit" (whole track) or "follow" (centred on the
    car or the fleet leader, `zoom` times the fit scale). `size` is the initial window size, by
    default the game viewport size plus the panel.
    """

    def __init__(
        self,
        track: Track,
        mode: str,
        fps: int,
        side_panel: SidePanel | None = None,
        camera: str = "fit",
        zoom: float = 1.0,
        resizable: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unsupported render mode {mode!r}")
        self._track = track
        self._mode = mode
        self._fps = fps
        self._panel = side_panel
        self._resizable = resizable
        panel_width = side_panel.width if side_panel is not None else 0
        self._size = size if size is not None else (GAME_WIDTH + panel_width, GAME_HEIGHT)
        self._size = (max(self._size[0], MIN_WINDOW[0]), max(self._size[1], MIN_WINDOW[1]))
        self._layout = compute_layout(self._size, panel_width)
        self._canvas = pygame.Surface(self._size)
        pygame.font.init()
        self._font = pygame.font.Font(None, 26)
        self._hint_font = pygame.font.Font(None, 22)
        self._show_hint = mode == "human"
        if mode == "human":
            pygame.display.init()
            flags = pygame.RESIZABLE if resizable else 0
            self._window = pygame.display.set_mode(self._size, flags)
            pygame.display.set_caption("RL Fun — гонки")
            self._clock = pygame.time.Clock()
        low, high = track_bounds(track)
        self._camera = Camera(
            (low, high),
            (self._layout.viewport[2], self._layout.viewport[3]),
            margin=MARGIN,
            mode=camera,
            zoom=zoom,
        )
        self._background = TrackBackground(track)
        self._crashes = CrashMarkers()
        self._halos: dict[int, pygame.Surface] = {}
        self._markers: dict[int, pygame.Surface] = {}
        self._car_px = 8
        self._update_car_size()

    # -- public camera and window controls --------------------------------------------------

    @property
    def camera_mode(self) -> str:
        return self._camera.mode

    @property
    def zoom(self) -> float:
        return self._camera.zoom

    def set_zoom(self, zoom: float) -> None:
        """Set the zoom relative to the fit scale (0.5 to 6); it applies in follow mode."""
        self._camera.set_zoom(zoom)
        self._update_car_size()

    def toggle_camera(self) -> None:
        """Switch between showing the whole track and following the car."""
        self._camera.toggle_mode()
        self._update_car_size()

    def resize(self, size: tuple[int, int]) -> None:
        """Change the window size; the panel keeps its width, the viewport takes the rest."""
        size = (max(int(size[0]), MIN_WINDOW[0]), max(int(size[1]), MIN_WINDOW[1]))
        if size == self._size:
            return
        self._size = size
        panel_width = self._panel.width if self._panel is not None else 0
        self._layout = compute_layout(size, panel_width)
        self._canvas = pygame.Surface(size)
        self._camera.set_viewport((self._layout.viewport[2], self._layout.viewport[3]))
        self._update_car_size()
        if self._mode == "human" and pygame.display.get_surface().get_size() != size:
            flags = pygame.RESIZABLE if self._resizable else 0
            self._window = pygame.display.set_mode(size, flags)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """React to window and zoom input; True if the event was used.

        Mouse wheel, + and - change the zoom (zooming in from the fit view switches to following),
        C toggles fit/follow, and a window resize relayouts the view.
        """
        if event.type == pygame.VIDEORESIZE:
            self.resize((event.w, event.h))
            return True
        if event.type == pygame.MOUSEWHEEL and event.y != 0:
            self._zoom_step(event.y > 0)
            return True
        if event.type == pygame.KEYDOWN:
            if event.key in KEYS_IN or event.key in KEYS_OUT:
                self._zoom_step(event.key in KEYS_IN)
                return True
            if event.key == pygame.K_c:
                self.toggle_camera()
                return True
        return False

    def _zoom_step(self, zoom_in: bool) -> None:
        if self._camera.mode == "fit":
            if not zoom_in:
                return
            self._camera.toggle_mode()
        if zoom_in:
            self._camera.zoom_in()
        else:
            self._camera.zoom_out()
        self._update_car_size()

    # -- drawing ------------------------------------------------------------------------------

    def draw(
        self,
        x: float,
        y: float,
        heading: float,
        ray_points: np.ndarray,
        lines: Sequence[str],
    ) -> np.ndarray | None:
        """Draw one frame. Returns an RGB array in `rgb_array` mode, otherwise None."""
        self._begin_frame((x, y))
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
        steps: int | None = None,
        finished: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Draw many cars: crashed ones dimmed with a cross, living ones bright, the leader ringed.

        Rays are drawn for the leader only; `ray_points` has shape (n_rays, 2) in world units.
        `steps` (the fleet's step counter) lets the crash markers notice a new episode, and
        `finished` keeps cars that completed their laps from being marked as crashes.
        """
        if leader is not None:
            target = (float(x[leader]), float(y[leader]))
        else:
            target = (float(np.mean(x)), float(np.mean(y)))
        self._begin_frame(target)
        self._crashes.update(x, y, alive, steps, finished)
        if leader is not None:
            self._draw_halo((x[leader], y[leader]))
        # Crashed cars and their crosses first so living ones end up on top; leader above all.
        order = np.argsort(alive, kind="stable")
        for index in order[~alive[order]]:
            if index != leader:
                self._draw_car(x[index], y[index], heading[index], alive=False)
        self._draw_crash_markers()
        for index in order[alive[order]]:
            if index != leader:
                self._draw_car(x[index], y[index], heading[index], alive=True)
        if leader is not None:
            self._draw_car(
                x[leader], y[leader], heading[leader], alive=bool(alive[leader]), leader=True
            )
            if ray_points is not None:
                self._draw_rays((x[leader], y[leader]), ray_points)
            self._draw_leader_marker((x[leader], y[leader]))
        self._draw_hud(lines)
        return self._end_frame()

    def close(self) -> None:
        """Release pygame resources."""
        pygame.display.quit()
        pygame.font.quit()

    def _to_screen(self, point: Sequence[float]) -> tuple[float, float]:
        return self._camera.world_to_screen(point)

    def _update_car_size(self) -> None:
        scale = self._camera.scale
        self._car_px = quantize_length(CAR_LENGTH * car_visual_factor(scale) * scale)

    def _begin_frame(self, target: Sequence[float]) -> None:
        if self._mode == "human":
            actual = pygame.display.get_surface().get_size()
            if actual != self._size:
                self.resize(actual)
        self._camera.update(target, 1.0 / max(self._fps, 1))
        self._update_car_size()
        self._canvas.fill(PANEL_BACKGROUND)
        self._background.draw(self._canvas, self._camera)

    def _on_screen(self, position: tuple[float, float], margin: float) -> bool:
        width, height = self._camera.viewport
        inside_x = -margin <= position[0] <= width + margin
        return inside_x and -margin <= position[1] <= height + margin

    def _draw_rays(self, origin_world: tuple[float, float], ray_points: np.ndarray) -> None:
        origin = self._to_screen(origin_world)
        for point in ray_points:
            pygame.draw.aaline(self._canvas, RAY, origin, self._to_screen(point))

    def _draw_hud(self, lines: Sequence[str]) -> None:
        for index, text in enumerate(lines):
            position = (12, 10 + index * 24)
            self._canvas.blit(self._font.render(text, True, TEXT_SHADOW), (13, position[1] + 1))
            self._canvas.blit(self._font.render(text, True, TEXT), position)
        if self._show_hint:
            camera = self._camera
            text = HINT
            if camera.mode == "follow":
                text += f"   [слежение ×{camera.zoom:.1f}]"
            top = camera.viewport[1] - 26
            self._canvas.blit(self._hint_font.render(text, True, TEXT_SHADOW), (13, top + 1))
            self._canvas.blit(self._hint_font.render(text, True, TEXT), (12, top))

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
        if self._panel is not None and self._layout.panel[2] > 0:
            area = pygame.Rect(self._layout.panel)
            self._panel.draw(
                self._canvas.subsurface(area), pygame.Rect(0, 0, area.width, area.height)
            )

    # -- sprites and markers ------------------------------------------------------------------

    def _halo(self) -> pygame.Surface:
        radius = quantize_length(HALO_RADIUS * self._camera.scale)
        halo = self._halos.get(radius)
        if halo is None:
            halo = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
            for ring in range(radius, 0, -1):  # concentric discs build a soft falloff
                alpha = round(HALO_ALPHA * (1 - ring / radius) ** 1.5)
                pygame.draw.circle(halo, (*LEADER_HALO, alpha), (radius, radius), ring)
            if len(self._halos) > 80:
                self._halos.clear()
            self._halos[radius] = halo
        return halo

    def _draw_halo(self, center_world: tuple[float, float]) -> None:
        cx, cy = self._to_screen(center_world)
        halo = self._halo()
        self._canvas.blit(halo, (cx - halo.get_width() / 2, cy - halo.get_height() / 2))

    def _marker_sprite(self, radius: int) -> pygame.Surface:
        """Ring around the leader with a small downward pointer above it, smoothed from 4x."""
        sprite = self._markers.get(radius)
        if sprite is None:
            k = 4
            pointer = 16
            side = 2 * (radius + pointer + 8)
            big = pygame.Surface((side * k, side * k), pygame.SRCALPHA)
            c = side * k // 2
            pygame.draw.circle(big, (*LEADER_RING_DARK, 150), (c, c), (radius + 2) * k, 7 * k // 2)
            pygame.draw.circle(big, LEADER_RING, (c, c), radius * k, 3 * k)
            tip = c - (radius + 3) * k
            wing = 8 * k
            triangle = [(c, tip), (c - wing, tip - pointer * k), (c + wing, tip - pointer * k)]
            pygame.draw.polygon(big, LEADER_RING_DARK, triangle)
            inner = [(c, tip - 2 * k), (c - wing + 3 * k, tip - pointer * k + 1 * k),
                     (c + wing - 3 * k, tip - pointer * k + 1 * k)]
            pygame.draw.polygon(big, LEADER_RING, inner)
            sprite = pygame.transform.smoothscale(big, (side, side))
            if len(self._markers) > 80:
                self._markers.clear()
            self._markers[radius] = sprite
        return sprite

    def _draw_leader_marker(self, leader_world: tuple[float, float]) -> None:
        """Ring and pointer that stay the same size on screen except for following the car size."""
        radius = max(14, round(0.6 * self._car_px) + 4)
        radius = quantize_length(radius)
        sprite = self._marker_sprite(radius)
        cx, cy = self._to_screen(leader_world)
        self._canvas.blit(sprite, sprite.get_rect(center=(round(cx), round(cy))))

    def _draw_crash_markers(self) -> None:
        scale = self._camera.scale
        half = round(min(max(0.8 * scale, 4.0), 22.0))
        for point in self._crashes.points:
            cx, cy = self._to_screen(point)
            if not self._on_screen((cx, cy), half + 4):
                continue
            a = (cx - half, cy - half)
            b = (cx + half, cy + half)
            c = (cx - half, cy + half)
            d = (cx + half, cy - half)
            outline = max(3, half // 2 + 2)
            for pair in ((a, b), (c, d)):
                pygame.draw.line(self._canvas, CRASH_DARK, *pair, outline + 2)
            for pair in ((a, b), (c, d)):
                pygame.draw.line(self._canvas, CURB_RED, *pair, max(2, outline - 2))

    def _draw_car(
        self, x: float, y: float, heading: float, alive: bool = True, leader: bool = False
    ) -> None:
        center = self._to_screen((x, y))
        if not self._on_screen(center, self._car_px):
            return
        variant = "leader" if leader else ("alive" if alive else "dead")
        sprite = rotated_car_sprite(variant, self._car_px, heading)
        self._canvas.blit(sprite, sprite.get_rect(center=center))
