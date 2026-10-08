"""Pure maths of the racing view (no pygame): camera, window layout and sprite size steps.

Screen coordinates are in pixels of the game viewport (origin top-left, y down); world
coordinates are metres (y up).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

MIN_ZOOM = 0.5  # relative to the "fit" scale
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25
MODES = ("fit", "follow")

Point = tuple[float, float]
IntRect = tuple[int, int, int, int]


class Camera:
    """Maps the world to a viewport either showing the whole track or following a target.

    "fit": the bounds fill the viewport (minus `margin` pixels). "follow": the viewport is centred
    on a smoothed target at `zoom` times the fit scale; `update` moves the centre towards the
    target with exponential smoothing (time constant `smoothing_time`).
    """

    def __init__(
        self,
        bounds: tuple[Sequence[float], Sequence[float]],
        viewport: tuple[int, int],
        margin: float = 40.0,
        mode: str = "fit",
        zoom: float = 1.0,
        smoothing_time: float = 0.25,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown camera mode {mode!r}")
        low, high = bounds
        self._low = (float(low[0]), float(low[1]))
        self._high = (float(high[0]), float(high[1]))
        self._margin = float(margin)
        self._smoothing_time = float(smoothing_time)
        self.mode = mode
        self.zoom = _clamp_zoom(zoom)
        self._fit_center = ((self._low[0] + self._high[0]) / 2, (self._low[1] + self._high[1]) / 2)
        self._follow_center = self._fit_center
        self._has_target = False
        self.viewport = (1, 1)
        self.fit_scale = 1.0
        self.set_viewport(viewport)

    # -- state -------------------------------------------------------------------------------

    def set_viewport(self, viewport: tuple[int, int]) -> None:
        """Change the viewport size in pixels and recompute the fit scale."""
        width, height = max(1, int(viewport[0])), max(1, int(viewport[1]))
        self.viewport = (width, height)
        span_x = max(self._high[0] - self._low[0], 1e-9)
        span_y = max(self._high[1] - self._low[1], 1e-9)
        free_x = max(width - 2 * self._margin, 1.0)
        free_y = max(height - 2 * self._margin, 1.0)
        self.fit_scale = min(free_x / span_x, free_y / span_y)

    def set_zoom(self, zoom: float) -> None:
        """Set the zoom (relative to the fit scale), clamped to [MIN_ZOOM, MAX_ZOOM]."""
        self.zoom = _clamp_zoom(zoom)

    def zoom_in(self, factor: float = ZOOM_STEP) -> None:
        self.set_zoom(self.zoom * factor)

    def zoom_out(self, factor: float = ZOOM_STEP) -> None:
        self.set_zoom(self.zoom / factor)

    def toggle_mode(self) -> None:
        self.mode = "follow" if self.mode == "fit" else "fit"

    def update(self, target: Sequence[float], dt: float) -> None:
        """Advance the follow smoothing by `dt` seconds towards `target`; no-op in fit mode."""
        if self.mode != "follow":
            return
        tx, ty = float(target[0]), float(target[1])
        if not self._has_target:
            self._follow_center = (tx, ty)
            self._has_target = True
            return
        blend = 1.0 - math.exp(-max(dt, 0.0) / self._smoothing_time)
        cx, cy = self._follow_center
        self._follow_center = (cx + (tx - cx) * blend, cy + (ty - cy) * blend)

    # -- transforms --------------------------------------------------------------------------

    @property
    def scale(self) -> float:
        """Pixels per metre."""
        return self.fit_scale if self.mode == "fit" else self.fit_scale * self.zoom

    @property
    def center(self) -> Point:
        """World point shown in the middle of the viewport."""
        return self._fit_center if self.mode == "fit" else self._follow_center

    def world_to_screen(self, point: Sequence[float]) -> Point:
        cx, cy = self.center
        scale = self.scale
        return (
            (point[0] - cx) * scale + self.viewport[0] / 2,
            self.viewport[1] / 2 - (point[1] - cy) * scale,
        )

    def screen_to_world(self, point: Sequence[float]) -> Point:
        cx, cy = self.center
        scale = self.scale
        return (
            (point[0] - self.viewport[0] / 2) / scale + cx,
            (self.viewport[1] / 2 - point[1]) / scale + cy,
        )

    def visible_world_rect(self) -> tuple[float, float, float, float]:
        """World rectangle (x_min, y_min, x_max, y_max) covered by the viewport."""
        x0, y_top = self.screen_to_world((0, 0))
        x1, y_bottom = self.screen_to_world(self.viewport)
        return x0, y_bottom, x1, y_top


def _clamp_zoom(zoom: float) -> float:
    return min(max(float(zoom), MIN_ZOOM), MAX_ZOOM)


# -- window layout -------------------------------------------------------------------------------


class Layout(NamedTuple):
    """Pixel rectangles (x, y, width, height) of the game viewport and the side panel."""

    viewport: IntRect
    panel: IntRect


def compute_layout(
    window_size: tuple[int, int], panel_width: int, min_viewport_width: int = 240
) -> Layout:
    """Side panel of fixed width at the right edge, the viewport takes the rest of the window.

    In a window too narrow for both, the panel gives way so the viewport keeps its minimum.
    """
    width, height = int(window_size[0]), int(window_size[1])
    panel = min(max(int(panel_width), 0), max(width - min_viewport_width, 0))
    viewport_width = width - panel
    return Layout((0, 0, viewport_width, height), (viewport_width, 0, panel, height))


# -- sizes -------------------------------------------------------------------------------------

LENGTH_MIN = 8
LENGTH_MAX = 320
LENGTH_RATIO = 1.06  # neighbouring sprite sizes differ by about 6 %


def _length_levels() -> list[int]:
    levels = {LENGTH_MIN, LENGTH_MAX}
    value = float(LENGTH_MIN)
    while value < LENGTH_MAX:
        levels.add(round(value))
        value *= LENGTH_RATIO
    return sorted(levels)


_LENGTH_LEVELS = _length_levels()


def quantize_length(pixels: float) -> int:
    """Snap a sprite length in pixels to one of a few dozen sizes, so sprite caches stay small."""
    target = min(max(float(pixels), LENGTH_MIN), LENGTH_MAX)
    return min(_LENGTH_LEVELS, key=lambda level: abs(math.log(level / target)))


def car_visual_factor(scale: float) -> float:
    """How much larger than real size cars are drawn: 1.6x when far away, true size up close."""
    low, high = 6.0, 14.0
    t = min(max((scale - low) / (high - low), 0.0), 1.0)
    return 1.6 + (1.0 - 1.6) * t
