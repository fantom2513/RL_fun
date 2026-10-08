"""Static track scene, pre-rendered once at high resolution and cropped per frame for any camera."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pygame

from rl_fun.racing.camera import Camera
from rl_fun.racing.style import (
    CHEQUER_DARK,
    CHEQUER_LIGHT,
    CURB_RED,
    CURB_STRIPE,
    CURB_WHITE,
    GRASS,
    ROAD,
    ROAD_LINE,
    curb_spans,
)
from rl_fun.racing.track import Track

BACKGROUND_PPM = 20.0  # target pixels per metre of the master image
BACKGROUND_MAX_SIDE = 4096  # longest side of the master image; lower resolution for huge tracks
SUPERSAMPLE = 2  # the master is drawn this much larger, then smoothed down for clean edges
PADDING = 14.0  # m of grass kept around the track in the master image
LINE_WIDTH = 0.45  # m, white road edge lines
CURB_WIDTH = 1.8  # m, how far a curb reaches onto the road
CHEQUER_SIZE = 1.0  # m, size of one start-line square
MAX_LEVELS = 6  # master plus up to five half-size copies

_masters: dict[tuple[str, int], list[pygame.Surface]] = {}


def track_bounds(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Lower-left and upper-right corner of the track's boundary points, in metres."""
    points = np.concatenate([track.left, track.right])
    return points.min(axis=0), points.max(axis=0)


def _halve(surface: pygame.Surface) -> pygame.Surface:
    """Half-size copy by exact 2x2 averaging (smoothscale would darken flat colours)."""
    width, height = surface.get_width() // 2, surface.get_height() // 2
    result = pygame.Surface((width, height))
    source = pygame.surfarray.pixels3d(surface)
    target = pygame.surfarray.pixels3d(result)
    step = 256
    for start in range(0, width, step):
        stop = min(start + step, width)
        block = source[2 * start : 2 * stop, : 2 * height].astype(np.uint16)
        total = block[0::2, 0::2] + block[1::2, 0::2] + block[0::2, 1::2] + block[1::2, 1::2]
        target[start:stop] = ((total + 2) >> 2).astype(np.uint8)
    del source, target
    return result


class TrackBackground:
    """The grass, road, curbs, start line and edge lines of a track.

    `draw(target, camera)` crops the part of the master image the camera sees, scales it to the
    viewport and fills whatever lies outside the master with the grass colour. Half-size copies
    (mip levels) keep the scaling cheap when the camera is far away.
    """

    def __init__(self, track: Track) -> None:
        low, high = track_bounds(track)
        self.origin = (float(low[0]) - PADDING, float(high[1]) + PADDING)  # world x_min, y_max
        extent = (high - low) + 2 * PADDING
        self.ppm = min(BACKGROUND_PPM, BACKGROUND_MAX_SIDE / float(extent.max()))
        self._size = (math.ceil(extent[0] * self.ppm), math.ceil(extent[1] * self.ppm))
        key = (track.name, hash(track.left.tobytes() + track.right.tobytes()))
        if key not in _masters:
            if len(_masters) >= 2:
                _masters.clear()
            _masters[key] = [self._render_master(track)]
        self._levels = _masters[key]
        self._cache_key: tuple | None = None
        self._cache: pygame.Surface | None = None

    @property
    def size(self) -> tuple[int, int]:
        """Pixel size of the master image."""
        return self._size

    def level(self, index: int) -> pygame.Surface:
        """Master image (index 0) or its `index`-times-halved copy, built on demand."""
        index = min(max(index, 0), MAX_LEVELS - 1)
        while len(self._levels) <= index:
            self._levels.append(_halve(self._levels[-1]))
        return self._levels[index]

    def draw(self, target: pygame.Surface, camera: Camera) -> None:
        """Paint the camera's view into the top-left `camera.viewport` area of `target`."""
        width, height = camera.viewport
        clip = pygame.Rect(0, 0, width, height)
        scale = camera.scale
        level = 0
        if scale < self.ppm:
            level = min(int(math.floor(math.log2(self.ppm / scale))), MAX_LEVELS - 1)
        surface = self.level(level)
        ppm = surface.get_width() / self._size[0] * self.ppm
        x0, y0, x1, y1 = camera.visible_world_rect()
        fx0, fx1 = (x0 - self.origin[0]) * ppm, (x1 - self.origin[0]) * ppm
        fy0, fy1 = (self.origin[1] - y1) * ppm, (self.origin[1] - y0) * ppm
        ix0, iy0 = max(0, math.floor(fx0)), max(0, math.floor(fy0))
        ix1 = min(surface.get_width(), math.ceil(fx1))
        iy1 = min(surface.get_height(), math.ceil(fy1))
        if ix1 <= ix0 or iy1 <= iy0:
            target.fill(GRASS, clip)
            return
        ratio = scale / ppm
        dest_size = (max(1, round((ix1 - ix0) * ratio)), max(1, round((iy1 - iy0) * ratio)))
        key = (level, ix0, iy0, ix1, iy1, dest_size)
        if key != self._cache_key or self._cache is None:
            crop = surface.subsurface(pygame.Rect(ix0, iy0, ix1 - ix0, iy1 - iy0))
            self._cache = pygame.transform.smoothscale(crop, dest_size)
            self._cache_key = key
        # The rim of the master image is plain grass; sampling the scaled rim keeps the fill
        # outside the master the same shade as the scaled grass inside it.
        corner = (0, 0) if fx0 < 0 or fy0 < 0 else (dest_size[0] - 1, dest_size[1] - 1)
        outside = self._cache.get_at(corner)[:3] if (
            fx0 < 0 or fy0 < 0 or fx1 > surface.get_width() or fy1 > surface.get_height()
        ) else GRASS
        target.fill(outside, clip)
        previous_clip = target.get_clip()
        target.set_clip(clip)
        target.blit(self._cache, (round((ix0 - fx0) * ratio), round((iy0 - fy0) * ratio)))
        target.set_clip(previous_clip)

    # -- master image ------------------------------------------------------------------------

    def _render_master(self, track: Track) -> pygame.Surface:
        k = SUPERSAMPLE
        big = pygame.Surface((self._size[0] * k, self._size[1] * k))
        big.fill(GRASS)
        unit = self.ppm * k  # big pixels per metre
        origin_x, origin_y = self.origin

        def to_big(point: Sequence[float]) -> tuple[float, float]:
            return (point[0] - origin_x) * unit, (origin_y - point[1]) * unit

        left, right = track.left, track.right
        count = len(left)
        for i in range(count):
            j = (i + 1) % count
            quad = [to_big(left[i]), to_big(left[j]), to_big(right[j]), to_big(right[i])]
            pygame.draw.polygon(big, ROAD, quad)
            pygame.draw.polygon(big, ROAD, quad, 1)
        self._draw_curbs(big, track, to_big)
        self._draw_start_line(big, track, to_big)
        width = max(1, round(LINE_WIDTH * unit))
        for ring in (left, right):
            points = [to_big(point) for point in ring]
            pygame.draw.lines(big, ROAD_LINE, True, points, width)
            for point in points:  # round joins
                pygame.draw.circle(big, ROAD_LINE, point, width / 2)
        return _halve(big)

    @staticmethod
    def _draw_curbs(surface: pygame.Surface, track: Track, to_big) -> None:  # noqa: ANN001
        """Red and white stripes on the inside of corners, `CURB_STRIPE` metres each."""
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
                        to_big(ring[i] + (ring[j] - ring[i]) * t0),
                        to_big(ring[i] + (ring[j] - ring[i]) * t1),
                        to_big(inner[i] + (inner[j] - inner[i]) * t1),
                        to_big(inner[i] + (inner[j] - inner[i]) * t0),
                    ]
                    pygame.draw.polygon(surface, color, quad)
                    pygame.draw.polygon(surface, color, quad, 1)
                distance += length

    @staticmethod
    def _draw_start_line(surface: pygame.Surface, track: Track, to_big) -> None:  # noqa: ANN001
        """Two rows of small black and white squares across the road at the start."""
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
                quad = [to_big(p) for p in (a, b, b + forward, a + forward)]
                pygame.draw.polygon(surface, color, quad)
                pygame.draw.polygon(surface, color, quad, 1)
