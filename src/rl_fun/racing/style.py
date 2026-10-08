"""Palette and pure geometry for the racing visuals (no pygame): car silhouette and curb spans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_fun.racing.track import Track

Color = tuple[int, int, int]

# Scene.
GRASS: Color = (122, 142, 156)
ROAD: Color = (79, 97, 110)
ROAD_LINE: Color = (255, 255, 252)
CURB_RED: Color = (214, 46, 62)
CURB_WHITE: Color = (250, 250, 248)
CHEQUER_DARK: Color = (24, 26, 32)
CHEQUER_LIGHT: Color = (250, 250, 248)

# Cars: color key -> (alive, dead).
CAR_COLORS: dict[str, tuple[Color, Color]] = {
    "body": ((226, 230, 235), (148, 160, 172)),
    "accent": ((186, 193, 202), (128, 140, 152)),
    "wing": ((60, 66, 76), (66, 78, 92)),
    "wheel": ((18, 20, 26), (34, 42, 52)),
    "cockpit": ((40, 46, 58), (70, 82, 96)),
}
LEADER_BODY: Color = (255, 255, 255)
LEADER_HALO: Color = (255, 255, 255)
RAY: Color = (255, 255, 255)
TEXT: Color = (245, 248, 250)
TEXT_SHADOW: Color = (40, 52, 62)

# Neural network panel (light, like the reference video).
PANEL_BACKGROUND: Color = (255, 255, 255)
PANEL_TEXT: Color = (60, 66, 76)
PANEL_MUTED: Color = (120, 128, 138)
NODE_OUTLINE: Color = (60, 74, 82)
NODE_NEUTRAL: Color = (236, 238, 240)
NODE_POSITIVE: Color = (90, 255, 80)
NODE_NEGATIVE: Color = (250, 62, 82)
EDGE_POSITIVE: Color = (40, 200, 60)
EDGE_NEGATIVE: Color = (214, 50, 70)

CURB_MIN_CURVATURE = 0.02  # rad/m; tighter turns than this (radius < 50 m) get curbs
CURB_STRIPE = 2.0  # metres per red or white stripe


@dataclass(frozen=True)
class CarPart:
    """One polygon of the car in the car frame: metres, nose along +x, y to the left."""

    name: str
    points: tuple[tuple[float, float], ...]
    color_key: str


@dataclass(frozen=True)
class CurbSpan:
    """Consecutive boundary segments `start..stop-1` of one boundary ring that get curbs."""

    side: str  # "left" or "right"
    start: int
    stop: int


def _mirrored(half: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Close a half outline (nose to tail, y >= 0) into a polygon symmetric about the x axis."""
    return tuple(half) + tuple((x, -y) for x, y in reversed(half) if y != 0.0)


def _box(x0: float, x1: float, y0: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x1, y1), (x1, y0), (x0, y0), (x0, y1))


def car_polygons() -> list[CarPart]:
    """F1-style silhouette, drawn back to front: wheels, wings, body, cockpit. Length 4.5 m."""
    body = _mirrored(
        [
            (2.25, 0.0),
            (1.75, 0.10),
            (0.95, 0.17),
            (0.75, 0.50),
            (-0.10, 0.52),
            (-0.35, 0.34),
            (-1.15, 0.26),
            (-1.85, 0.30),
        ]
        + [(-1.85, 0.0)]
    )
    wheels = [
        CarPart("wheel_front_left", _box(0.80, 1.45, 0.62, 1.00), "wheel"),
        CarPart("wheel_front_right", _box(0.80, 1.45, -1.00, -0.62), "wheel"),
        CarPart("wheel_rear_left", _box(-1.75, -0.95, 0.64, 1.05), "wheel"),
        CarPart("wheel_rear_right", _box(-1.75, -0.95, -1.05, -0.64), "wheel"),
    ]
    return [
        *wheels,
        CarPart("front_wing", _box(1.55, 2.00, -1.00, 1.00), "wing"),
        CarPart("rear_wing", _box(-2.25, -1.80, -0.88, 0.88), "wing"),
        CarPart("body", body, "body"),
        CarPart(
            "cockpit",
            ((0.45, 0.0), (0.30, 0.17), (-0.25, 0.20), (-0.35, 0.0), (-0.25, -0.20), (0.30, -0.17)),
            "cockpit",
        ),
    ]


def curb_spans(track: Track, min_curvature: float = CURB_MIN_CURVATURE) -> list[CurbSpan]:
    """Boundary segments on the inside of noticeable turns (left ring for left turns).

    Indices refer to segments of `track.left` / `track.right` (segment `i` joins vertex `i` to
    `i + 1`). A span never crosses the wrap-around of the ring; it is split in two instead.
    """
    points = track.centerline
    vectors = np.roll(points, -1, axis=0) - points
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    turn = (headings - np.roll(headings, 1) + np.pi) % (2 * np.pi) - np.pi  # turn at each vertex
    lengths = np.linalg.norm(vectors, axis=1)
    curvature = (turn + np.roll(turn, -1)) / 2 / lengths  # signed, positive = left
    spans: list[CurbSpan] = []
    for side, flagged in (
        ("left", curvature >= min_curvature),
        ("right", curvature <= -min_curvature),
    ):
        start: int | None = None
        for index, value in enumerate([*flagged, False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                spans.append(CurbSpan(side, start, index))
                start = None
    return spans
