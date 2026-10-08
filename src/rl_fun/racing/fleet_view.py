"""pygame view of a whole fleet, with an optional panel showing the leader's neural network."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pygame

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.render import GAME_HEIGHT, Renderer
from rl_fun.racing.style import (
    EDGE_NEGATIVE,
    EDGE_POSITIVE,
    NODE_NEGATIVE,
    NODE_NEUTRAL,
    NODE_OUTLINE,
    NODE_POSITIVE,
    PANEL_BACKGROUND,
    PANEL_MUTED,
    PANEL_TEXT,
)

EDGE_MIN_STRENGTH = 0.05
NODE_RADIUS = 10
PANEL_TOP = 60
PANEL_BOTTOM = GAME_HEIGHT - 24
LABEL_GUTTER = 84
VALUE_GUTTER = 84


class ViewClosed(Exception):
    """Raised when the user closes the window; stops the training loop."""


def _mix(
    low: tuple[int, int, int], high: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    """Blend from `low` to `high`; `amount` is clipped to [0, 1]."""
    t = min(max(amount, 0.0), 1.0)
    red, green, blue = (int(round(a + (b - a) * t)) for a, b in zip(low, high, strict=True))
    return red, green, blue


def _layer_xs(count: int, left: int, right: int) -> list[float]:
    if count == 1:
        return [(left + right) / 2]
    step = (right - left) / (count - 1)
    return [left + i * step for i in range(count)]


def _node_ys(count: int) -> list[float]:
    spacing = (PANEL_BOTTOM - PANEL_TOP) / count
    return [PANEL_TOP + (i + 0.5) * spacing for i in range(count)]


class NetworkPanel:
    """Side panel drawing a fully connected network: nodes per layer, weighted edges, values.

    Edges are green for positive weights and red for negative ones; thickness and saturation grow
    with |w|, and weak edges are skipped. Nodes are filled green or red by the sign of the
    activation. Every input and output label has its current value printed underneath. Without a
    state set by `update` only the title is drawn.
    """

    def __init__(
        self, input_labels: Sequence[str], output_labels: Sequence[str], width: int = 360
    ) -> None:
        pygame.font.init()
        self.input_labels = list(input_labels)
        self.output_labels = list(output_labels)
        self.width = int(width)
        self._matrices: list[np.ndarray] | None = None
        self._activations: list[np.ndarray] | None = None
        self._font = pygame.font.Font(None, 22)
        self._value_font = pygame.font.Font(None, 19)
        self._title_font = pygame.font.Font(None, 28)

    def update(self, matrices: Sequence[np.ndarray], activations: Sequence[np.ndarray]) -> None:
        """Store weights (layer k maps activations[k] to activations[k+1]) and activations."""
        self._matrices = [np.asarray(m, dtype=np.float64) for m in matrices]
        self._activations = [np.asarray(a, dtype=np.float64) for a in activations]

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw the network into `rect` of `surface`."""
        surface.fill(PANEL_BACKGROUND, rect)
        title = self._title_font.render("Нейросеть лидера", True, PANEL_TEXT)
        surface.blit(title, (rect.left + 12, rect.top + 14))
        if self._activations is None or self._matrices is None:
            return
        left = rect.left + LABEL_GUTTER
        right = rect.right - VALUE_GUTTER
        xs = _layer_xs(len(self._activations), left, right)
        positions = [
            [(x, y) for y in _node_ys(len(layer))]
            for x, layer in zip(xs, self._activations, strict=True)
        ]
        for matrix, (src, dst) in zip(
            self._matrices, zip(positions[:-1], positions[1:], strict=True), strict=True
        ):
            self._draw_edges(surface, matrix, src, dst)
        for layer_index, layer in enumerate(positions):
            for node_index, (x, y) in enumerate(layer):
                value = float(self._activations[layer_index][node_index])
                self._draw_node(surface, (x, y), value)
        self._draw_labels(surface, positions)

    def _draw_edges(
        self,
        surface: pygame.Surface,
        matrix: np.ndarray,
        src: list[tuple[float, float]],
        dst: list[tuple[float, float]],
    ) -> None:
        edges = []
        for out_index, (ox, oy) in enumerate(dst):
            for in_index, (ix, iy) in enumerate(src):
                weight = float(matrix[out_index, in_index])
                strength = min(abs(weight), 1.0)
                if strength >= EDGE_MIN_STRENGTH:
                    edges.append((strength, weight > 0, (ix, iy), (ox, oy)))
        for strength, positive, start, end in sorted(edges, key=lambda edge: edge[0]):
            base = EDGE_POSITIVE if positive else EDGE_NEGATIVE
            color = _mix(PANEL_BACKGROUND, base, 0.3 + 0.7 * strength)
            pygame.draw.line(surface, color, start, end, 1 + int(4 * strength))

    def _draw_node(
        self, surface: pygame.Surface, center: tuple[float, float], value: float
    ) -> None:
        target = NODE_POSITIVE if value >= 0 else NODE_NEGATIVE
        color = _mix(NODE_NEUTRAL, target, abs(value) ** 0.7)
        pygame.draw.circle(surface, color, center, NODE_RADIUS)
        pygame.draw.circle(surface, NODE_OUTLINE, center, NODE_RADIUS, 3)

    def _draw_labels(
        self, surface: pygame.Surface, positions: list[list[tuple[float, float]]]
    ) -> None:
        inputs = self._activations[0] if self._activations else np.zeros(0)
        for name, (x, y), value in zip(self.input_labels, positions[0], inputs, strict=False):
            x_text = x - NODE_RADIUS - 8
            self._blit_label(surface, name, f"{value:.2f}", (x_text, y), right_aligned=True)
        output_nodes = positions[-1] if len(positions) > 1 else []
        outputs = self._activations[-1] if self._activations else np.zeros(0)
        for name, (x, y), value in zip(self.output_labels, output_nodes, outputs, strict=False):
            x_text = x + NODE_RADIUS + 8
            self._blit_label(surface, name, f"{value:.2f}", (x_text, y), right_aligned=False)

    def _blit_label(
        self,
        surface: pygame.Surface,
        name: str,
        value: str,
        anchor: tuple[float, float],
        right_aligned: bool,
    ) -> None:
        """Name with its value underneath, vertically centred on the node."""
        arrow = name.startswith("↑")  # the default font has no arrow glyph, so it is drawn
        title = self._font.render(
            name.removeprefix("↑").strip() if arrow else name, True, PANEL_TEXT
        )
        number = self._value_font.render(value, True, PANEL_MUTED)
        x, y = anchor
        top = y - (title.get_height() + number.get_height() - 2) / 2
        for line, offset in ((title, 0), (number, title.get_height() - 2)):
            left = x - line.get_width() if right_aligned else x
            surface.blit(line, (left, top + offset))
        if arrow:
            self._draw_arrow(surface, (x - title.get_width() - 9, top + title.get_height() / 2))

    @staticmethod
    def _draw_arrow(surface: pygame.Surface, center: tuple[float, float]) -> None:
        cx, cy = center
        pygame.draw.line(surface, PANEL_TEXT, (cx, cy + 5), (cx, cy - 5), 2)
        pygame.draw.lines(
            surface, PANEL_TEXT, False, [(cx - 4, cy - 1), (cx, cy - 5), (cx + 4, cy - 1)], 2
        )


def default_input_labels(ray_angles: np.ndarray) -> list[str]:
    """Labels for the observation: one per ray angle in degrees, then speed, slip and yaw."""
    rays = [f"↑ {round(float(np.degrees(angle)))}°" for angle in ray_angles]
    return [*rays, "Скор.", "Бок.", "Угл."]


class FleetView:
    """Draw a `RacingFleet` each step; optionally show the leader's network next to the track.

    Use `mode="human"` for a window (closing it raises `ViewClosed`) or `"rgb_array"` to get frames.
    """

    def __init__(
        self,
        fleet: RacingFleet,
        mode: str = "human",
        show_network: bool = True,
        input_labels: Sequence[str] | None = None,
        output_labels: Sequence[str] = ("Руль", "Газ"),
    ) -> None:
        self._fleet = fleet
        self._mode = mode
        self.panel: NetworkPanel | None = None
        if show_network:
            labels = (
                input_labels if input_labels is not None else default_input_labels(fleet.ray_angles)
            )
            self.panel = NetworkPanel(labels, output_labels)
        self._renderer = Renderer(fleet.track, mode, fps=round(1 / fleet.dt), side_panel=self.panel)

    def draw(self, fleet: RacingFleet, lines: list[str], network: Any) -> np.ndarray | None:
        """Draw one frame. `network` is `(matrices, activations)` of the leader, or None."""
        if self.panel is not None and network is not None:
            matrices, activations = network
            self.panel.update(matrices, activations)
        leader = self._leader(fleet)
        ray_points = self._ray_points(fleet, leader)
        frame = self._renderer.draw_fleet(
            fleet.x, fleet.y, fleet.heading, fleet.alive, leader, ray_points, lines
        )
        if self._mode == "human" and any(event.type == pygame.QUIT for event in pygame.event.get()):
            raise ViewClosed
        return frame

    def close(self) -> None:
        """Release the window and pygame resources."""
        self._renderer.close()

    @staticmethod
    def _leader(fleet: RacingFleet) -> int:
        """Living car with the largest progress; if all crashed, the largest progress overall."""
        living = np.flatnonzero(fleet.alive)
        pool = living if len(living) > 0 else np.arange(fleet.n_cars)
        return int(pool[np.argmax(fleet.progress[pool])])

    @staticmethod
    def _ray_points(fleet: RacingFleet, car: int) -> np.ndarray:
        angles = fleet.heading[car] + fleet.ray_angles
        distances = fleet.distances[car]
        return np.column_stack(
            [
                fleet.x[car] + np.cos(angles) * distances,
                fleet.y[car] + np.sin(angles) * distances,
            ]
        )
