"""pygame view of a whole fleet, with an optional panel showing the leader's neural network."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pygame
from pygame import gfxdraw

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.render import Renderer
from rl_fun.racing.style import (
    EDGE_ALPHA_FIRST,
    EDGE_NEGATIVE,
    EDGE_POSITIVE,
    LEGEND_ENTRIES,
    NODE_NEGATIVE,
    NODE_NEUTRAL,
    NODE_OUTLINE,
    NODE_POSITIVE,
    PANEL_BACKGROUND,
    PANEL_MUTED,
    PANEL_TEXT,
    edge_threshold,
    network_column_titles,
    visible_edges,
)

NODE_RADIUS = 11
PANEL_TOP = 100
LEGEND_PADDING = 14  # px below the legend and between legend and network
LEGEND_GAP = 8  # px between legend entries
LABEL_GUTTER = 84
VALUE_GUTTER = 84
PANEL_SUPERSAMPLE = 2  # shapes are drawn this much larger, then smoothed down; text stays crisp
GLOW_THRESHOLD = 0.5  # |activation| above which a node gets a soft glow
EDGE_CURVE = 0.42  # share of the column gap used as horizontal tangent: 0 = straight lines
EDGE_SAMPLES = 16
FONT_NAMES = ["segoeui", "arial", "dejavusans", "freesans"]
LEGEND_TEXT_LEFT = 56

Point = tuple[float, float]


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


def _node_ys(count: int, top: float, bottom: float) -> list[float]:
    spacing = (bottom - top) / count
    return [top + (i + 0.5) * spacing for i in range(count)]


def _load_font(size: int, bold: bool = False) -> pygame.font.Font:
    """A system font that has Cyrillic if possible, else pygame's built-in default font."""
    try:
        return pygame.font.SysFont(FONT_NAMES, size, bold=bold)
    except Exception:  # noqa: BLE001 - any font-system failure falls back to the bundled font
        return pygame.font.Font(None, size + 6)


def _edge_polygon(start: Point, end: Point, width: float, scale: float) -> list[Point]:
    """Outline of a thick, gently S-shaped curve from `start` to `end` (one simple polygon)."""
    (x0, y0), (x1, y1) = start, end
    reach = (x1 - x0) * EDGE_CURVE
    controls = ((x0, y0), (x0 + reach, y0), (x1 - reach, y1), (x1, y1))
    samples = np.linspace(0.0, 1.0, EDGE_SAMPLES)[:, None]
    p0, p1, p2, p3 = (np.asarray(point, dtype=np.float64) for point in controls)
    curve = (
        (1 - samples) ** 3 * p0
        + 3 * (1 - samples) ** 2 * samples * p1
        + 3 * (1 - samples) * samples**2 * p2
        + samples**3 * p3
    )
    tangent = np.gradient(curve, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / norm
    half = width * scale / 2
    upper = (curve * scale + normal * half).tolist()
    lower = (curve * scale - normal * half).tolist()
    return [tuple(point) for point in upper + lower[::-1]]


class NetworkPanel:
    """Side panel drawing a fully connected network, with column captions and a legend.

    Edges are green for positive weights and red for negative ones; thickness and opacity grow
    with |w|, and edges below a share of the layer's largest |w| are skipped. Nodes are filled
    green or red by the sign of the activation, with a glow when strongly active. Every input and
    output label has its current value printed underneath. Shapes are drawn at 2x and smoothed
    down; without a state set by `update` only the title, captions and legend are drawn.
    """

    def __init__(
        self, input_labels: Sequence[str], output_labels: Sequence[str], width: int = 420
    ) -> None:
        pygame.font.init()
        self.input_labels = list(input_labels)
        self.output_labels = list(output_labels)
        self.width = int(width)
        self._matrices: list[np.ndarray] | None = None
        self._activations: list[np.ndarray] | None = None
        self._font = _load_font(15)
        self._value_font = _load_font(13)
        self._title_font = _load_font(22, bold=True)
        self._header_font = _load_font(14, bold=True)
        self._legend_font = _load_font(15)
        self._big: pygame.Surface | None = None

    def update(self, matrices: Sequence[np.ndarray], activations: Sequence[np.ndarray]) -> None:
        """Store weights (layer k maps activations[k] to activations[k+1]) and activations."""
        self._matrices = [np.asarray(m, dtype=np.float64) for m in matrices]
        self._activations = [np.asarray(a, dtype=np.float64) for a in activations]

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw the network into `rect` of `surface`."""
        k = PANEL_SUPERSAMPLE
        size = (rect.width * k, rect.height * k)
        if self._big is None or self._big.get_size() != size:
            self._big = pygame.Surface(size)
        big = self._big
        big.fill(PANEL_BACKGROUND)
        positions: list[list[Point]] = []
        if self._activations is not None and self._matrices is not None:
            bottom = max(PANEL_TOP + 60, self._legend_top(rect.width, rect.height) - 18)
            xs = _layer_xs(len(self._activations), LABEL_GUTTER, rect.width - VALUE_GUTTER)
            positions = [
                [(x, y) for y in _node_ys(len(layer), PANEL_TOP, bottom)]
                for x, layer in zip(xs, self._activations, strict=True)
            ]
            for index, (matrix, src, dst) in enumerate(
                zip(self._matrices, positions[:-1], positions[1:], strict=True)
            ):
                self._draw_edges(big, matrix, src, dst, index)
            for layer_index, layer in enumerate(positions):
                for node_index, center in enumerate(layer):
                    value = float(self._activations[layer_index][node_index])
                    self._draw_node(big, center, value)
        legend = self._legend_layout(rect.width, rect.height)
        divider = self._legend_top(rect.width, rect.height)
        self._draw_legend_swatches(big, legend, rect.width, divider)
        panel = pygame.transform.smoothscale(big, (rect.width, rect.height))
        self._draw_text(panel, positions, legend)
        surface.blit(panel, rect.topleft)

    # -- shapes (drawn on the supersampled surface) ------------------------------------------

    def _draw_edges(
        self,
        big: pygame.Surface,
        matrix: np.ndarray,
        src: list[Point],
        dst: list[Point],
        layer_index: int = 1,
    ) -> None:
        shown = visible_edges(matrix, edge_threshold(layer_index))
        fade = EDGE_ALPHA_FIRST if layer_index == 0 else 1.0
        largest = float(np.abs(matrix).max()) if matrix.size else 0.0
        edges = []
        for out_index, end in enumerate(dst):
            for in_index, start in enumerate(src):
                if shown[out_index, in_index]:
                    weight = float(matrix[out_index, in_index])
                    edges.append((abs(weight) / largest, weight > 0, start, end))
        for strength, positive, start, end in sorted(edges, key=lambda edge: edge[0]):
            self._fill_edge(big, start, end, strength, positive, fade)

    @staticmethod
    def _fill_edge(
        big: pygame.Surface,
        start: Point,
        end: Point,
        strength: float,
        positive: bool,
        fade: float = 1.0,
    ) -> None:
        base = EDGE_POSITIVE if positive else EDGE_NEGATIVE
        alpha = int((70 + 150 * strength) * fade)
        polygon = _edge_polygon(start, end, 0.9 + 4.4 * strength, PANEL_SUPERSAMPLE)
        gfxdraw.filled_polygon(big, polygon, (*base, alpha))
        gfxdraw.aapolygon(big, polygon, (*base, alpha))

    @staticmethod
    def _draw_node(big: pygame.Surface, center: Point, value: float) -> None:
        k = PANEL_SUPERSAMPLE
        cx, cy = round(center[0] * k), round(center[1] * k)
        radius = NODE_RADIUS * k
        target = NODE_POSITIVE if value >= 0 else NODE_NEGATIVE
        strength = min(abs(value), 1.0)
        fill = _mix(NODE_NEUTRAL, target, strength**0.7)
        if strength > GLOW_THRESHOLD:
            for grow, alpha in ((9, 24), (6, 34), (3, 46)):
                gfxdraw.filled_circle(big, cx, cy, radius + grow * k // 2, (*target, alpha))
        gfxdraw.filled_circle(big, cx, cy, radius, NODE_OUTLINE)
        gfxdraw.aacircle(big, cx, cy, radius, NODE_OUTLINE)
        inner = radius - round(2.2 * k)
        gfxdraw.filled_circle(big, cx, cy, inner, fill)
        gfxdraw.aacircle(big, cx, cy, inner, fill)

    # -- legend -------------------------------------------------------------------------------

    def _legend_entries(self, width: int) -> tuple[list[tuple[str, list[str]]], float]:
        """Legend entries wrapped to the panel width and the total height of the block."""
        limit = width - LEGEND_TEXT_LEFT - 10
        line_height = self._legend_font.get_linesize()
        entries = []
        for kind, text in LEGEND_ENTRIES:
            lines: list[str] = []
            current = ""
            for word in text.split():
                trial = f"{current} {word}".strip()
                if current and self._legend_font.size(trial)[0] > limit:
                    lines.append(current)
                    current = word
                else:
                    current = trial
            lines.append(current)
            entries.append((kind, lines))
        total = sum(len(lines) * line_height + LEGEND_GAP for _, lines in entries) - LEGEND_GAP
        return entries, total

    def _legend_top(self, width: int, height: int) -> float:
        """y of the divider line above the legend block, which sits at the bottom of the panel."""
        _, total = self._legend_entries(width)
        return height - LEGEND_PADDING - total - LEGEND_PADDING

    def _legend_layout(self, width: int, height: int) -> list[tuple[str, float, list[str]]]:
        """(kind, centre y, wrapped lines) for each legend entry, stacked at the panel bottom."""
        entries, _ = self._legend_entries(width)
        line_height = self._legend_font.get_linesize()
        top = self._legend_top(width, height) + LEGEND_PADDING
        laid_out = []
        for kind, lines in entries:
            block = len(lines) * line_height
            laid_out.append((kind, top + block / 2, lines))
            top += block + LEGEND_GAP
        return laid_out

    def _draw_legend_swatches(
        self,
        big: pygame.Surface,
        legend: list[tuple[str, float, list[str]]],
        width: int,
        divider: float,
    ) -> None:
        k = PANEL_SUPERSAMPLE
        line = ((12 * k, divider * k), ((width - 12) * k, divider * k))
        pygame.draw.line(big, (214, 218, 224), *line, k)
        for kind, y, _ in legend:
            left, right = 12.0, 46.0
            if kind in ("positive", "negative"):
                self._fill_edge(
                    big, (left, y + 5), (right, y - 5), 0.8, positive=kind == "positive"
                )
            elif kind == "thickness":
                for offset, strength in ((-6.0, 0.05), (0.0, 0.45), (6.0, 1.0)):
                    polygon = _edge_polygon(
                        (left, y + offset), (right, y + offset), 0.9 + 4.4 * strength, k
                    )
                    gfxdraw.filled_polygon(big, polygon, (*PANEL_MUTED, 210))
                    gfxdraw.aapolygon(big, polygon, (*PANEL_MUTED, 210))
            else:
                self._draw_node(big, (left + 9, y), 1.0)
                self._draw_node(big, (right - 9, y), -1.0)

    # -- text (drawn at 1x on the smoothed surface) --------------------------------------------

    def _draw_text(
        self,
        panel: pygame.Surface,
        positions: list[list[Point]],
        legend: list[tuple[str, float, list[str]]],
    ) -> None:
        panel.blit(self._title_font.render("Нейросеть лидера", True, PANEL_TEXT), (12, 12))
        if positions:
            titles = network_column_titles(len(positions))
            for title, layer in zip(titles, positions, strict=True):
                header = self._header_font.render(title, True, PANEL_MUTED)
                panel.blit(header, (layer[0][0] - header.get_width() / 2, PANEL_TOP - 30))
            self._draw_labels(panel, positions)
        line_height = self._legend_font.get_linesize()
        for _kind, y, lines in legend:
            top = y - len(lines) * line_height / 2
            for index, line in enumerate(lines):
                text = self._legend_font.render(line, True, PANEL_TEXT)
                panel.blit(text, (LEGEND_TEXT_LEFT, top + index * line_height))

    def _draw_labels(self, surface: pygame.Surface, positions: list[list[Point]]) -> None:
        inputs = self._activations[0] if self._activations else np.zeros(0)
        for name, (x, y), value in zip(self.input_labels, positions[0], inputs, strict=False):
            self._blit_label(surface, name, f"{value:.2f}", (x - NODE_RADIUS - 8, y), True)
        output_nodes = positions[-1] if len(positions) > 1 else []
        outputs = self._activations[-1] if self._activations else np.zeros(0)
        for name, (x, y), value in zip(self.output_labels, output_nodes, outputs, strict=False):
            self._blit_label(surface, name, f"{value:.2f}", (x + NODE_RADIUS + 8, y), False)

    def _blit_label(
        self, surface: pygame.Surface, name: str, value: str, anchor: Point, right_aligned: bool
    ) -> None:
        """Name with its value underneath, vertically centred on the node."""
        arrow = name.startswith("↑")  # fonts may lack the arrow glyph, so it is always drawn
        title = self._font.render(
            name.removeprefix("↑").strip() if arrow else name, True, PANEL_TEXT
        )
        number = self._value_font.render(value, True, PANEL_MUTED)
        x, y = anchor
        top = y - (title.get_height() + number.get_height() - 4) / 2
        for line, offset in ((title, 0), (number, title.get_height() - 4)):
            left = x - line.get_width() if right_aligned else x
            surface.blit(line, (left, top + offset))
        if arrow:
            self._draw_arrow(surface, (x - title.get_width() - 9, top + title.get_height() / 2))

    @staticmethod
    def _draw_arrow(surface: pygame.Surface, center: Point) -> None:
        cx, cy = center
        pygame.draw.aaline(surface, PANEL_TEXT, (cx, cy + 5), (cx, cy - 5))
        pygame.draw.aaline(surface, PANEL_TEXT, (cx - 4, cy - 1), (cx, cy - 5))
        pygame.draw.aaline(surface, PANEL_TEXT, (cx + 4, cy - 1), (cx, cy - 5))

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
        output_labels: Sequence[str] | None = None,
        camera: str = "fit",
        zoom: float = 1.0,
        resizable: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        self._fleet = fleet
        self._mode = mode
        self.panel: NetworkPanel | None = None
        if show_network:
            labels = input_labels if input_labels is not None else fleet.model.input_labels
            outputs = output_labels if output_labels is not None else fleet.model.output_labels
            self.panel = NetworkPanel(labels, outputs)
        self._renderer = Renderer(
            fleet.track,
            mode,
            fps=round(1 / fleet.dt),
            side_panel=self.panel,
            camera=camera,
            zoom=zoom,
            resizable=resizable,
            size=size,
        )

    def draw(self, fleet: RacingFleet, lines: list[str], network: Any) -> np.ndarray | None:
        """Draw one frame. `network` is `(matrices, activations)` of the leader, or None."""
        if self.panel is not None and network is not None:
            matrices, activations = network
            self.panel.update(matrices, activations)
        leader = self._leader(fleet)
        ray_points = self._ray_points(fleet, leader)
        frame = self._renderer.draw_fleet(
            fleet.x,
            fleet.y,
            fleet.heading,
            fleet.alive,
            leader,
            ray_points,
            lines,
            steps=fleet.steps,
            finished=fleet.finished,
        )
        if self._mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise ViewClosed
                self._renderer.handle_event(event)
        return frame

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Forward a pygame event (zoom, camera, resize) to the renderer."""
        return self._renderer.handle_event(event)

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
