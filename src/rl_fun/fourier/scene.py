"""Manim scene: a chain of epicycles draws a contour while N grows by stages."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from manim import UP, WHITE, Scene, Text, ValueTracker, VMobject, linear, tempconfig

from rl_fun.fourier.series import chain_positions, partial_sum
from rl_fun.fourier.spec import DrawingSpec

CIRCLE_SEGMENTS = 32
MIN_RADIUS = 0.01  # frame units; smaller circles are sub-pixel
TRAIL_STEPS = 800
QUALITY = {"low": (540, 960, 30), "high": (1080, 1920, 60)}
FRAME_WIDTH, FRAME_HEIGHT = 9.0, 16.0


def _to_points(z: np.ndarray) -> np.ndarray:
    return np.column_stack([z.real, z.imag, np.zeros(len(z))])


class FourierDrawing(Scene):
    def __init__(self, spec: DrawingSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self._terms = 0  # number of series terms in play (circles, plus a fixed offset)
        self._trail_steps = TRAIL_STEPS
        self._show_chain = True

    def _place(self, z: np.ndarray) -> np.ndarray:
        return z * self.spec.scale + self.spec.centre

    def construct(self):
        spec = self.spec
        t = ValueTracker(0.0)

        # The zero-frequency term is a fixed offset, not a circle: put it first so a stage
        # of N shows exactly N circles; the rest keep their descending |c| order.
        zero = np.flatnonzero(spec.freqs == 0)
        order = np.concatenate((zero, np.setdiff1d(np.arange(len(spec.freqs)), zero)))
        freqs = spec.freqs[order]
        coeffs = spec.coeffs[order]
        has_offset = len(zero) > 0

        outline = VMobject().set_stroke(WHITE, 2, opacity=0.12).set_fill(opacity=0)
        closed = np.append(spec.outline, spec.outline[0])
        outline.set_points_as_corners(_to_points(self._place(closed)))

        circles = VMobject().set_stroke(WHITE, 1, opacity=0.35).set_fill(opacity=0)
        arms = VMobject().set_stroke("#FFD166", 2).set_fill(opacity=0)
        trail = VMobject().set_stroke("#FF4D6D", 4).set_fill(opacity=0)
        angles = np.linspace(0.0, 2 * np.pi, CIRCLE_SEGMENTS + 1)
        ring = np.exp(1j * angles)

        def positions() -> np.ndarray:
            chain = chain_positions(freqs, coeffs, self._terms, t.get_value())
            return self._place(chain)

        def update_circles(mob: VMobject) -> None:
            mob.clear_points()
            if not self._show_chain:
                return
            centres = positions()
            for k in range(self._terms):
                radius = abs(coeffs[k]) * spec.scale
                if freqs[k] == 0 or radius < MIN_RADIUS:
                    continue  # the zero frequency is a fixed offset, not a circle
                pts = _to_points(centres[k] + radius * ring)
                mob.start_new_path(pts[0])
                mob.add_points_as_corners(pts[1:])

        def update_arms(mob: VMobject) -> None:
            mob.clear_points()
            if self._show_chain:
                mob.set_points_as_corners(_to_points(positions()))

        def update_trail(mob: VMobject) -> None:
            tv = t.get_value()
            times = np.linspace(0.0, tv, max(2, int(self._trail_steps * tv)))
            curve = partial_sum(freqs, coeffs, self._terms, times)
            mob.set_points_as_corners(_to_points(self._place(curve)))

        circles.add_updater(update_circles)
        arms.add_updater(update_arms)
        trail.add_updater(update_trail)
        self.add(outline, circles, arms, trail)

        label: Text | None = None
        for n in spec.stages:
            self._terms = n + 1 if has_offset else n
            top_frequency = int(np.max(np.abs(freqs[: self._terms])))
            self._trail_steps = max(TRAIL_STEPS, 16 * top_frequency)
            new_label = Text(f"N = {n}", font_size=56).to_edge(UP, buff=1.2)
            if label is not None:
                self.remove(label)
            label = new_label
            self.add(label)
            t.set_value(0.0)
            self.wait(0.3, frozen_frame=False)
            self.play(t.animate.set_value(1.0), run_time=spec.seconds_per_stage, rate_func=linear)

        self.wait(1.0, frozen_frame=False)
        self._show_chain = False
        self.wait(2.0, frozen_frame=False)


def render(spec: DrawingSpec, out_path: Path, quality: str = "high") -> Path:
    """Render ``spec`` to ``out_path`` (mp4) and return the final path."""
    width, height, fps = QUALITY[quality]
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "pixel_width": width,
        "pixel_height": height,
        "frame_rate": fps,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "media_dir": str(out_path.parent / "_manim"),
        "output_file": out_path.stem,
        "format": "mp4",
        "disable_caching": True,
        "verbosity": "WARNING",
    }
    with tempconfig(settings):
        scene = FourierDrawing(spec)
        scene.render()
        produced = Path(scene.renderer.file_writer.movie_file_path)
    shutil.copyfile(produced, out_path)
    return out_path
