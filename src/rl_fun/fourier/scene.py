"""Manim scene: two real Fourier chains (x on top, y on the left) draw a contour."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from manim import WHITE, Scene, Text, ValueTracker, VMobject, linear, tempconfig

from rl_fun.fourier.series import approximation_error
from rl_fun.fourier.shapes import circles_as_beziers, segments_as_beziers
from rl_fun.fourier.spec import DrawingSpec, format_error

MIN_RADIUS = 0.01  # frame units; smaller circles are sub-pixel
TRAIL_MIN_STEPS = 1500
TRAIL_STEPS_PER_FREQ = 32  # trail samples per unit of the highest frequency in play
QUALITY = {"low": (540, 960, 30), "high": (1080, 1920, 60)}
FRAME_WIDTH, FRAME_HEIGHT = 9.0, 16.0


def _to_points(z: np.ndarray) -> np.ndarray:
    return np.column_stack([z.real, z.imag, np.zeros(len(z))])


def _set(mob: VMobject, points: np.ndarray) -> None:
    mob.clear_points()
    if len(points):
        mob.set_points(points)


class FourierDrawing(Scene):
    def __init__(self, spec: DrawingSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self._n = min(spec.stages[0], len(spec.x.freqs))
        self._trail_steps = TRAIL_MIN_STEPS
        self._show_chain = True

    def construct(self):
        spec, lay = self.spec, self.spec.layout
        s, row, col = lay.shape_scale, lay.x_row_y, lay.y_col_x
        cx, cy = lay.shape_centre.real, lay.shape_centre.imag
        t = ValueTracker(0.0)

        def place(z: np.ndarray) -> np.ndarray:
            return z * s + lay.shape_centre

        outline = VMobject().set_stroke(WHITE, 2, opacity=0.12).set_fill(opacity=0)
        outline.set_points_as_corners(_to_points(place(np.append(spec.outline, spec.outline[0]))))

        circles = VMobject().set_stroke(WHITE, 1.2, opacity=0.4).set_fill(opacity=0)
        hands = VMobject().set_stroke("#FFD166", 2.5).set_fill(opacity=0)
        drops = VMobject().set_stroke(WHITE, 1, opacity=0.25).set_fill(opacity=0)
        guides = VMobject().set_stroke(WHITE, 1.5, opacity=0.5).set_fill(opacity=0)
        trail = VMobject().set_stroke("#FF4D6D", 4).set_fill(opacity=0)

        def geometry():
            """Centres, radii, hand tips and projection feet of both chains at the current t."""
            n, tv = self._n, t.get_value()
            ax, ay = spec.x.amplitudes[:n] * s, spec.y.amplitudes[:n] * s
            th_x, th_y = spec.x.angles(n, tv), spec.y.angles(n, tv)
            xs = cx + s * spec.x.chain(n, tv)  # X_0 .. X_n
            ys = cy + s * spec.y.chain(n, tv)  # Y_0 .. Y_n
            x_centres = xs[:-1] + 1j * row
            x_tips = xs[1:] + 1j * (row + ax * np.sin(th_x))
            x_feet = xs[1:] + 1j * row
            y_centres = col + 1j * ys[:-1]
            y_tips = (col + ay * np.sin(th_y)) + 1j * ys[1:]
            y_feet = col + 1j * ys[1:]
            return (x_centres, ax, x_tips, x_feet), (y_centres, ay, y_tips, y_feet), xs[-1], ys[-1]

        def update_chains(_mob: VMobject) -> None:
            if not self._show_chain:
                for mob in (circles, hands, drops, guides):
                    mob.clear_points()
                return
            (xc, xr, xt, xf), (yc, yr, yt, yf), px, py = geometry()
            keep_x, keep_y = xr >= MIN_RADIUS, yr >= MIN_RADIUS
            _set(
                circles,
                np.vstack(
                    [
                        circles_as_beziers(xc[keep_x], xr[keep_x]),
                        circles_as_beziers(yc[keep_y], yr[keep_y]),
                    ]
                ),
            )
            _set(hands, segments_as_beziers(np.concatenate([xc, yc]), np.concatenate([xt, yt])))
            _set(drops, segments_as_beziers(np.concatenate([xt, yt]), np.concatenate([xf, yf])))
            point = px + 1j * py
            _set(
                guides,
                segments_as_beziers(
                    np.array([px + 1j * row, col + 1j * py]), np.array([point, point])
                ),
            )

        def update_trail(mob: VMobject) -> None:
            tv = t.get_value()
            times = np.linspace(0.0, tv, max(2, int(self._trail_steps * tv)))
            curve = spec.x.value(self._n, times) + 1j * spec.y.value(self._n, times)
            mob.set_points_as_corners(_to_points(place(curve)))

        circles.add_updater(update_chains)  # one updater drives all four chain mobjects
        trail.add_updater(update_trail)
        self.add(outline, circles, hands, drops, guides, trail)

        label: Text | None = None
        error_label: Text | None = None
        for stage in spec.stages:
            n = min(stage, len(spec.x.freqs))
            self._n = n
            top = max(float(np.max(spec.x.freqs[:n])), float(np.max(spec.y.freqs[:n])))
            self._trail_steps = max(TRAIL_MIN_STEPS, int(TRAIL_STEPS_PER_FREQ * top))
            error = approximation_error(spec.x, spec.y, n, spec.outline)
            new_label = Text(f"N = {n}", font_size=56).move_to([0, 7.2, 0])
            new_error = Text(f"error {format_error(error)}", font_size=36).move_to([0, 6.4, 0])
            for old in (label, error_label):
                if old is not None:
                    self.remove(old)
            label, error_label = new_label, new_error
            self.add(label, error_label)
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
