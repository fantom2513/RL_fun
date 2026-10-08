"""Catalog and track geometry for the lab API: everything the browser needs to draw its forms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rl_fun.lab.config import RunConfig
from rl_fun.racing.fitness import PRESETS, available_terms
from rl_fun.racing.model_spec import (
    _ACTIVATIONS,
    _INPUT_LABELS,
    _OUTPUT_LABELS,
    available_inputs,
    available_outputs,
)
from rl_fun.racing.style import (
    CAR_COLORS,
    CHEQUER_DARK,
    CHEQUER_LIGHT,
    CURB_RED,
    CURB_WHITE,
    GRASS,
    LEADER_BODY,
    RAY,
    ROAD,
    ROAD_LINE,
    Color,
    car_polygons,
    curb_spans,
)
from rl_fun.racing.track import available_tracks, load_track

RAY_MIN_ANGLE = -180.0
RAY_MAX_ANGLE = 180.0

_INPUT_HINTS = {
    "speed": "Скорость вдоль машинки",
    "lateral_speed": "Боковое скольжение",
    "yaw_rate": "Скорость поворота корпуса",
    "acceleration": "Текущее ускорение",
    "steering_angle": "Угол поворота колёс",
}
_OUTPUT_HINTS = {
    "steer": "Поворот руля (обязателен)",
    "throttle": "Газ и тормоз одним выходом",
    "accelerate": "Газ (отдельно от тормоза)",
    "brake": "Тормоз (нужен выход accelerate)",
    "boost": "Кратковременное ускорение",
}
_TERM_LABELS = {
    "progress": "Прогресс: пройденная доля круга",
    "lap_bonus": "Бонус за быстрый круг: чем раньше финиш, тем больше",
    "speed": "Средняя скорость за заезд",
    "survival": "Выживание: доля времени без аварии",
    "centering": "Движение по центру трассы",
    "smoothness": "Плавность руления",
}
_ACTIVATION_LABELS = {
    "tanh": "tanh: гладкая, от −1 до 1",
    "relu": "relu: ноль для отрицательных",
    "sigmoid": "sigmoid: гладкая, от 0 до 1",
}


def _hex(color: Color) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _label_input(name: str) -> str:
    return _INPUT_LABELS[name]


def build_catalog(tracks: Sequence[str] | None = None) -> dict[str, Any]:
    """Everything the interface needs: model parts, fitness terms and presets, defaults, style."""
    names = list(available_tracks() if tracks is None else tracks)
    return {
        "inputs": [
            {"id": name, "label": _label_input(name), "hint": _INPUT_HINTS.get(name, "")}
            for name in available_inputs()
        ],
        "ray": {
            "prefix": "ray:",
            "label": "Луч дальномера",
            "hint": "Расстояние до края трассы под углом к курсу, градусы",
            "min": RAY_MIN_ANGLE,
            "max": RAY_MAX_ANGLE,
        },
        "outputs": [
            {"id": name, "label": _OUTPUT_LABELS[name], "hint": _OUTPUT_HINTS.get(name, "")}
            for name in available_outputs()
        ],
        "activations": [
            {"id": name, "label": _ACTIVATION_LABELS.get(name, name)} for name in _ACTIVATIONS
        ],
        "fitness": {
            "terms": [
                {"id": name, "label": _TERM_LABELS.get(name, name)} for name in available_terms()
            ],
            "presets": {name: spec.to_dict() for name, spec in PRESETS.items()},
        },
        "defaults": RunConfig().to_dict(),
        "tracks": names,
        "style": _build_style(),
    }


def _build_style() -> dict[str, Any]:
    return {
        "grass": _hex(GRASS),
        "road": _hex(ROAD),
        "road_line": _hex(ROAD_LINE),
        "curb_red": _hex(CURB_RED),
        "curb_white": _hex(CURB_WHITE),
        "chequer_dark": _hex(CHEQUER_DARK),
        "chequer_light": _hex(CHEQUER_LIGHT),
        "leader_body": _hex(LEADER_BODY),
        "ray": _hex(RAY),
        "car_colors": {
            key: {"alive": _hex(alive), "dead": _hex(dead)}
            for key, (alive, dead) in CAR_COLORS.items()
        },
        "car": [
            {
                "name": part.name,
                "points": [[x, y] for x, y in part.points],
                "color": part.color_key,
            }
            for part in car_polygons()
        ],
    }


def build_track(name: str) -> dict[str, Any]:
    """Geometry of a built-in track; unknown names (including file paths) raise ValueError."""
    if name not in available_tracks():
        raise ValueError(f"неизвестная трасса: {name!r}")
    track = load_track(name)
    position, heading = track.start_pose()
    return {
        "name": track.name,
        "width": track.width,
        "length": track.length,
        "centerline": track.centerline.tolist(),
        "left": track.left.tolist(),
        "right": track.right.tolist(),
        "curbs": [
            {"side": span.side, "start": span.start, "stop": span.stop}
            for span in curb_spans(track)
        ],
        "start": {"x": float(position[0]), "y": float(position[1]), "heading": heading},
    }
