"""Tracks drawn in the lab: validation of a definition, the folder they live in and listing."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from rl_fun.racing.track import Track, builtin_tracks, load_track

NAME_PATTERN = re.compile(r"^[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё _-]{0,39}$")
MIN_WIDTH = 6.0
MAX_WIDTH = 24.0
MIN_POINTS = 4
MAX_POINTS = 400
MAX_COORDINATE = 1000.0
MIN_LENGTH = 100.0


class TrackExistsError(ValueError):
    """The name is already taken by another saved track."""


def _boundaries_overlap(track: Track) -> bool:
    """True when the road edges cross each other or themselves: too tight a turn, or two parts
    of the track closer to each other than its width."""
    left, right = track.left, track.right
    count = len(left)
    heading = np.roll(track.centerline, -1, axis=0) - track.centerline
    for edge in (left, right):  # an edge running backwards is a turn tighter than the road is wide
        if np.any(np.einsum("ij,ij->i", np.roll(edge, -1, axis=0) - edge, heading) <= 0):
            return True
    starts = np.concatenate([left, right])
    ends = np.concatenate([np.roll(left, -1, axis=0), np.roll(right, -1, axis=0)])
    index = np.arange(len(starts))
    ring = index // count
    position = index % count
    distance = np.abs(position[:, None] - position[None, :])
    neighbours = (ring[:, None] == ring[None, :]) & ((distance <= 1) | (distance == count - 1))
    # segments of the two edges that face each other never cross; compare them as well as the rest
    p, q = starts[:, None, :], ends[:, None, :]
    r, s = starts[None, :, :], ends[None, :, :]

    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    d1, d2 = cross(q - p, r - p), cross(q - p, s - p)
    d3, d4 = cross(s - r, p - r), cross(s - r, q - r)
    crossing = (d1 * d2 < 0) & (d3 * d4 < 0) & ~neighbours
    return bool(crossing.any())


def _parts_too_close(track: Track) -> bool:
    """True when two parts of the track that are far apart along it come nearer than its width."""
    starts = track.centerline
    ends = np.roll(starts, -1, axis=0)
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    middle = np.cumsum(lengths) - lengths / 2
    along = np.abs(middle[:, None] - middle[None, :])
    along = np.minimum(along, track.length - along)
    far = along > 1.5 * track.width + lengths.max()

    def to_segments(points: np.ndarray) -> np.ndarray:
        """Distance of every point to every segment: shape (points, segments)."""
        offset = points[:, None, :] - starts[None, :, :]
        fraction = np.clip(
            np.einsum("psk,sk->ps", offset, vectors) / (lengths**2)[None, :], 0.0, 1.0
        )
        nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
        return np.linalg.norm(points[:, None, :] - nearest, axis=2)

    # a point of segment i against segment j covers both orders of every pair
    nearest = np.minimum(to_segments(starts), to_segments(ends))
    return bool(np.any(far & (np.minimum(nearest, nearest.T) < track.width)))


def validate_definition(data: Any) -> Track:
    """Check a `{name, width, centerline}` definition and build the track; ValueError on any
    problem, with a Russian reason that names what to change."""
    if not isinstance(data, dict):
        raise ValueError("ожидался объект с названием, шириной и точками трассы")
    name = data.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(
            "название: от 1 до 40 символов — буквы, цифры, пробел, дефис и подчёркивание"
        )
    if name.lower() in {builtin.lower() for builtin in builtin_tracks()}:
        raise ValueError(f"название «{name}» занято встроенной трассой")
    width = data.get("width")
    if isinstance(width, bool) or not isinstance(width, int | float) or not math.isfinite(width):
        raise ValueError("ширина: нужно число в метрах")
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise ValueError(f"ширина трассы от {MIN_WIDTH:g} до {MAX_WIDTH:g} м")
    points = data.get("centerline")
    if not isinstance(points, list) or not MIN_POINTS <= len(points) <= MAX_POINTS:
        raise ValueError(f"нужно от {MIN_POINTS} до {MAX_POINTS} точек оси трассы")
    for point in points:
        if (
            not isinstance(point, list | tuple)
            or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, int | float) for v in point)
            or any(not math.isfinite(v) or abs(v) > MAX_COORDINATE for v in point)
        ):
            raise ValueError(
                f"координаты точек — числа от −{MAX_COORDINATE:g} до {MAX_COORDINATE:g} м"
            )
    array = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.roll(array, -1, axis=0) - array, axis=1)
    if np.any(steps < 1.0):
        raise ValueError("две соседние точки слишком близко: расстояние не меньше 1 м")
    try:
        track = Track(name, points, float(width))
    except ValueError:
        raise ValueError("трасса пересекает сама себя: уберите самопересечение") from None
    if track.length < MIN_LENGTH:
        raise ValueError(f"трасса слишком короткая: длина не меньше {MIN_LENGTH:g} м")
    if _parts_too_close(track) or _boundaries_overlap(track):
        raise ValueError(
            "трасса слишком узкая для своих поворотов или две её части слишком близко: "
            "уменьшите ширину или раздвиньте участки"
        )
    return track


def _row(track: Track, builtin: bool) -> dict[str, Any]:
    return {
        "name": track.name,
        "builtin": builtin,
        "length": track.length,
        "width": track.width,
        "centerline": track.centerline.tolist(),
    }


def save_track(tracks_dir: Path, data: Any, *, overwrite: bool = False) -> dict[str, Any]:
    """Validate and store a track; returns its listing row."""
    track = validate_definition(data)
    folder = Path(tracks_dir)
    path = folder / f"{track.name}.json"
    taken = path.is_file() or any(
        existing.stem.lower() == track.name.lower() for existing in folder.glob("*.json")
    )
    if taken and not overwrite:
        raise TrackExistsError(f"трасса «{track.name}» уже есть")
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": track.name,
        "width": track.width,
        "centerline": [[float(x), float(y)] for x, y in track.centerline],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return _row(track, builtin=False)


def delete_track(tracks_dir: Path, name: str) -> None:
    """Remove a saved track: ValueError for a built-in one, KeyError when there is none."""
    if name.lower() in {builtin.lower() for builtin in builtin_tracks()}:
        raise ValueError("встроенную трассу удалить нельзя")
    path = Path(tracks_dir) / f"{name}.json"
    if Path(name).name != name or not path.is_file():
        raise KeyError(name)
    path.unlink()


def list_tracks(tracks_dir: Path) -> list[dict[str, Any]]:
    """Built-in tracks first, then the saved ones, each as a listing row with its centerline."""
    rows = [_row(load_track(name, tracks_dir), True) for name in builtin_tracks()]
    folder = Path(tracks_dir)
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                rows.append(_row(Track.from_json(path), False))
            except (ValueError, OSError, KeyError):
                continue  # a damaged file must not hide the other tracks
    return rows
