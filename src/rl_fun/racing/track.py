"""Closed 2D racing track defined by a centerline and a width."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rl_fun.racing.geometry import cross

TRACKS_DIR = Path(__file__).parent / "tracks"


def _segments_intersect(p: np.ndarray, q: np.ndarray, r: np.ndarray, s: np.ndarray) -> bool:
    d1 = cross(q - p, r - p)
    d2 = cross(q - p, s - p)
    d3 = cross(s - r, p - r)
    d4 = cross(s - r, q - r)
    return bool(d1 * d2 < 0 and d3 * d4 < 0)


def _self_intersects(starts: np.ndarray, ends: np.ndarray) -> bool:
    count = len(starts)
    for i in range(count):
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            if _segments_intersect(starts[i], ends[i], starts[j], ends[j]):
                return True
    return False


class Track:
    """Closed track: a centerline polyline and a constant width in meters."""

    def __init__(self, name: str, centerline: Sequence[Sequence[float]], width: float) -> None:
        points = np.asarray(centerline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("centerline must contain at least 3 points of [x, y]")
        if not width > 0:
            raise ValueError("width must be positive")
        ends = np.roll(points, -1, axis=0)
        vectors = ends - points
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths == 0):
            raise ValueError("centerline must not repeat consecutive points")
        if _self_intersects(points, ends):
            raise ValueError("centerline must not intersect itself")

        self.name = name
        self.width = float(width)
        self.centerline = points
        self._vectors = vectors
        self._lengths = lengths
        self._offsets = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
        self.length = float(lengths.sum())
        self.left, self.right = self._boundaries()
        self.boundary_segments = np.concatenate(
            [
                np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)
                for ring in (self.left, self.right)
            ]
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Track:
        """Build a track from the JSON structure `{name, width, centerline}`."""
        for key in ("name", "width", "centerline"):
            if key not in values:
                raise ValueError(f"track definition is missing {key!r}")
        return cls(str(values["name"]), values["centerline"], float(values["width"]))  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, path: Path) -> Track:
        """Load a track from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def project(self, point: Sequence[float]) -> tuple[float, float]:
        """Return arclength along the centerline and distance to the nearest centerline point."""
        position = np.asarray(point, dtype=np.float64)
        relative = position - self.centerline
        fractions = np.clip(
            np.einsum("ij,ij->i", relative, self._vectors) / self._lengths**2, 0.0, 1.0
        )
        closest = self.centerline + fractions[:, None] * self._vectors
        distances = np.linalg.norm(position - closest, axis=1)
        index = int(np.argmin(distances))
        progress = float(self._offsets[index] + fractions[index] * self._lengths[index])
        return progress, float(distances[index])

    def contains(self, point: Sequence[float]) -> bool:
        """Return whether a point lies on the road."""
        return self.project(point)[1] <= self.width / 2

    def start_pose(self) -> tuple[np.ndarray, float]:
        """Return the start position and heading along the first centerline segment."""
        direction = self._vectors[0]
        return self.centerline[0].copy(), float(np.arctan2(direction[1], direction[0]))

    def progress_delta(self, previous: float, current: float) -> float:
        """Return the signed arclength change, wrapping around the start line."""
        delta = current - previous
        half = self.length / 2
        if delta > half:
            delta -= self.length
        elif delta < -half:
            delta += self.length
        return delta

    def _boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        directions = self._vectors / self._lengths[:, None]
        normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1)
        mean = normals + np.roll(normals, 1, axis=0)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        scale = np.clip(np.einsum("ij,ij->i", mean, normals), 0.5, 1.0)
        offset = mean * (self.width / 2 / scale)[:, None]
        return self.centerline + offset, self.centerline - offset


def available_tracks() -> list[str]:
    """Return the names of the built-in tracks."""
    return sorted(path.stem for path in TRACKS_DIR.glob("*.json"))


def load_track(name_or_path: str | Path) -> Track:
    """Load a built-in track by name or a track JSON file by path."""
    text = str(name_or_path)
    builtin = TRACKS_DIR / f"{text}.json"
    path = builtin if builtin.is_file() else Path(text)
    if not path.is_file():
        raise ValueError(
            f"unknown track {text!r}; built-in tracks: {', '.join(available_tracks())}"
        )
    return Track.from_json(path)
