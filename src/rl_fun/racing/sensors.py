"""Ray sensors that measure the distance from the car to the track boundaries."""

from __future__ import annotations

import math

import numpy as np

from rl_fun.racing.geometry import cross


def ray_angles(count: int, fov_degrees: float = 180.0) -> np.ndarray:
    """Return ray angles relative to the heading, spread evenly over the field of view."""
    if count < 1:
        raise ValueError("at least one ray is required")
    if count == 1:
        return np.zeros(1)
    half = math.radians(fov_degrees) / 2
    return np.linspace(-half, half, count)


def cast_rays(
    segments: np.ndarray,
    origin: np.ndarray,
    heading: float,
    angles: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Return the distance to the nearest segment for every ray, clipped to max_range.

    `segments` has shape (S, 2, 2): start and end point of each wall segment.
    """
    absolute = heading + angles
    directions = np.stack([np.cos(absolute), np.sin(absolute)], axis=1)[:, None, :]
    starts = segments[:, 0, :]
    edges = segments[:, 1, :] - starts
    offsets = (starts - origin)[None, :, :]
    denominator = cross(directions, edges[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        along_ray = cross(offsets, edges[None]) / denominator
        along_edge = cross(offsets, directions) / denominator
    hit = (np.abs(denominator) > 1e-12) & (along_ray >= 0) & (along_edge >= 0) & (along_edge <= 1)
    nearest = np.where(hit, along_ray, np.inf).min(axis=1)
    return np.minimum(nearest, max_range)


def cast_rays_many(
    segments: np.ndarray,
    origins: np.ndarray,
    headings: np.ndarray,
    angles: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Vectorized `cast_rays` for N origins at once; returns an (N, R) array.

    `origins` has shape (N, 2), `headings` shape (N,), `segments` shape (S, 2, 2).
    """
    absolute = headings[:, None] + angles[None, :]
    directions = np.stack([np.cos(absolute), np.sin(absolute)], axis=2)[:, :, None, :]
    starts = segments[:, 0, :]
    edges = segments[:, 1, :] - starts
    offsets = (starts[None, :, :] - origins[:, None, :])[:, None, :, :]
    edges_b = edges[None, None, :, :]
    denominator = cross(directions, edges_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        along_ray = cross(offsets, edges_b) / denominator
        along_edge = cross(offsets, directions) / denominator
    hit = (np.abs(denominator) > 1e-12) & (along_ray >= 0) & (along_edge >= 0) & (along_edge <= 1)
    nearest = np.where(hit, along_ray, np.inf).min(axis=2)
    return np.minimum(nearest, max_range)
