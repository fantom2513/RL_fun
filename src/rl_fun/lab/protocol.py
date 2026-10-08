"""Messages the lab worker sends to the manager: frames, generation summaries and status.

Every value is a plain Python type (int, float, str, bool, list, dict or None), so messages go
straight through `json.dumps`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import GenerationResult

STATUSES = ("running", "paused", "finished", "stopped", "error")
STATE_CRASHED = 0
STATE_ALIVE = 1
STATE_FINISHED = 2


def _round(value: Any) -> float:
    return round(float(value), 2)


def _car_state(fleet: RacingFleet, index: int) -> int:
    if fleet.finished[index]:
        return STATE_FINISHED
    if fleet.alive[index]:
        return STATE_ALIVE
    return STATE_CRASHED


def _leader(fleet: RacingFleet) -> int:
    """Index of the living car with the most progress; if nobody lives, of the best car overall."""
    progress = fleet.progress
    living = np.flatnonzero(fleet.alive)
    pool = living if living.size > 0 else np.arange(fleet.n_cars)
    return int(pool[np.argmax(progress[pool])])


def build_frame(
    fleet: RacingFleet,
    generation: int,
    network: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None = None,
) -> dict[str, Any]:
    """One animation frame: car poses and states, the leader's rays, HUD and optional network."""
    leader = _leader(fleet)
    cars = [
        [
            _round(fleet.x[index]),
            _round(fleet.y[index]),
            _round(fleet.heading[index]),
            _car_state(fleet, index),
        ]
        for index in range(fleet.n_cars)
    ]
    angles = fleet.heading[leader] + fleet.ray_angles
    distances = fleet.distances[leader]
    ray_x = fleet.x[leader] + np.cos(angles) * distances
    ray_y = fleet.y[leader] + np.sin(angles) * distances
    rays = [[_round(x), _round(y)] for x, y in zip(ray_x, ray_y, strict=True)]

    net = None
    if network is not None:
        matrices, activations = network
        net = {
            "matrices": [np.asarray(m, dtype=np.float64).tolist() for m in matrices],
            "activations": [np.asarray(a, dtype=np.float64).tolist() for a in activations],
        }

    return {
        "t": "frame",
        "gen": int(generation),
        "step": int(fleet.steps),
        "alive": int(np.count_nonzero(fleet.alive)),
        "n": int(fleet.n_cars),
        "cars": cars,
        "leader": leader,
        "rays": rays,
        "net": net,
        "hud": {
            "speed": _round(fleet.v_long[leader]),
            "progress": _round(fleet.progress[leader]),
        },
    }


def build_gen_message(
    generation: int,
    result: GenerationResult,
    scores: np.ndarray,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Summary of a finished generation, with the tunable parameters it was run with."""
    progress = np.asarray(result.progress, dtype=np.float64)
    finished = np.asarray(result.finished, dtype=bool)
    lap_steps = np.asarray(result.lap_steps, dtype=np.float64)
    finished_laps = lap_steps[finished & np.isfinite(lap_steps)]
    echoed: dict[str, Any] = {}
    if "mutation_rate" in params:
        echoed["mutation_rate"] = float(params["mutation_rate"])
    if "mutation_scale" in params:
        echoed["mutation_scale"] = float(params["mutation_scale"])
    if "elite" in params:
        echoed["elite"] = int(params["elite"])
    return {
        "t": "gen",
        "gen": int(generation),
        "best": float(np.max(progress)),
        "mean": float(np.mean(progress)),
        "finished": int(np.count_nonzero(finished)),
        "best_fitness": float(np.max(np.asarray(scores, dtype=np.float64))),
        "best_lap_steps": int(finished_laps.min()) if finished_laps.size > 0 else None,
        "params": echoed,
    }


def status_message(status: str, message: str | None = None) -> dict[str, Any]:
    """Run status change; `message` carries the error text for status 'error'."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; supported: {STATUSES}")
    payload: dict[str, Any] = {"t": "status", "status": status}
    if message is not None:
        payload["message"] = str(message)
    return payload


def notice_message(text: str) -> dict[str, Any]:
    """Non-fatal note for the user, for example a rejected parameter update."""
    return {"t": "notice", "text": str(text)}
