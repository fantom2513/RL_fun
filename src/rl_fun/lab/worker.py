"""Lab worker: runs one evolution in its own process and streams it to the manager.

`run_worker` is the process entry point. It never raises: a stop request ends the run with status
"stopped", any other failure ends it with status "error" carrying the error text.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rl_fun.lab.config import RunConfig
from rl_fun.lab.protocol import build_gen_message, notice_message, status_message
from rl_fun.lab.stream_view import StopRun, StreamView
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import run_generation

UPDATABLE = ("mutation_rate", "mutation_scale", "elite", "fitness")
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class WorkerConnection(Protocol):
    """The worker's end of the channel: `multiprocessing.Connection` or a stand-in."""

    def send(self, message: dict[str, Any]) -> None:
        """Send one JSON-compatible message to the manager."""

    def poll(self, timeout: float = 0) -> bool:
        """Return whether a command is waiting to be received."""

    def recv(self) -> Any:
        """Receive one command."""


def run_worker(
    config: dict[str, Any],
    conn: WorkerConnection,
    runs_dir: str | Path,
    *,
    speed: object = "max",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run the evolution described by `config` and report everything through `conn`."""
    try:
        _run(config, conn, Path(runs_dir), speed, clock, sleep)
    except StopRun:
        _report(conn, status_message("stopped"))
    except Exception as error:
        _report(conn, status_message("error", f"{type(error).__name__}: {error}"))


def _run(
    config: dict[str, Any],
    conn: WorkerConnection,
    runs_dir: Path,
    speed: object,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    cfg = RunConfig.from_dict(config)
    run_dir = _create_run_dir(runs_dir, cfg.name)
    _write_json(run_dir / "config.json", cfg.to_dict())

    sizes = cfg.model.layer_sizes
    activation = cfg.model.activation
    fleet = RacingFleet(
        cfg.population,
        track=cfg.track,
        model=cfg.model,
        max_steps=cfg.max_steps,
        ray_range=cfg.ray_range,
    )
    rng = np.random.default_rng(cfg.seed)
    population = reference.init_population(rng, cfg.population, sizes, cfg.init_scale)
    view = StreamView(
        conn.send,
        lambda: _drain(conn),
        clock=clock,
        sleep=sleep,
        speed=speed,
    )
    conn.send(status_message("running"))

    best_weights = population[0].copy()
    best_fitness = -np.inf
    generation = 0
    while cfg.generations is None or generation < cfg.generations:
        view.current_generation = generation
        result = run_generation(
            fleet,
            population,
            lambda weights, observation: reference.forward(weights, observation, sizes, activation),
            view=view,
            inspect=lambda weights, observation: reference.inspect(
                weights, observation, sizes, activation
            ),
            label=f"Лаборатория, поколение {generation + 1}",
        )
        scores = cfg.fitness.score(result)
        message = build_gen_message(
            generation,
            result,
            scores,
            {
                "mutation_rate": cfg.mutation_rate,
                "mutation_scale": cfg.mutation_scale,
                "elite": cfg.elite,
            },
        )

        top = int(np.argmax(scores))
        if scores[top] > best_fitness:
            best_fitness = float(scores[top])
            best_weights = population[top].copy()
        _write_json(
            run_dir / "best_weights.json",
            {
                "sizes": sizes,
                "activation": activation,
                "generation": generation,
                "fitness": best_fitness,
                "weights": best_weights.tolist(),
            },
        )
        _append_history(run_dir / "history.jsonl", message)
        conn.send(message)

        updates = list(view.pending_updates)
        view.pending_updates.clear()
        cfg = _apply_updates(cfg, updates, conn)
        population = reference.next_generation(
            population,
            scores,
            rng,
            reference.EvolutionParams(
                population=cfg.population,
                elite=cfg.elite,
                mutation_rate=cfg.mutation_rate,
                mutation_scale=cfg.mutation_scale,
                init_scale=cfg.init_scale,
            ),
        )
        generation += 1

    conn.send(status_message("finished"))


def _apply_updates(
    cfg: RunConfig, updates: list[dict[str, Any]], conn: WorkerConnection
) -> RunConfig:
    """Apply allowed parameter changes; each rejected value produces a notice and is skipped."""
    for params in updates:
        for key, value in params.items():
            if key not in UPDATABLE:
                conn.send(notice_message(f"параметр {key} нельзя менять во время запуска"))
                continue
            try:
                cfg = replace(cfg, **{key: value})
            except ValueError as error:
                conn.send(notice_message(f"параметр {key} не применён: {error}"))
    return cfg


def _drain(conn: WorkerConnection) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    while conn.poll(0):
        commands.append(conn.recv())
    return commands


def _report(conn: WorkerConnection, message: dict[str, Any]) -> None:
    """Send a final status. If the channel is already closed, the manager is gone and there is
    nobody left to tell, so the failure to send is the only thing dropped here."""
    try:
        conn.send(message)
    except (OSError, EOFError):
        return


def _create_run_dir(runs_dir: Path, name: str) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _UNSAFE_NAME_CHARS.sub("_", name).strip() or "run"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = runs_dir / f"{safe_name}-{stamp}"
    counter = 2
    while candidate.exists():
        candidate = runs_dir / f"{safe_name}-{stamp}-{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _append_history(path: Path, message: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(message, ensure_ascii=False) + "\n")
