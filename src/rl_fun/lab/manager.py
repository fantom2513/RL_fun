"""Run manager: owns the worker processes of the lab and the state the server shows.

Each run is one `spawn`ed worker process connected by a `multiprocessing` Pipe. A daemon reader
thread per run turns the worker's messages into run state (status, generation history, notices and
the latest frame). Everything is guarded by one condition variable, which also wakes the SSE
subscribers (`subscribe`).

Delivery guarantees for subscribers: `gen`, `status` and `notice` messages are kept in an
append-only log and every subscriber sees all of them in order; frames are not logged, only the
latest one is kept, so a slow subscriber skips frames but never other messages.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from rl_fun.lab.config import RunConfig
from rl_fun.lab.protocol import status_message
from rl_fun.lab.stream_view import SPEEDS
from rl_fun.lab.worker import run_worker
from rl_fun.racing.track import TRACKS_ENV

TERMINAL_STATUSES = ("finished", "stopped", "error")
UNEXPECTED_EXIT = "процесс запуска завершился неожиданно"
_COMMANDS = ("pause", "resume", "stop", "speed", "update")
_POLL_INTERVAL = 0.2
ARCHIVE_LIMIT = 30
"""How many run folders of earlier sessions are brought back (the newest ones)."""
DISMISSED = ".dismissed"
"""Marker file of a run folder whose run was deleted in the interface: it is not restored."""

WorkerTarget = Callable[[dict[str, Any], Any, str], None]
Event = tuple[str, dict[str, Any]]


def run_worker_entry(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Process entry point: a module-level function so `spawn` can import it in the child."""
    run_worker(config, conn, runs_dir)


def _validate_command(command: Any) -> None:
    """Check the shape of a command (spec 5.3); the worker validates parameter values itself."""
    if not isinstance(command, dict):
        raise ValueError("команда должна быть объектом с полем cmd")
    name = command.get("cmd")
    if name not in _COMMANDS:
        raise ValueError(f"неизвестная команда {name!r}; допустимы: {', '.join(_COMMANDS)}")
    if name == "speed":
        value = command.get("value")
        if isinstance(value, bool) or value not in SPEEDS:
            raise ValueError(
                f"скорость должна быть одной из 1, 2, 4, 8 или max, получено {value!r}"
            )
    elif name == "update" and not isinstance(command.get("params"), dict):
        raise ValueError("для команды update нужен объект params")


class _Run:
    """State of one run. Mutable fields are guarded by the manager's condition."""

    def __init__(self, run_id: str, config: dict[str, Any], process: Any, conn: Any) -> None:
        self.id = run_id
        self.archived = False
        self.dir_name: str | None = None
        self.name = str(config["name"])
        self.config = config
        self.process = process
        self.conn = conn
        self.send_lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self.status = "running"
        self.error: str | None = None
        self.exit_code: int | None = None
        self.terminal = False
        self.deleted = False
        self.history: list[dict[str, Any]] = []
        self.notices: list[str] = []
        self.events: list[Event] = []
        self.last_frame: dict[str, Any] | None = None
        self.frame_seq = 0


class RunManager:
    """Creates, controls and observes lab runs; safe to use from many threads."""

    def __init__(
        self,
        runs_dir: str | Path,
        *,
        worker_target: WorkerTarget = run_worker_entry,
        stop_timeout: float = 3.0,
    ) -> None:
        self._runs_dir = Path(runs_dir)
        self._worker_target = worker_target
        self._stop_timeout = stop_timeout
        self._context = multiprocessing.get_context("spawn")
        self._cond = threading.Condition()
        self._shutdown_lock = threading.Lock()
        self._runs: dict[str, _Run] = {}
        self._counter = 0
        self._closed = False
        os.environ[TRACKS_ENV] = str(self.tracks_dir)
        self._load_archive()

    @property
    def tracks_dir(self) -> Path:
        """Folder of the tracks drawn in the lab."""
        return self._runs_dir / "tracks"

    # ---- public API --------------------------------------------------------------------------

    def create(self, config: dict[str, Any]) -> str:
        """Validate `config`, start a worker process for it and return the run id."""
        # saved tracks are found through the environment, which the worker process inherits
        os.environ[TRACKS_ENV] = str(self.tracks_dir)
        cfg = RunConfig.from_dict(config)
        with self._cond:
            if self._closed:
                raise RuntimeError("менеджер запусков остановлен")
            self._counter += 1
            run_id = f"r{self._counter}"
        normalized = cfg.to_dict()
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._worker_target,
            args=(normalized, child, str(self._runs_dir)),
            name=f"lab-{run_id}",
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()  # only the child holds its end, so its death shows up as EOF here
        run = _Run(run_id, normalized, process, parent)
        run.reader = threading.Thread(
            target=self._read, args=(run,), name=f"lab-reader-{run_id}", daemon=True
        )
        run.reader.start()
        with self._cond:
            closed = self._closed
            if not closed:
                self._runs[run_id] = run
        if closed:
            self._stop_runs([run])
            raise RuntimeError("менеджер запусков остановлен")
        return run_id

    def list(self) -> list[dict[str, Any]]:
        """One summary row per run: id, name, status, generations, learner and best progress."""
        with self._cond:
            return [
                {
                    "id": run.id,
                    "name": run.name,
                    "status": run.status,
                    "gen": len(run.history),
                    "learner": run.config.get("learner", "evolution"),
                    "archived": run.archived,
                    "best": max((entry["best"] for entry in run.history), default=None),
                }
                for run in self._runs.values()
            ]

    def get(self, run_id: str) -> dict[str, Any]:
        """Config, status, generation history, notices and error info of one run."""
        with self._cond:
            run = self._lookup(run_id)
            return {
                "id": run.id,
                "name": run.name,
                "config": dict(run.config),
                "status": run.status,
                "history": list(run.history),
                "notices": list(run.notices),
                "error": run.error,
                "exit_code": run.exit_code,
            }

    def process(self, run_id: str) -> Any:
        """The worker's process object (for diagnostics and tests)."""
        with self._cond:
            return self._lookup(run_id).process

    def command(self, run_id: str, command: dict[str, Any]) -> None:
        """Send a command (spec 5.3) to the run's worker. Commands to a finished run are no-ops."""
        with self._cond:
            run = self._lookup(run_id)
            finished = run.terminal
        _validate_command(command)
        if finished:
            return
        self._send(run, command)

    def delete(self, run_id: str) -> None:
        """Stop the run's process and forget the run; its subscribers are woken and end."""
        with self._cond:
            run = self._lookup(run_id)
            del self._runs[run_id]
            run.deleted = True
            self._cond.notify_all()
        self._stop_runs([run])
        self._dismiss(run)

    def subscribe(
        self, run_id: str, from_index: int = 0, *, idle_timeout: float | None = None
    ) -> Iterator[Event | tuple[str, None]]:
        """Events for SSE as `(kind, payload)`: the log of generations/status/notices (skipping
        the first `from_index` generations), then the latest frame, then everything new.

        Ends after the terminal status was delivered, or when the run is deleted / the manager
        shuts down and nothing is left to deliver. Raises KeyError right away for unknown ids.

        With `idle_timeout` set, whenever that many seconds pass with nothing to deliver the
        generator yields the sentinel `("idle", None)` so the caller gets a chance to notice that
        it should stop (otherwise it would stay parked on a quiet run). Without it, no sentinel.
        """
        with self._cond:
            run = self._lookup(run_id)
        return self._events(run, max(0, int(from_index)), idle_timeout)

    def shutdown(self) -> None:
        """Stop every worker (gracefully first), wait for them and the reader threads.

        Safe to call repeatedly.
        """
        with self._shutdown_lock:
            with self._cond:
                self._closed = True
                runs = [*self._runs.values()]
                self._cond.notify_all()
            self._stop_runs(runs)

    # ---- subscribers -------------------------------------------------------------------------

    def _events(
        self, run: _Run, from_index: int, idle_timeout: float | None
    ) -> Iterator[Event | tuple[str, None]]:
        position = 0
        seen_frame = 0
        gens_seen = 0
        while True:
            batch: list[Event | tuple[str, None]] = []
            done = False
            idle_deadline = None if idle_timeout is None else time.monotonic() + idle_timeout
            with self._cond:
                while True:
                    has_events = position < len(run.events)
                    has_frame = run.frame_seq > seen_frame
                    if has_events or has_frame:
                        break
                    if run.deleted or self._closed:
                        return
                    if idle_deadline is None:
                        self._cond.wait(timeout=1.0)
                        continue
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)
                if not (has_events or has_frame):
                    batch.append(("idle", None))
                    frame = None
                else:
                    frame = run.last_frame if has_frame else None
                seen_frame = run.frame_seq
                for kind, payload in run.events[position:]:
                    if kind == "gen":
                        gens_seen += 1
                        if gens_seen <= from_index:
                            continue
                    terminal = kind == "status" and payload["status"] in TERMINAL_STATUSES
                    if terminal and frame is not None:
                        batch.append(("frame", frame))
                        frame = None
                    batch.append((kind, payload))
                    if terminal:
                        done = True
                        break
                position = len(run.events)
                if frame is not None:
                    batch.append(("frame", frame))
            yield from batch
            if done:
                return

    # ---- reader thread -----------------------------------------------------------------------

    def _read(self, run: _Run) -> None:
        conn = run.conn
        while True:
            try:
                if conn.poll(_POLL_INTERVAL):
                    message = conn.recv()
                elif not run.process.is_alive() and not conn.poll(0):
                    break
                else:
                    continue
            except (EOFError, OSError):
                break
            self._handle(run, message)
        run.process.join(5.0)
        with self._cond:
            run.exit_code = run.process.exitcode
            if not run.terminal:
                if run.deleted or self._closed:
                    self._apply_status(run, status_message("stopped"))
                else:
                    self._apply_status(run, status_message("error", UNEXPECTED_EXIT))
            self._cond.notify_all()

    def _handle(self, run: _Run, message: Any) -> None:
        if not isinstance(message, dict):
            return
        kind = message.get("t")
        with self._cond:
            if kind == "frame":
                run.last_frame = message
                run.frame_seq += 1
            elif kind == "gen":
                run.history.append(message)
                run.events.append(("gen", message))
            elif kind == "status":
                self._apply_status(run, message)
            elif kind == "notice":
                run.notices.append(str(message.get("text", "")))
                run.events.append(("notice", message))
            elif kind == "meta":
                name = str(message.get("dir", ""))
                run.dir_name = name if name and Path(name).name == name else None
            else:
                return
            self._cond.notify_all()

    @staticmethod
    def _apply_status(run: _Run, message: dict[str, Any]) -> None:
        """Record a status message. Caller holds the condition."""
        status = message["status"]
        run.status = status
        if status == "error":
            run.error = message.get("message")
        if status in TERMINAL_STATUSES:
            run.terminal = True
        run.events.append(("status", message))

    # ---- helpers -----------------------------------------------------------------------------

    def _load_archive(self) -> None:
        """Bring back the runs of earlier sessions from their folders: the config, the history of
        generations and a final status. They cannot be continued, but they can be looked at,
        compared and restarted with their settings."""
        root = self._runs_dir
        if not root.is_dir():
            return
        folders = [
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name != "tracks"
            and (path / "config.json").is_file()
            and not (path / DISMISSED).exists()
        ]
        folders.sort(key=lambda path: (path / "config.json").stat().st_mtime)
        for path in folders[-ARCHIVE_LIMIT:]:
            run = self._read_archived(path)
            if run is not None:
                self._runs[run.id] = run

    def _read_archived(self, path: Path) -> _Run | None:
        try:
            config = RunConfig.from_dict(json.loads((path / "config.json").read_text("utf-8")))
        except (OSError, ValueError, TypeError):
            return None
        history: list[dict[str, Any]] = []
        try:
            lines = (path / "history.jsonl").read_text("utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                message = json.loads(line)
            except ValueError:
                continue  # the process died in the middle of writing this line
            if isinstance(message, dict) and message.get("t") == "gen":
                history.append(message)
        self._counter += 1
        run = _Run(f"r{self._counter}", config.to_dict(), None, None)
        run.archived = True
        run.dir_name = path.name
        run.history = history
        done = config.generations is not None and len(history) >= config.generations
        status = "finished" if done else "stopped"
        run.status = status
        run.terminal = True
        run.events = [("gen", message) for message in history]
        run.events.append(("status", status_message(status)))
        return run

    def _dismiss(self, run: _Run) -> None:
        """Mark the folder of a deleted run so a later session skips it. The files stay."""
        if run.dir_name is None:
            return
        try:
            (self._runs_dir / run.dir_name / DISMISSED).write_text("", encoding="utf-8")
        except OSError:
            return

    def _lookup(self, run_id: str) -> _Run:
        try:
            return self._runs[run_id]
        except KeyError:
            raise KeyError(f"неизвестный запуск: {run_id}") from None

    @staticmethod
    def _send(run: _Run, command: dict[str, Any]) -> None:
        """Deliver a command; if the worker is already gone there is nobody to tell."""
        with run.send_lock:
            try:
                run.conn.send(command)
            except (OSError, ValueError, EOFError):
                return

    def _stop_runs(self, runs: list[_Run]) -> None:
        """Ask the workers to stop, terminate the ones that do not, and reap everything."""
        runs = [run for run in runs if run.process is not None]
        for run in runs:
            self._send(run, {"cmd": "stop"})
        deadline = time.monotonic() + self._stop_timeout
        for run in runs:
            run.process.join(max(0.0, deadline - time.monotonic()))
        for run in runs:
            if run.process.is_alive():
                run.process.terminate()
                run.process.join(2.0)
            if run.process.is_alive():
                run.process.kill()
                run.process.join()
        for run in runs:
            if run.reader is not None:
                run.reader.join(10.0)
                if run.reader.is_alive():
                    continue
            with run.send_lock:
                try:
                    run.conn.close()
                except OSError:
                    pass
