"""StreamView: the worker's stand-in for a fleet view. It streams frames, paces the simulation,
and applies commands from the user between steps.

`run_generation` calls `draw(fleet, lines, network)` after every step, so the view decides how
fast the run goes (speed and pause) and when to stop it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rl_fun.lab.protocol import build_frame, notice_message, status_message
from rl_fun.racing.fleet import RacingFleet

SPEEDS: tuple[object, ...] = (1, 2, 4, 8, "max")

Send = Callable[[dict[str, Any]], None]
Poll = Callable[[], list[dict[str, Any]]]


class StopRun(Exception):
    """Raised from `draw` when the user asked to stop the run."""


def _check_speed(value: object) -> None:
    if isinstance(value, bool) or value not in SPEEDS:
        raise ValueError(f"скорость должна быть одной из 1, 2, 4, 8 или max, получено {value!r}")


class StreamView:
    """Sends frames at most `fps` times per second, paces steps to `speed`, and obeys commands.

    `send` receives messages; `poll` returns the commands received since the last call without
    blocking. Parameter updates are collected in `pending_updates` for the worker to apply
    between generations.
    """

    def __init__(
        self,
        send: Send,
        poll: Poll,
        *,
        fps: float = 30.0,
        dt: float = 1 / 30,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        speed: object = "max",
    ) -> None:
        if not fps > 0 or not dt > 0:
            raise ValueError("fps and dt must be positive")
        _check_speed(speed)
        self.speed = speed
        self.current_generation = 0
        self.pending_updates: list[dict[str, Any]] = []
        self._send = send
        self._poll = poll
        self._frame_interval = 1.0 / fps
        self._dt = dt
        self._clock = clock
        self._sleep = sleep
        self._paused = False
        self._last_frame_at: float | None = None
        self._due: float | None = None

    def draw(self, fleet: RacingFleet, lines: list[str], network: Any) -> None:
        """Handle pending commands, hold while paused, send a frame if due, then pace the step."""
        self._handle_commands()
        while self._paused:
            self._sleep(self._frame_interval)
            self._handle_commands()
            if self._paused:
                self._maybe_send_frame(fleet, network)
        self._maybe_send_frame(fleet, network)
        self._pace()

    def _handle_commands(self) -> None:
        for command in self._poll():
            self._apply(command)

    def _apply(self, command: dict[str, Any]) -> None:
        name = command.get("cmd")
        if name == "pause":
            if not self._paused:
                self._paused = True
                self._send(status_message("paused"))
        elif name == "resume":
            if self._paused:
                self._paused = False
                self._due = None
                self._send(status_message("running"))
        elif name == "stop":
            raise StopRun()
        elif name == "speed":
            value = command.get("value")
            _check_speed(value)
            self.speed = value
            self._due = None
        elif name == "update":
            self.pending_updates.append(dict(command.get("params", {})))
        else:
            self._send(notice_message(f"неизвестная команда: {name!r}"))

    def _maybe_send_frame(self, fleet: RacingFleet, network: Any) -> None:
        now = self._clock()
        if self._last_frame_at is not None:
            if now - self._last_frame_at < self._frame_interval - 1e-9:
                return
        self._send(build_frame(fleet, self.current_generation, network))
        self._last_frame_at = now

    def _pace(self) -> None:
        if self.speed == "max":
            return
        interval = self._dt / float(self.speed)  # type: ignore[arg-type]
        now = self._clock()
        self._due = (self._due if self._due is not None else now) + interval
        wait = self._due - now
        if wait > 0:
            self._sleep(wait)
        else:
            self._due = now
