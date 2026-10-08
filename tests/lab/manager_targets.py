"""Cheap worker targets for the run-manager tests.

They live in a real module (not in the test file) so a `spawn`ed child can import them by name.
They use only the standard library, so the child starts fast.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def idle_target(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Report `running`, echo every command as a notice, finish with `stopped` on `stop`."""
    conn.send({"t": "status", "status": "running"})
    while True:
        if conn.poll(0.05):
            command = conn.recv()
            conn.send({"t": "notice", "text": json.dumps(command, sort_keys=True)})
            if command.get("cmd") == "stop":
                conn.send({"t": "status", "status": "stopped"})
                return


def stubborn_target(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Report `running`, then ignore every command forever; only termination ends it."""
    conn.send({"t": "status", "status": "running"})
    while True:
        time.sleep(0.05)


def raising_target(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Die with an exception before saying anything."""
    raise RuntimeError("boom")


def exiting_target(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Exit hard with code 3 before saying anything."""
    os._exit(3)


def burst_target(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Send 200 frames with three generation messages and a notice mixed in, then finish."""
    conn.send({"t": "status", "status": "running"})
    for step in range(200):
        conn.send({"t": "frame", "gen": step // 70, "step": step})
        if step % 70 == 69:
            conn.send({"t": "gen", "gen": step // 70, "best": 0.1 * (step // 70)})
        if step == 100:
            conn.send({"t": "notice", "text": "hello"})
    conn.send({"t": "status", "status": "finished"})


def touching_target(config: dict[str, Any], conn: Any, runs_dir: str) -> None:
    """Record that it ran by writing into `runs_dir`, then finish."""
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    (Path(runs_dir) / "touched.txt").write_text("1", encoding="utf-8")
    conn.send({"t": "status", "status": "finished"})
