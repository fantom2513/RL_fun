import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from rl_fun.lab.manager import RunManager

from . import manager_targets as targets

TIMEOUT = 90.0
TERMINAL = ("finished", "stopped", "error")


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "t",
        "track": "oval",
        "population": 8,
        "elite": 2,
        "max_steps": 60,
        "generations": 2,
    }
    config.update(overrides)
    return config


def wait_for(condition: Callable[[], Any], timeout: float = TIMEOUT) -> Any:
    """Poll `condition` until it returns something truthy; fail after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        value = condition()
        if value:
            return value
        if time.monotonic() > deadline:
            pytest.fail("condition was not met in time")
        time.sleep(0.05)


def wait_status(manager: RunManager, run_id: str, *statuses: str) -> dict[str, Any]:
    return wait_for(
        lambda: (info := manager.get(run_id))["status"] in statuses and info or None
    )


def wait_dead(manager: RunManager, run_id: str) -> None:
    process = manager.process(run_id)
    process.join(TIMEOUT)
    assert not process.is_alive()


def consume(
    manager: RunManager, run_id: str, *, from_index: int = 0
) -> tuple[list[tuple[str, dict[str, Any]]], threading.Thread]:
    """Consume `subscribe` in a thread; returns the (live) event list and the thread."""
    events: list[tuple[str, dict[str, Any]]] = []

    def run() -> None:
        for event in manager.subscribe(run_id, from_index=from_index):
            events.append(event)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return events, thread


@pytest.fixture
def manager(tmp_path: Path) -> Iterator[RunManager]:
    instance = RunManager(tmp_path / "runs", stop_timeout=0.5)
    try:
        yield instance
    finally:
        instance.shutdown()
        # No zombie or leftover child process may survive a test.
        assert multiprocessing.active_children() == []


def make_manager(tmp_path: Path, target: Callable[..., None]) -> RunManager:
    return RunManager(tmp_path / "runs", worker_target=target, stop_timeout=0.5)


@pytest.fixture
def idle_manager(tmp_path: Path) -> Iterator[RunManager]:
    instance = make_manager(tmp_path, targets.idle_target)
    try:
        yield instance
    finally:
        instance.shutdown()
        assert multiprocessing.active_children() == []


# ---- lifecycle with real workers -------------------------------------------------------------


def test_create_runs_to_finish_and_records_history(manager: RunManager, tmp_path: Path) -> None:
    run_id = manager.create(_config(name="alpha", generations=2))

    info = wait_status(manager, run_id, *TERMINAL)

    assert run_id == "r1"
    assert info["status"] == "finished"
    assert [entry["gen"] for entry in info["history"]] == [0, 1]
    assert info["name"] == "alpha"
    assert info["config"]["population"] == 8
    assert [row["id"] for row in manager.list()] == ["r1"]
    row = manager.list()[0]
    assert (row["name"], row["status"], row["gen"]) == ("alpha", "finished", 2)
    assert row["best"] == max(entry["best"] for entry in info["history"])
    artifacts = [path for path in (tmp_path / "runs").iterdir() if path.name.startswith("alpha-")]
    assert len(artifacts) == 1
    assert (artifacts[0] / "config.json").is_file()


def test_two_runs_complete_independently(manager: RunManager) -> None:
    first = manager.create(_config(name="a", seed=1, generations=2))
    second = manager.create(_config(name="b", seed=2, generations=3))

    info_a = wait_status(manager, first, *TERMINAL)
    info_b = wait_status(manager, second, *TERMINAL)

    assert first != second
    assert (info_a["status"], len(info_a["history"])) == ("finished", 2)
    assert (info_b["status"], len(info_b["history"])) == ("finished", 3)
    assert {row["id"] for row in manager.list()} == {first, second}


def test_pause_resume_stop_cycle(manager: RunManager) -> None:
    run_id = manager.create(_config(generations=None, max_steps=200))
    wait_for(lambda: manager.get(run_id)["history"])

    manager.command(run_id, {"cmd": "pause"})
    wait_status(manager, run_id, "paused")
    manager.command(run_id, {"cmd": "resume"})
    wait_status(manager, run_id, "running")
    manager.command(run_id, {"cmd": "stop"})
    info = wait_status(manager, run_id, *TERMINAL)

    assert info["status"] == "stopped"
    wait_dead(manager, run_id)


def test_worker_receives_valid_commands(idle_manager: RunManager) -> None:
    run_id = idle_manager.create(_config())

    idle_manager.command(run_id, {"cmd": "speed", "value": 4})
    idle_manager.command(run_id, {"cmd": "update", "params": {"elite": 3}})

    notices = wait_for(lambda: len(idle_manager.get(run_id)["notices"]) >= 2 and True)
    assert notices
    texts = idle_manager.get(run_id)["notices"]
    assert texts == [
        '{"cmd": "speed", "value": 4}',
        '{"cmd": "update", "params": {"elite": 3}}',
    ]


# ---- validation ------------------------------------------------------------------------------


def test_create_rejects_invalid_config_without_starting_a_process(manager: RunManager) -> None:
    with pytest.raises(ValueError, match="population"):
        manager.create(_config(population=1))

    assert manager.list() == []
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize(
    "bad_command",
    [
        {},
        {"cmd": "explode"},
        {"cmd": "speed"},
        {"cmd": "speed", "value": 3},
        {"cmd": "speed", "value": True},
        {"cmd": "update"},
        {"cmd": "update", "params": [1]},
        "pause",
    ],
)
def test_command_with_bad_shape_is_a_value_error(
    idle_manager: RunManager, bad_command: Any
) -> None:
    run_id = idle_manager.create(_config())

    with pytest.raises(ValueError):
        idle_manager.command(run_id, bad_command)

    assert idle_manager.get(run_id)["notices"] == []


def test_unknown_run_id_is_a_key_error(idle_manager: RunManager) -> None:
    with pytest.raises(KeyError):
        idle_manager.command("r99", {"cmd": "pause"})
    with pytest.raises(KeyError):
        idle_manager.get("r99")
    with pytest.raises(KeyError):
        idle_manager.delete("r99")
    with pytest.raises(KeyError):
        idle_manager.subscribe("r99")


def test_runs_dir_is_passed_to_the_worker(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, targets.touching_target)
    try:
        run_id = manager.create(_config())
        wait_status(manager, run_id, "finished")
        assert (tmp_path / "runs" / "touched.txt").is_file()
    finally:
        manager.shutdown()


# ---- delete, crash, shutdown -----------------------------------------------------------------


def test_delete_running_real_run_kills_process_and_removes_it(manager: RunManager) -> None:
    run_id = manager.create(_config(generations=None, max_steps=200))
    wait_for(lambda: manager.get(run_id)["history"])
    process = manager.process(run_id)

    manager.delete(run_id)

    assert manager.list() == []
    assert not process.is_alive()
    with pytest.raises(KeyError):
        manager.get(run_id)


def test_delete_terminates_a_worker_that_ignores_stop(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, targets.stubborn_target)
    try:
        run_id = manager.create(_config())
        wait_status(manager, run_id, "running")
        process = manager.process(run_id)

        manager.delete(run_id)

        assert not process.is_alive()
        assert manager.list() == []
    finally:
        manager.shutdown()
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize(
    ("target", "exit_code"),
    [(targets.raising_target, 1), (targets.exiting_target, 3)],
)
def test_worker_that_dies_silently_ends_in_error(
    tmp_path: Path, target: Callable[..., None], exit_code: int
) -> None:
    manager = make_manager(tmp_path, target)
    try:
        run_id = manager.create(_config())

        info = wait_status(manager, run_id, *TERMINAL)

        assert info["status"] == "error"
        assert info["error"] == "процесс запуска завершился неожиданно"
        assert info["exit_code"] == exit_code
        assert manager.list()[0]["status"] == "error"
    finally:
        manager.shutdown()
    assert multiprocessing.active_children() == []


def test_one_crashed_run_does_not_affect_another(idle_manager: RunManager) -> None:
    healthy = idle_manager.create(_config())
    wait_status(idle_manager, healthy, "running")
    idle_manager._worker_target = targets.exiting_target  # type: ignore[attr-defined]
    crashed = idle_manager.create(_config())

    wait_status(idle_manager, crashed, "error")

    assert idle_manager.get(healthy)["status"] == "running"
    assert idle_manager.process(healthy).is_alive()


def test_shutdown_stops_everything_and_is_idempotent(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, targets.idle_target)
    ids = [manager.create(_config()) for _ in range(2)]
    stubborn = RunManager(
        tmp_path / "runs2", worker_target=targets.stubborn_target, stop_timeout=0.5
    )
    stubborn_id = stubborn.create(_config())
    processes = [manager.process(i) for i in ids] + [stubborn.process(stubborn_id)]
    for run_id in ids:
        wait_status(manager, run_id, "running")
    wait_status(stubborn, stubborn_id, "running")

    manager.shutdown()
    stubborn.shutdown()
    manager.shutdown()
    stubborn.shutdown()

    assert all(not process.is_alive() for process in processes)
    assert multiprocessing.active_children() == []
    assert {manager.get(i)["status"] for i in ids} == {"stopped"}
    assert stubborn.get(stubborn_id)["status"] == "stopped"
    with pytest.raises(RuntimeError):
        manager.create(_config())


def test_shutdown_ends_subscribers(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, targets.stubborn_target)
    try:
        run_id = manager.create(_config())
        wait_status(manager, run_id, "running")
        events, thread = consume(manager, run_id)
        wait_for(lambda: events)

        manager.shutdown()

        thread.join(TIMEOUT)
        assert not thread.is_alive()
    finally:
        manager.shutdown()


# ---- subscribe -------------------------------------------------------------------------------


def test_late_subscriber_gets_history_then_new_events_then_terminal_status(
    manager: RunManager,
) -> None:
    run_id = manager.create(_config(generations=None, max_steps=200))
    wait_for(lambda: manager.get(run_id)["history"])

    events, thread = consume(manager, run_id)
    wait_for(lambda: len([e for e in events if e[0] == "gen"]) >= 3)
    manager.command(run_id, {"cmd": "stop"})
    thread.join(TIMEOUT)

    assert not thread.is_alive()
    kinds = [kind for kind, _ in events]
    gens = [payload["gen"] for kind, payload in events if kind == "gen"]
    assert gens == list(range(len(gens)))
    assert len(gens) >= 3
    assert "frame" in kinds
    assert events[-1] == ("status", {"t": "status", "status": "stopped"})
    assert kinds.count("status") >= 2  # running ... stopped
    first_gen = kinds.index("gen")
    assert all(kind in ("status", "notice") for kind in kinds[:first_gen])


def test_subscribe_from_index_skips_earlier_generations(manager: RunManager) -> None:
    run_id = manager.create(_config(generations=3))
    wait_status(manager, run_id, "finished")

    events = list(manager.subscribe(run_id, from_index=2))

    assert [p["gen"] for k, p in events if k == "gen"] == [2]
    assert events[-1][1]["status"] == "finished"


def test_subscriber_to_finished_run_terminates_after_replay(manager: RunManager) -> None:
    run_id = manager.create(_config(generations=2))
    wait_status(manager, run_id, "finished")

    events = list(manager.subscribe(run_id))

    assert [p["gen"] for k, p in events if k == "gen"] == [0, 1]
    assert events[-1] == ("status", {"t": "status", "status": "finished"})
    frames = [p for k, p in events if k == "frame"]
    assert len(frames) == 1
    assert events.index(("frame", frames[0])) < len(events) - 1  # frame precedes the terminal


def test_slow_subscriber_sees_only_latest_frame_but_every_other_message(
    tmp_path: Path,
) -> None:
    manager = make_manager(tmp_path, targets.burst_target)
    try:
        run_id = manager.create(_config())
        wait_status(manager, run_id, "finished")

        events = list(manager.subscribe(run_id))

        frames = [p for k, p in events if k == "frame"]
        assert [p["step"] for p in frames] == [199]
        assert [p["gen"] for k, p in events if k == "gen"] == [0, 1]
        assert [p["text"] for k, p in events if k == "notice"] == ["hello"]
        assert events[-1][0] == "status"
        assert events[-1][1]["status"] == "finished"
    finally:
        manager.shutdown()


def test_live_slow_subscriber_frames_never_go_back_and_gens_are_complete(
    tmp_path: Path,
) -> None:
    manager = make_manager(tmp_path, targets.burst_target)
    try:
        run_id = manager.create(_config())
        seen: list[tuple[str, dict[str, Any]]] = []
        for event in manager.subscribe(run_id):
            seen.append(event)
            time.sleep(0.02)  # slower than the producer

        steps = [p["step"] for k, p in seen if k == "frame"]
        assert steps == sorted(steps)
        assert steps[-1] == 199
        assert len(steps) < 200
        assert [p["gen"] for k, p in seen if k == "gen"] == [0, 1]
    finally:
        manager.shutdown()


def test_two_subscribers_do_not_block_each_other(idle_manager: RunManager) -> None:
    run_id = idle_manager.create(_config())
    wait_status(idle_manager, run_id, "running")
    first, first_thread = consume(idle_manager, run_id)
    second, second_thread = consume(idle_manager, run_id)
    wait_for(lambda: first and second)

    idle_manager.command(run_id, {"cmd": "speed", "value": 2})
    wait_for(lambda: any(k == "notice" for k, _ in first) and any(k == "notice" for k, _ in second))
    idle_manager.command(run_id, {"cmd": "stop"})

    first_thread.join(TIMEOUT)
    second_thread.join(TIMEOUT)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first[-1][1]["status"] == "stopped" == second[-1][1]["status"]


def test_delete_wakes_and_ends_subscribers(idle_manager: RunManager) -> None:
    run_id = idle_manager.create(_config())
    wait_status(idle_manager, run_id, "running")
    events, thread = consume(idle_manager, run_id)
    wait_for(lambda: events)

    idle_manager.delete(run_id)

    thread.join(TIMEOUT)
    assert not thread.is_alive()


def test_crash_is_delivered_to_subscribers(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, targets.exiting_target)
    try:
        run_id = manager.create(_config())

        events = list(manager.subscribe(run_id))

        assert events[-1][0] == "status"
        assert events[-1][1]["status"] == "error"
        assert events[-1][1]["message"] == "процесс запуска завершился неожиданно"
    finally:
        manager.shutdown()


def test_subscribe_idle_timeout_yields_idle_sentinel_while_nothing_happens(
    idle_manager: RunManager,
) -> None:
    run_id = idle_manager.create(_config())
    wait_status(idle_manager, run_id, "running")
    events = idle_manager.subscribe(run_id, idle_timeout=0.05)
    try:
        received = [next(events) for _ in range(4)]
    finally:
        events.close()

    assert received[0] == ("status", {"t": "status", "status": "running"})
    assert received[1:] == [("idle", None)] * 3


def test_subscribe_without_idle_timeout_never_yields_idle(idle_manager: RunManager) -> None:
    run_id = idle_manager.create(_config())
    wait_status(idle_manager, run_id, "running")
    events, thread = consume(idle_manager, run_id)
    wait_for(lambda: events)

    time.sleep(1.5)  # longer than the internal wake-up interval
    idle_manager.command(run_id, {"cmd": "stop"})
    thread.join(TIMEOUT)

    assert not thread.is_alive()
    assert all(kind != "idle" for kind, _ in events)
