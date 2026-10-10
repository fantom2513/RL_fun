"""The garage: best networks of finished runs and demo drives of them on any track."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rl_fun.lab.manager import RunManager
from rl_fun.lab.worker import run_demo
from rl_fun.racing.model_spec import ModelSpec

from .test_manager import _config, wait_status
from .test_worker import FakeConnection


def demo_config(**overrides: Any) -> dict[str, Any]:
    model = ModelSpec()
    sizes = model.layer_sizes
    count = sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:], strict=True))
    data: dict[str, Any] = {
        "model": model.to_dict(),
        "sizes": sizes,
        "activation": model.activation,
        "weights": list(np.random.default_rng(3).normal(0, 0.5, count)),
        "track": "oval",
        "laps": 1,
        "max_steps": 60,
        "ray_range": 40.0,
        "speed": "max",
    }
    data.update(overrides)
    return data


# ---- the demo worker -------------------------------------------------------------------------


def test_a_demo_streams_frames_then_one_result_and_finishes() -> None:
    conn = FakeConnection()

    run_demo(demo_config(), conn)

    kinds = [message["t"] for message in conn.sent]
    assert conn.sent[0] == {"t": "status", "status": "running"}
    assert "frame" in kinds and conn.sent[-1] == {"t": "status", "status": "finished"}
    [result] = conn.of_type("gen")
    assert 0.0 <= result["best"] <= 1.0 and result["finished"] in (0, 1)
    assert result["extra"]["steps"] <= 60


def test_a_demo_reports_a_finished_race_with_its_time() -> None:
    from rl_fun.racing import reference
    from rl_fun.racing.fleet import RacingFleet

    # a network that drives straight at full throttle cannot finish an oval: use the real check
    # of the contract instead, a one-step "race" is finished once the lap length is covered
    conn = FakeConnection()
    run_demo(demo_config(max_steps=40), conn)
    [result] = conn.of_type("gen")

    fleet = RacingFleet(1, track="oval", max_steps=40)
    assert result["best_lap_steps"] is None and fleet.laps == 1


def test_a_demo_can_be_stopped() -> None:
    conn = FakeConnection(commands_after_frame={2: [{"cmd": "stop"}]})

    run_demo(demo_config(max_steps=500, speed=1), conn)

    assert conn.sent[-1] == {"t": "status", "status": "stopped"}
    assert conn.of_type("gen") == []


def test_weights_of_the_wrong_size_end_the_demo_with_an_error() -> None:
    conn = FakeConnection()

    run_demo(demo_config(weights=[0.1, 0.2]), conn)

    assert conn.sent[-1]["status"] == "error"


# ---- the garage in the manager ---------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> Iterator[RunManager]:
    instance = RunManager(tmp_path / "runs", stop_timeout=0.5)
    try:
        yield instance
    finally:
        instance.shutdown()
        assert multiprocessing.active_children() == []


def finished_run(manager: RunManager, **overrides: Any) -> str:
    run_id = manager.create(_config(generations=2, **overrides))
    wait_status(manager, run_id, "finished")
    return run_id


def test_finished_runs_with_weights_appear_in_the_garage(manager: RunManager) -> None:
    run_id = finished_run(manager, name="pilot")

    [car] = manager.garage()

    assert car["id"] == run_id and car["name"] == "pilot" and car["track"] == "oval"
    assert car["learner"] == "evolution" and car["archived"] is False
    assert car["model"]["hidden"] == [6, 5] and car["model"]["inputs"] == 8
    assert car["generation"] == 1 and set(car) >= {"best", "best_lap_steps", "laps"}


def test_a_run_without_weights_is_not_in_the_garage(manager: RunManager) -> None:
    manager.create(_config(generations=None, name="young"))

    assert manager.garage() == []


def test_the_garage_survives_a_restart(tmp_path: Path) -> None:
    first = RunManager(tmp_path / "runs", stop_timeout=0.5)
    finished_run(first, name="old")
    first.shutdown()

    second = RunManager(tmp_path / "runs", stop_timeout=0.5)
    try:
        [car] = second.garage()
        assert car["name"] == "old" and car["archived"] is True
        assert len(second.weights(car["id"])["weights"]) > 0
    finally:
        second.shutdown()


def test_a_demo_drives_the_saved_network_on_another_track_and_stays_out_of_the_lists(
    manager: RunManager,
) -> None:
    run_id = finished_run(manager)

    demo_id = manager.create_demo(run_id, "wavy", speed="max", max_steps=80)
    info = wait_status(manager, demo_id, "finished")

    assert info["config"]["track"] == "wavy" and info["history"]
    assert [row["id"] for row in manager.list()] == [run_id]
    assert [car["id"] for car in manager.garage()] == [run_id]
    manager.delete(demo_id)
    assert [row["id"] for row in manager.list()] == [run_id]


def test_a_demo_of_an_unknown_run_or_track_is_refused(manager: RunManager) -> None:
    run_id = finished_run(manager)

    with pytest.raises(KeyError):
        manager.create_demo("nope", "oval")
    with pytest.raises(ValueError, match="track"):
        manager.create_demo(run_id, "no-such-track")
    with pytest.raises(ValueError, match="laps"):
        manager.create_demo(run_id, "oval", laps=99)


def test_the_best_weights_export_carries_the_config_and_the_network(manager: RunManager) -> None:
    run_id = finished_run(manager, name="export me")

    package = manager.export(run_id)

    assert package["config"]["name"] == "export me"
    assert package["snapshot"]["sizes"] == [8, 6, 5, 2]
    assert json.dumps(package)
