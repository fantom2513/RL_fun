"""Runs of earlier sessions: the manager lists them again from the run folders on disk."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rl_fun.lab.config import RunConfig
from rl_fun.lab.manager import RunManager

from .test_manager import _config, wait_status


def make_run_dir(
    runs: Path, name: str, *, generations: int | None = 3, gens: int = 2, learner: str = "evolution"
) -> Path:
    folder = runs / f"{name}-20261009-101500"
    folder.mkdir(parents=True)
    config = RunConfig(
        name=name, track="oval", generations=generations, learner=learner, laps=2
    ).to_dict()
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    lines = [
        json.dumps(
            {
                "t": "gen",
                "gen": index,
                "best": 0.2 * (index + 1),
                "mean": 0.1,
                "finished": 0,
                "best_fitness": 1.0,
                "best_lap_steps": None,
                "params": {},
            }
        )
        for index in range(gens)
    ]
    (folder / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def manager_factory() -> Iterator[Any]:
    made: list[RunManager] = []

    def make(path: Path) -> RunManager:
        manager = RunManager(path, stop_timeout=0.5)
        made.append(manager)
        return manager

    try:
        yield make
    finally:
        for manager in made:
            manager.shutdown()
        assert multiprocessing.active_children() == []


def test_earlier_runs_are_listed_with_their_history(runs: Path, manager_factory: Any) -> None:
    make_run_dir(runs, "alpha", generations=5, gens=2)

    manager = manager_factory(runs)

    [row] = manager.list()
    assert (row["name"], row["status"], row["gen"], row["learner"]) == (
        "alpha",
        "stopped",
        2,
        "evolution",
    )
    assert row["archived"] is True and row["best"] == pytest.approx(0.4)
    info = manager.get(row["id"])
    assert info["config"]["laps"] == 2 and [h["gen"] for h in info["history"]] == [0, 1]


def test_a_run_that_reached_its_generations_counts_as_finished(
    runs: Path, manager_factory: Any
) -> None:
    make_run_dir(runs, "done", generations=2, gens=2)

    [row] = manager_factory(runs).list()

    assert row["status"] == "finished"


def test_the_stream_of_an_earlier_run_replays_its_generations_and_ends(
    runs: Path, manager_factory: Any
) -> None:
    make_run_dir(runs, "alpha", generations=5, gens=3)
    manager = manager_factory(runs)
    [row] = manager.list()

    events = list(manager.subscribe(row["id"]))

    assert [kind for kind, _ in events] == ["gen", "gen", "gen", "status"]
    assert events[-1][1]["status"] == "stopped"
    assert list(manager.subscribe(row["id"], from_index=2))[0][1]["gen"] == 2


def test_commands_to_an_earlier_run_do_nothing(runs: Path, manager_factory: Any) -> None:
    make_run_dir(runs, "alpha")
    manager = manager_factory(runs)
    [row] = manager.list()

    manager.command(row["id"], {"cmd": "pause"})

    assert manager.get(row["id"])["status"] == "stopped"


def test_damaged_and_foreign_folders_are_skipped(runs: Path, manager_factory: Any) -> None:
    make_run_dir(runs, "good")
    (runs / "tracks").mkdir()
    (runs / "no-config").mkdir()
    broken = runs / "broken-20261009-101600"
    broken.mkdir()
    (broken / "config.json").write_text("{not json", encoding="utf-8")
    unknown = runs / "unknown-20261009-101700"
    unknown.mkdir()
    (unknown / "config.json").write_text(json.dumps({"nope": 1}), encoding="utf-8")
    (runs / "stray.txt").write_text("x", encoding="utf-8")
    torn = make_run_dir(runs, "torn", gens=1)
    with (torn / "history.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"t": "gen", "ge')  # the process died in the middle of a line

    names = sorted(row["name"] for row in manager_factory(runs).list())

    assert names == ["good", "torn"]


def test_only_the_newest_folders_are_restored(runs: Path, manager_factory: Any) -> None:
    for index in range(35):
        folder = make_run_dir(runs, f"run{index:02d}")
        folder.rename(runs / f"run{index:02d}-20261009-1015{index:02d}")

    rows = manager_factory(runs).list()

    assert len(rows) == 30
    assert "run34" in {row["name"] for row in rows} and "run00" not in {row["name"] for row in rows}


def test_deleting_an_earlier_run_hides_it_for_good_but_keeps_its_files(
    runs: Path, manager_factory: Any
) -> None:
    folder = make_run_dir(runs, "alpha")
    manager = manager_factory(runs)
    [row] = manager.list()

    manager.delete(row["id"])

    assert manager.list() == []
    assert (folder / "config.json").is_file() and (folder / "history.jsonl").is_file()
    assert manager_factory(runs).list() == []


def test_a_finished_live_run_is_restored_after_a_restart_and_dismissed_when_deleted(
    runs: Path, manager_factory: Any
) -> None:
    first = manager_factory(runs)
    run_id = first.create(_config(name="live", generations=2))
    wait_status(first, run_id, "finished")
    first.shutdown()

    second = manager_factory(runs)
    [row] = second.list()
    assert (row["name"], row["status"], row["gen"], row["archived"]) == (
        "live",
        "finished",
        2,
        True,
    )
    second.delete(row["id"])

    assert manager_factory(runs).list() == []


def test_a_deleted_live_run_does_not_come_back(runs: Path, manager_factory: Any) -> None:
    first = manager_factory(runs)
    run_id = first.create(_config(name="gone", generations=2))
    wait_status(first, run_id, "finished")

    first.delete(run_id)
    first.shutdown()

    assert manager_factory(runs).list() == []
