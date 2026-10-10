import time

import pytest

from rl_fun.snake.manager import SnakeManager


def wait_terminal(manager):
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        state = manager.state()
        if state["run"] and state["run"]["status"] in ("finished", "error", "stopped"):
            return state["run"]
        time.sleep(0.05)
    pytest.fail("Snake worker did not finish")


def test_split_sessions_seed_and_direction_queue(tmp_path):
    manager = SnakeManager(tmp_path)
    try:
        game = manager.create_game({"mode": "split", "size": 8, "seed": 12})
        assert game["human"]["snake"] == game["agent"]["snake"]
        assert game["human"]["food"] == game["agent"]["food"]
        moved = manager.step_game(game["id"], {"direction": 1})
        assert moved["human"]["direction"] == 1
        moved = manager.step_game(game["id"], {"direction": 3})
        assert moved["human"]["direction"] == 1  # reverse rejected
        manager.delete_game(game["id"])
        with pytest.raises(KeyError):
            manager.step_game(game["id"], {})
    finally:
        manager.shutdown()


@pytest.mark.parametrize("algorithm", ["dqn", "evolution"])
def test_process_training_checkpoint_resume_and_archive(tmp_path, algorithm):
    manager = SnakeManager(tmp_path)
    config = {
        "algorithm": algorithm,
        "size": 8,
        "episodes": 2,
        "generations": 1,
        "max_steps": 12,
        "population": 4,
        "evaluation_episodes": 1,
        "batch_size": 4,
        "warmup": 4,
    }
    try:
        run_id = manager.train(config)
        with pytest.raises(RuntimeError):
            manager.train(config)
        run = wait_terminal(manager)
        assert run["status"] == "finished", run
        assert run["history"] and run["saved"]
        assert manager.models()[0]["config"]["population"] == 4
        game = manager.create_game({"mode": "agent", "model": run_id, "size": 8})
        assert game["trained"]
        manager.train(config, resume=run_id)
        resumed = wait_terminal(manager)
        assert resumed["status"] == "finished", resumed
        assert resumed["history"][-1]["iteration"] > run["history"][-1]["iteration"]
    finally:
        manager.shutdown()
    reopened = SnakeManager(tmp_path)
    try:
        assert len(reopened.models()) == 2
    finally:
        reopened.shutdown()


def test_models_are_confined_and_incompatible_resume_rejected(tmp_path):
    manager = SnakeManager(tmp_path)
    try:
        with pytest.raises(ValueError):
            manager.create_game({"mode": "agent", "model": "../outside"})
        with pytest.raises(ValueError):
            manager.create_game({"mode": "manual", "seed": True})
        with pytest.raises(ValueError):
            manager.create_game({"mode": "bogus"})
        with pytest.raises(ValueError):
            manager.control("pause")
    finally:
        manager.shutdown()


def test_sessions_bounded_and_crash_reported(tmp_path):
    manager = SnakeManager(tmp_path)
    try:
        first = manager.create_game({"mode": "manual"})["id"]
        for _ in range(33):
            manager.create_game({"mode": "manual"})
        with pytest.raises(KeyError):
            manager.step_game(first, {})
        manager.train({"episodes": 1000})
        manager.process.terminate()
        assert wait_terminal(manager)["status"] == "error"
    finally:
        manager.shutdown()
