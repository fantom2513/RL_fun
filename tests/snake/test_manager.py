import threading
import time

import pytest

from rl_fun.snake.manager import SnakeManager


def test_finished_status_waits_for_worker_exit(tmp_path):
    manager = SnakeManager(tmp_path)
    run_id = "a" * 32
    manager._path(run_id).mkdir()
    manager.run = {"id": run_id, "status": "running", "error": None}
    exiting = threading.Event()
    allow_exit = threading.Event()

    class DelayedProcess:
        def join(self, timeout=None):
            exiting.set()
            allow_exit.wait(2)

        def is_alive(self):
            return not allow_exit.is_set()

    class TerminalConnection:
        delivered = False

        def poll(self, timeout):
            return True

        def recv(self):
            if self.delivered:
                raise EOFError
            self.delivered = True
            return {"kind": "status", "status": "finished"}

        def close(self):
            pass

    reader = threading.Thread(
        target=manager._read, args=(run_id, TerminalConnection(), DelayedProcess())
    )
    reader.start()
    try:
        assert exiting.wait(0.5), "terminal message must wait for worker shutdown"
        assert manager.run["status"] == "running"
        allow_exit.set()
        reader.join(2)
        assert not reader.is_alive()
        assert manager.run["status"] == "finished"
    finally:
        allow_exit.set()
        reader.join(2)
        manager.shutdown()


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


def test_agent_restarts_without_resetting_dead_human(tmp_path):
    manager = SnakeManager(tmp_path)
    try:
        game = manager.create_game({"mode": "split", "size": 8})
        session = manager.sessions[game["id"]]
        for parameter in session.policy.parameters():
            parameter.data.zero_()  # greedy straight: deterministic wall collision
        for _ in range(4):
            game = manager.step_game(game["id"], {})
        assert game["agent"]["done"] and game["human"]["done"]
        human = game["human"]
        game = manager.step_game(game["id"], {})
        assert not game["agent"]["done"]
        assert game["agent_episode"] == 2
        assert game["human"] == human
    finally:
        manager.shutdown()


def test_follow_training_updates_existing_session_and_keeps_selected_model(tmp_path):
    from rl_fun.snake.checkpoint import save_checkpoint
    from rl_fun.snake.config import SnakeConfig
    from rl_fun.snake.dqn import DQNTrainer

    manager = SnakeManager(tmp_path)
    try:
        game = manager.create_game({"mode": "split", "size": 8, "follow_training": True})
        trainer = DQNTrainer(SnakeConfig(size=8))
        trainer.completed = trainer.updates = 1
        run_id = "a" * 32
        path = manager.root / run_id
        path.mkdir()
        save_checkpoint(path / "model.pt", trainer.state_dict())
        manager.run = {
            "id": run_id,
            "config": trainer.config.to_dict(),
            "saved": True,
            "saved_iteration": 1,
        }
        game = manager.step_game(game["id"], {"direction": 1})
        assert game["model"] == run_id and game["trained"]
        assert game["model_iteration"] == 1
        fixed = manager.create_game({"mode": "agent", "size": 8, "model": run_id})
        trainer.completed = 2
        for parameter in trainer.model.parameters():
            parameter.data.zero_()
        trainer.model[-1].bias.data[1] = 10
        save_checkpoint(path / "model.pt", trainer.state_dict())
        manager.run["saved_iteration"] = 2
        game = manager.step_game(game["id"], {})
        assert game["model_iteration"] == 2
        assert manager.sessions[game["id"]].policy.act([0.0] * 11) == 1
        assert manager.step_game(fixed["id"], {})["model_iteration"] == 1
        assert game["human"]["steps"] == 2
    finally:
        manager.run = None
        manager.shutdown()
