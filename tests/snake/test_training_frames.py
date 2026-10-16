import time

import pytest

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.dqn import DQNTrainer
from rl_fun.snake.evolution import EvolutionTrainer
from rl_fun.snake.manager import SnakeManager


def test_dqn_frames_are_actual_replay_transitions_without_evaluation_games():
    trainer = DQNTrainer(SnakeConfig(size=8, episodes=2, max_steps=10))
    frames = []
    trainer.run(lambda: True, lambda _: None, frames.append)
    transitions = [row for row in frames if row["frame"]["steps"] > 0]
    assert len(transitions) == trainer.steps == len(trainer.replay)
    assert {row["episode"] for row in transitions} == {1, 2}
    assert all(row["epsilon"] > .9 for row in transitions)
    assert sum(row["ended"] for row in transitions) == 2
    assert transitions[-1]["frame"] == trainer.env.snapshot()


def test_evolution_frames_identify_each_candidate_and_game():
    config = SnakeConfig(algorithm="evolution", size=8, generations=1,
                         population=4, evaluation_episodes=2, max_steps=10)
    trainer = EvolutionTrainer(config)
    frames = []
    trainer.run(lambda: True, lambda _: None, frames.append)
    assert {(row["candidate"], row["episode"]) for row in frames} == {
        (candidate, episode) for candidate in range(1, 5) for episode in (1, 2)
    }
    assert all(row["generation"] == 1 for row in frames)
    assert sum(row["ended"] for row in frames) == 8


def wait_for(manager, predicate):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = manager.state()
        if predicate(state):
            return state
        time.sleep(.01)
    pytest.fail("Timed out waiting for training state")


@pytest.mark.parametrize("algorithm", ["dqn", "evolution", "reinforce", "a2c", "ppo", "td3"])
def test_live_worker_frames_pause_and_switch_to_fast_training(tmp_path, algorithm):
    manager = SnakeManager(tmp_path)
    try:
        manager.train({"algorithm": algorithm, "size": 8, "episodes": 2,
                       "generations": 1, "population": 4, "max_steps": 10},
                      visualization={"enabled": True, "fps": 8})
        state = wait_for(manager, lambda state: state.get("frame") is not None)
        assert state["frame"]["run_id"] == state["run"]["id"]
        assert state["frame"]["frame"]["size"] == 8
        manager.control("pause")
        paused = wait_for(manager, lambda state: state["run"]["status"] == "paused")
        time.sleep(.2)
        assert manager.state()["frame_revision"] == paused["frame_revision"]
        manager.configure_visualization({"enabled": False, "fps": 20})
        manager.control("resume")
        final = wait_for(manager, lambda state: state["run"]["status"] == "finished")
        assert final["visualization"] == {"enabled": False, "fps": 20}
        assert final["run"]["history"]
        assert "frame" not in final["run"]
    finally:
        manager.shutdown()
