import json

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.manager import SnakeManager


def test_pretrained_choices_pick_best_evaluated_model_per_algorithm(tmp_path):
    manager = SnakeManager(tmp_path)
    try:
        for index, (algorithm, wins, occupancy) in enumerate(
            [
                ("ppo", 0, 0.1),
                ("ppo", 1, 0.08),
                ("double_dqn", 0, 0.12),
                ("double_dqn", 0, 0.09),
                ("a2c", 0, 0.02),
            ],
            start=1,
        ):
            directory = manager.root / f"{index:032x}"
            directory.mkdir()
            config = SnakeConfig(algorithm=algorithm, size=16, observation_mode="board").to_dict()
            checkpoint = directory / "model.pt"
            checkpoint.write_bytes(b"listing fixture")
            (directory / "record.json").write_text(
                json.dumps(
                    {
                        "config": config,
                        "created": index,
                        "status": "finished",
                    }
                ),
                encoding="utf-8",
            )
            (directory / "evaluation.json").write_text(
                json.dumps(
                    {
                        "phase": "tuning",
                        "config": config,
                        "episodes": 20,
                        "wins": wins,
                        "occupancy": occupancy,
                        "model_mtime_ns": checkpoint.stat().st_mtime_ns,
                    }
                ),
                encoding="utf-8",
            )
        choices = {
            p["config"]["algorithm"]: p for p in manager.presets() if p["id"].startswith("best_")
        }
        assert set(choices) == {"ppo", "double_dqn", "a2c"}
        assert choices["ppo"]["model"] == f"{2:032x}"
        assert choices["double_dqn"]["model"] == f"{3:032x}"
        assert all(p["status"] == "experimental" for p in choices.values())
        assert choices["ppo"]["result"] == {"wins": 1, "episodes": 20, "occupancy": 0.08}
    finally:
        manager.shutdown()
