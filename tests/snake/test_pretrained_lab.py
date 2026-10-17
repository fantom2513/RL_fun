import pytest

from rl_fun.snake.manager import SnakeManager

NAME = "snake16-cycle-clone"


@pytest.fixture
def manager(tmp_path):
    manager = SnakeManager(tmp_path)
    yield manager
    manager.shutdown()


def test_the_built_in_model_is_listed_with_its_evidence(manager):
    rows = [row for row in manager.models() if row["id"] == NAME]

    assert len(rows) == 1
    assert rows[0]["size"] == 16 and rows[0]["verification"]["status"] == "verified"


def test_the_built_in_model_comes_after_the_trained_runs(manager):
    assert manager.models()[-1]["id"] == NAME


def test_there_is_exactly_one_verified_preset_for_the_built_in_model(manager):
    presets = [row for row in manager.presets() if row.get("model") == NAME]

    assert len(presets) == 1
    assert presets[0]["status"] == "verified" and presets[0]["config"]["size"] == 16


def test_an_agent_game_replays_the_built_in_model(manager):
    game = manager.create_game({"mode": "agent", "size": 16, "seed": 3000000, "model": NAME})

    for _ in range(5):
        game = manager.step_game(game["id"], {})

    assert game["trained"] is True and game["model"] == NAME
    assert game["agent"]["size"] == 16 and game["agent"]["steps"] == 5


def test_the_built_in_model_needs_its_own_board_size(manager):
    with pytest.raises(ValueError, match="размер"):
        manager.create_game({"mode": "agent", "size": 12, "model": NAME})


def test_other_non_hex_model_ids_are_still_rejected(manager):
    with pytest.raises(ValueError, match="идентификатор"):
        manager.create_game({"mode": "agent", "size": 16, "model": "not-a-model"})
