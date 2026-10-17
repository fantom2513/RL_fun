import json

import pytest

from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.gpu.pretrained import PRETRAINED, load_pretrained, pretrained_rows

NAME = "snake16-cycle-clone"


def test_the_tracked_checkpoint_loads_as_a_16x16_policy():
    policy = load_pretrained(NAME)

    assert policy.size == 16 and policy.observation_mode == "egocentric_cycle"


def test_the_policy_plays_legal_moves_and_eats_within_the_first_laps():
    policy = load_pretrained(NAME)
    env = SnakeEnv(16, observation_mode=policy.observation_mode)
    observation, _ = env.reset(seed=3000000)

    for _ in range(400):
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        assert not (terminated or truncated)

    assert env.score >= 1


def test_an_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="модель"):
        load_pretrained("no-such-model")


def test_the_registry_describes_the_tracked_model():
    assert NAME in PRETRAINED and PRETRAINED[NAME]["size"] == 16


def test_rows_are_built_from_the_evidence_file(tmp_path):
    evidence = {"games": 100, "wins": 100, "win_rate": 1.0, "occupancy": 1.0, "seeds": [1, 100]}
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    registry = {NAME: {"size": 16, "evidence": "evidence.json", "file": "model.pt"}}

    rows = pretrained_rows(registry, tmp_path)

    assert rows[0]["id"] == NAME and rows[0]["algorithm"] == "pretrained"
    assert rows[0]["verification"] == {"status": "verified", "wins": 100, "episodes": 100}


def test_weak_evidence_does_not_count_as_verified(tmp_path):
    evidence = {"games": 100, "wins": 60, "win_rate": 0.6, "occupancy": 0.8, "seeds": [1, 100]}
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    registry = {NAME: {"size": 16, "evidence": "evidence.json", "file": "model.pt"}}

    rows = pretrained_rows(registry, tmp_path)

    assert rows[0]["verification"]["status"] == "experimental"


def test_missing_evidence_hides_the_model(tmp_path):
    registry = {NAME: {"size": 16, "evidence": "evidence.json", "file": "model.pt"}}

    assert pretrained_rows(registry, tmp_path) == []
