import json
from collections import deque
from typing import Any

from rl_fun.lab.worker import run_worker


class Connection:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._inbox: deque[dict[str, Any]] = deque()

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def poll(self, timeout: float = 0) -> bool:
        return bool(self._inbox)

    def recv(self) -> dict[str, Any]:
        return self._inbox.popleft()


def _config(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "t",
        "track": "oval",
        "population": 8,
        "elite": 2,
        "max_steps": 60,
        "generations": 2,
    }
    data.update(overrides)
    return data


def test_evolution_gen_messages_carry_no_extra(tmp_path):
    # Arrange
    conn = Connection()

    # Act
    run_worker(_config(), conn, tmp_path, speed="max")

    # Assert
    gens = [m for m in conn.sent if m["t"] == "gen"]
    assert len(gens) == 2
    assert all("extra" not in message for message in gens)


def test_config_artifact_records_the_learner_and_ppo_params(tmp_path):
    # Arrange
    conn = Connection()

    # Act
    run_worker(_config(generations=1), conn, tmp_path, speed="max")

    # Assert
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["learner"] == "evolution"
    assert saved["ppo"]["learning_rate"] == 3e-4


def test_unimplemented_learner_ends_with_error_status_not_a_crash(tmp_path):
    # Arrange
    conn = Connection()

    # Act
    run_worker(_config(learner="ppo"), conn, tmp_path, speed="max")

    # Assert
    last = conn.sent[-1]
    assert last["t"] == "status"
    if last["status"] == "error":
        assert "обучатель ppo ещё не реализован" in last["message"]
        assert list(tmp_path.iterdir()) == []
    else:
        assert last["status"] == "finished"  # task D implemented PPO


def test_unknown_learner_ends_with_error_status(tmp_path):
    # Arrange
    conn = Connection()

    # Act
    run_worker(_config(learner="sac"), conn, tmp_path, speed="max")

    # Assert
    assert conn.sent[-1]["status"] == "error"
    assert "learner" in conn.sent[-1]["message"]
