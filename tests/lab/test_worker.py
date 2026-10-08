import json
from collections import deque
from pathlib import Path
from typing import Any

from rl_fun.lab.config import RunConfig
from rl_fun.lab.worker import run_worker
from rl_fun.racing.model_spec import ModelSpec


class FakeConnection:
    """Records sent messages and replays scripted commands.

    `commands_after_frame[k]` becomes readable right after the k-th frame message is sent, so a
    test can deliver a command while a generation is in progress, without real time.
    """

    def __init__(self, commands_after_frame: dict[int, list[dict[str, Any]]] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._script = dict(commands_after_frame or {})
        self._inbox: deque[dict[str, Any]] = deque()
        self._frames = 0

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        if message["t"] == "frame":
            self._frames += 1
            self._inbox.extend(self._script.pop(self._frames, []))

    def poll(self, timeout: float = 0) -> bool:
        return bool(self._inbox)

    def recv(self) -> dict[str, Any]:
        return self._inbox.popleft()

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [message for message in self.sent if message["t"] == kind]


def _config(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test",
        "track": "oval",
        "population": 8,
        "elite": 2,
        "max_steps": 60,
        "generations": 2,
    }
    data.update(overrides)
    return data


def _run(conn: FakeConnection, runs_dir: Path, **overrides: Any) -> None:
    run_worker(_config(**overrides), conn, runs_dir, speed="max")


def _only_run_dir(runs_dir: Path) -> Path:
    directories = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def test_messages_start_running_send_generations_in_order_and_end_finished(tmp_path):
    # Arrange
    conn = FakeConnection()

    # Act
    _run(conn, tmp_path)

    # Assert
    kinds = [message["t"] for message in conn.sent]
    assert conn.sent[0] == {"t": "status", "status": "running"}
    assert set(kinds) <= {"status", "frame", "gen"}
    assert [message["gen"] for message in conn.of_type("gen")] == [0, 1]
    assert kinds[-1] == "status"
    assert kinds.index("frame") < kinds.index("gen")
    assert conn.sent[-1]["status"] == "finished"


def test_some_frame_carries_network_state(tmp_path):
    # Arrange
    conn = FakeConnection()
    expected_sizes = ModelSpec().layer_sizes

    # Act
    _run(conn, tmp_path)

    # Assert
    networks = [frame["net"] for frame in conn.of_type("frame") if frame["net"] is not None]
    assert networks
    assert len(networks[0]["matrices"]) == len(expected_sizes) - 1
    assert len(networks[0]["activations"]) == len(expected_sizes)


def test_artifacts_hold_config_history_and_best_weights(tmp_path):
    # Arrange
    conn = FakeConnection()
    model = ModelSpec()
    expected_config = RunConfig.from_dict(_config()).to_dict()

    # Act
    _run(conn, tmp_path)

    # Assert
    run_dir = _only_run_dir(tmp_path)
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8")) == expected_config
    history = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["gen"] for line in history] == [0, 1]
    best = json.loads((run_dir / "best_weights.json").read_text(encoding="utf-8"))
    assert best["sizes"] == model.layer_sizes
    assert len(best["weights"]) == model.weight_count


def test_stop_command_after_first_frame_ends_run_as_stopped_and_keeps_artifacts_clean(tmp_path):
    # Arrange
    conn = FakeConnection({1: [{"cmd": "stop"}]})

    # Act
    _run(conn, tmp_path)

    # Assert
    assert conn.sent[-1] == {"t": "status", "status": "stopped"}
    assert not any(message.get("status") == "finished" for message in conn.sent)
    assert conn.of_type("gen") == []
    run_dir = _only_run_dir(tmp_path)
    assert (run_dir / "config.json").is_file()
    assert not (run_dir / "history.jsonl").exists()


def test_valid_update_during_generation_zero_applies_from_generation_one(tmp_path):
    # Arrange
    update = {"cmd": "update", "params": {"mutation_rate": 0.5, "elite": 3}}
    conn = FakeConnection({1: [update]})

    # Act
    _run(conn, tmp_path)

    # Assert
    first, second = conn.of_type("gen")
    assert first["params"]["mutation_rate"] == 0.15
    assert first["params"]["elite"] == 2
    assert second["params"]["mutation_rate"] == 0.5
    assert second["params"]["elite"] == 3
    assert conn.sent[-1] == {"t": "status", "status": "finished"}


def test_invalid_update_is_rejected_with_notice_and_run_continues(tmp_path):
    # Arrange
    conn = FakeConnection({1: [{"cmd": "update", "params": {"elite": 99}}]})

    # Act
    _run(conn, tmp_path)

    # Assert
    notices = conn.of_type("notice")
    assert len(notices) == 1
    assert "elite" in notices[0]["text"]
    assert [message["gen"] for message in conn.of_type("gen")] == [0, 1]
    assert conn.of_type("gen")[1]["params"]["elite"] == 2
    assert conn.sent[-1] == {"t": "status", "status": "finished"}


def test_invalid_track_ends_with_error_status_carrying_text_and_does_not_raise(tmp_path):
    # Arrange
    conn = FakeConnection()

    # Act
    _run(conn, tmp_path, track="nope")

    # Assert
    assert len(conn.sent) == 1
    assert conn.sent[0]["t"] == "status"
    assert conn.sent[0]["status"] == "error"
    assert "track" in conn.sent[0]["message"]
    assert "nope" in conn.sent[0]["message"]
    assert list(tmp_path.iterdir()) == []


def test_custom_model_runs_one_generation(tmp_path):
    # Arrange
    model = ModelSpec(
        inputs=("ray:-45", "ray:0", "ray:45", "steering_angle"),
        hidden=(4,),
        outputs=("steer", "accelerate", "brake", "boost"),
        activation="relu",
    )
    conn = FakeConnection()

    # Act
    _run(conn, tmp_path, model=model.to_dict(), generations=1)

    # Assert
    assert conn.sent[-1] == {"t": "status", "status": "finished"}
    assert len(conn.of_type("gen")) == 1
    best = json.loads((_only_run_dir(tmp_path) / "best_weights.json").read_text(encoding="utf-8"))
    assert best["sizes"] == model.layer_sizes
    assert len(best["weights"]) == model.weight_count


def test_same_seed_gives_same_best_progress_per_generation(tmp_path):
    # Arrange
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = FakeConnection()
    second = FakeConnection()

    # Act
    _run(first, first_dir)
    _run(second, second_dir)

    # Assert
    first_gens = [(message["best"], message["mean"]) for message in first.of_type("gen")]
    second_gens = [(message["best"], message["mean"]) for message in second.of_type("gen")]
    assert len(first_gens) == 2
    assert first_gens == second_gens
