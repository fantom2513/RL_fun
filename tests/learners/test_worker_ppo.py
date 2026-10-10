import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from rl_fun.lab.worker import run_worker
from rl_fun.racing import reference
from rl_fun.racing.model_spec import ModelSpec

EXTRA_KEYS = {"policy_loss", "value_loss", "entropy", "std", "kl", "mean_return", "episodes"}


class FakeConnection:
    """Records sent messages; `commands_after_frame[k]` becomes readable after the k-th frame."""

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
        "name": "ppo",
        "track": "oval",
        "learner": "ppo",
        "population": 6,
        "max_steps": 50,
        "generations": 2,
        "ppo": {"rollout_steps": 16, "minibatch_size": 32, "epochs": 2},
    }
    data.update(overrides)
    return data


def _run(conn: FakeConnection, runs_dir: Path, **overrides: Any) -> None:
    run_worker(_config(**overrides), conn, runs_dir, speed="max")


def _only_run_dir(runs_dir: Path) -> Path:
    directories = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def test_ppo_run_streams_running_frames_two_iterations_with_extra_and_finishes(tmp_path):
    # Arrange
    conn = FakeConnection()

    # Act
    _run(conn, tmp_path)

    # Assert
    kinds = [message["t"] for message in conn.sent]
    assert conn.sent[0] == {"t": "status", "status": "running"}
    assert set(kinds) <= {"status", "frame", "gen", "meta"}
    gens = conn.of_type("gen")
    assert [message["gen"] for message in gens] == [0, 1]
    assert all(set(message["extra"]) == EXTRA_KEYS for message in gens)
    assert all(math.isfinite(value) for message in gens for value in message["extra"].values())
    assert conn.sent[-1] == {"t": "status", "status": "finished"}


def test_ppo_frames_carry_the_iteration_and_network_state(tmp_path):
    # Arrange
    conn = FakeConnection()
    sizes = ModelSpec().layer_sizes

    # Act
    _run(conn, tmp_path)

    # Assert
    frames = conn.of_type("frame")
    assert frames
    assert frames[0]["gen"] == 0
    assert frames[0]["n"] == 6
    networks = [frame["net"] for frame in frames if frame["net"] is not None]
    assert networks
    assert len(networks[0]["matrices"]) == len(sizes) - 1
    assert len(networks[0]["activations"]) == len(sizes)


def test_ppo_artifacts_use_the_shared_weights_format(tmp_path):
    # Arrange
    conn = FakeConnection()
    model = ModelSpec()

    # Act
    _run(conn, tmp_path)

    # Assert
    run_dir = _only_run_dir(tmp_path)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["learner"] == "ppo"
    assert config["ppo"]["rollout_steps"] == 16
    history = (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["gen"] for line in history] == [0, 1]
    best = json.loads((run_dir / "best_weights.json").read_text(encoding="utf-8"))
    assert set(best) == {"sizes", "activation", "generation", "fitness", "weights"}
    assert best["sizes"] == model.layer_sizes
    assert len(best["weights"]) == model.weight_count
    output = reference.forward(
        np.asarray(best["weights"]), np.full(len(model.inputs), 0.5), best["sizes"],
        best["activation"],
    )
    assert output.shape == (len(model.outputs),)


def test_ppo_update_during_iteration_zero_shows_in_the_next_gen(tmp_path):
    # Arrange
    update = {"cmd": "update", "params": {"ppo": {"learning_rate": 0.001}}}
    conn = FakeConnection({1: [update]})

    # Act
    _run(conn, tmp_path)

    # Assert
    first, second = conn.of_type("gen")
    assert first["params"]["ppo.learning_rate"] == 3e-4
    assert second["params"]["ppo.learning_rate"] == 0.001
    assert conn.of_type("notice") == []
    assert conn.sent[-1] == {"t": "status", "status": "finished"}


def test_ppo_invalid_update_key_gives_a_notice_and_the_run_continues(tmp_path):
    # Arrange
    update = {"cmd": "update", "params": {"ppo": {"rollout_steps": 64}, "elite": 3}}
    conn = FakeConnection({1: [update]})

    # Act
    _run(conn, tmp_path)

    # Assert
    notices = [message["text"] for message in conn.of_type("notice")]
    assert len(notices) == 2
    assert "ppo.rollout_steps" in notices[0]
    assert "elite" in notices[1]
    assert [message["gen"] for message in conn.of_type("gen")] == [0, 1]
    assert conn.sent[-1] == {"t": "status", "status": "finished"}


def test_ppo_stop_command_ends_the_run_as_stopped(tmp_path):
    # Arrange
    conn = FakeConnection({1: [{"cmd": "stop"}]})

    # Act
    _run(conn, tmp_path, generations=None)

    # Assert
    assert conn.sent[-1] == {"t": "status", "status": "stopped"}
    assert conn.of_type("gen") == []
    assert not (_only_run_dir(tmp_path) / "history.jsonl").exists()
