import json

from rl_fun.tracking.artifacts import create_run_dir, write_json_atomic


def test_run_directories_are_isolated_by_run_id(tmp_path):
    # Arrange / Act
    first = create_run_dir(tmp_path, "demo", "run-a")
    second = create_run_dir(tmp_path, "demo", "run-b")

    # Assert
    assert first != second


def test_atomic_json_write_replaces_previous_value(tmp_path):
    # Arrange
    path = tmp_path / "summary.json"
    write_json_atomic(path, {"status": "running"})

    # Act
    write_json_atomic(path, {"status": "success"})

    # Assert
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "success"}
