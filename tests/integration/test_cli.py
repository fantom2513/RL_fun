import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def cartpole_values(tmp_path: Path) -> dict[str, object]:
    return {
        "name": "cartpole-cli",
        "environment": {"id": "CartPole-v1", "kwargs": {}},
        "algorithm": {"id": "random", "kwargs": {}},
        "run": {
            "seeds": [11, 22], "total_steps": 20, "workers": 1,
            "output_root": str(tmp_path / "runs"),
        },
        "evaluation": {"episodes": 2, "max_episode_steps": 10},
    }


def write_config(tmp_path: Path, values: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def run_script(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *map(str, args)], capture_output=True, text=True,
        encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30, check=False,
    )


def write_metric_runs(tmp_path: Path, metric: str) -> list[Path]:
    paths = [tmp_path / "run-one", tmp_path / "run-two"]
    for multiplier, path in enumerate(paths, start=1):
        path.mkdir()
        events = [
            {"step": step, "metrics": {metric: float(step * multiplier)}}
            for step in (1, 2)
        ]
        path.joinpath("metrics.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events), encoding="utf-8"
        )
    return paths


def test_train_script_completes_bandit_smoke_run(tmp_path: Path) -> None:
    # Arrange
    values = cartpole_values(tmp_path)
    values["environment"] = {
        "id": "RLFun/StationaryBandit-v0", "kwargs": {"arms": 2, "reward_std": 0.0},
    }
    values["algorithm"] = {"id": "bandit_epsilon", "kwargs": {"epsilon": 0.1}}
    # Act
    completed = run_script("scripts/train.py", write_config(tmp_path, values))
    # Assert
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["metric"] == "train/cumulative_reward"


def test_train_script_runs_cartpole_config(tmp_path: Path) -> None:
    # Arrange
    config = write_config(tmp_path, cartpole_values(tmp_path))
    # Act
    completed = run_script("scripts/train.py", config, "--metric", "train/steps")
    # Assert
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mean"] == 20.0


def test_train_reports_missing_metric_as_parser_error(tmp_path: Path) -> None:
    # Arrange
    config = write_config(tmp_path, cartpole_values(tmp_path))
    # Act
    completed = run_script("scripts/train.py", config, "--metric", "missing/key")
    # Assert
    assert completed.returncode == 2
    assert "missing/key" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_evaluate_writes_reproducible_headless_episode_artifacts(tmp_path: Path) -> None:
    # Arrange
    values = cartpole_values(tmp_path)
    config = write_config(tmp_path, values)
    # Act
    completed = [run_script("scripts/evaluate.py", config, "--seed", 44) for _ in range(2)]
    paths = sorted((tmp_path / "runs" / "cartpole-cli").glob("eval-seed-44-*"))
    # Assert
    assert all(result.returncode == 0 for result in completed), completed
    assert "Средняя награда" in completed[0].stdout
    assert len(paths) == 2
    assert paths[0].joinpath("metrics.jsonl").read_bytes() == paths[1].joinpath(
        "metrics.jsonl"
    ).read_bytes()
    events = [
        json.loads(line)
        for line in paths[0].joinpath("metrics.jsonl").read_text().splitlines()
    ]
    assert [event["step"] for event in events] == [1, 2]
    assert all(set(event["metrics"]) == {"episode/reward", "episode/length"} for event in events)
    assert json.loads(paths[0].joinpath("config.json").read_text()) == values
    assert json.loads(paths[0].joinpath("metadata.json").read_text())["seed"] == 44


def test_evaluate_defaults_to_first_config_seed(tmp_path: Path) -> None:
    # Arrange
    config = write_config(tmp_path, cartpole_values(tmp_path))
    # Act
    completed = run_script("scripts/evaluate.py", config)
    # Assert
    assert completed.returncode == 0, completed.stderr
    assert len(list((tmp_path / "runs" / "cartpole-cli").glob("eval-seed-11-*"))) == 1


@pytest.mark.parametrize("failure", [ValueError("rollout failed"), KeyboardInterrupt()])
def test_evaluate_closes_environment_on_rollout_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException,
) -> None:
    # Arrange
    spec = importlib.util.spec_from_file_location("evaluate_cli", "scripts/evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    closed = []

    class Environment:
        def close(self) -> None:
            closed.append(True)

    def make_environment(config: object, render_mode: object = None) -> Environment:
        assert render_mode is None
        return Environment()

    def fail_rollout(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(module, "make_environment", make_environment)
    monkeypatch.setattr(module, "rollout_episode", fail_rollout)
    config = write_config(tmp_path, cartpole_values(tmp_path))
    # Act
    if isinstance(failure, KeyboardInterrupt):
        assert module.main([str(config)]) == 130
    else:
        with pytest.raises(SystemExit, match="2"):
            module.main([str(config)])
    # Assert
    assert closed == [True]


def test_compare_accepts_arbitrary_metric_and_writes_neutral_columns(tmp_path: Path) -> None:
    # Arrange
    run_dirs = write_metric_runs(tmp_path, "custom/arbitrary_metric")
    output = tmp_path / "out"
    # Act
    completed = run_script(
        "scripts/compare.py", *run_dirs, "--metric", "custom/arbitrary_metric",
        "--output-dir", output,
    )
    # Assert
    assert completed.returncode == 0, completed.stderr
    with output.joinpath("comparison.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"step": "1", "mean": "1.5", "std": "0.5"},
            {"step": "2", "mean": "3.0", "std": "1.0"},
        ]
    assert output.joinpath("comparison.png").is_file()


def test_compare_ignores_failed_runs_and_runs_without_metric(tmp_path: Path) -> None:
    # Arrange
    run_dirs = write_metric_runs(tmp_path, "custom/key")
    run_dirs[1].joinpath("summary.json").write_text('{"status": "failure"}')
    missing = tmp_path / "missing"
    missing.mkdir()
    missing.joinpath("metrics.jsonl").write_text('{"step": 1, "metrics": {"other": 99}}')
    # Act
    completed = run_script(
        "scripts/compare.py", *run_dirs, missing, "--metric", "custom/key",
        "--output-dir", tmp_path / "out",
    )
    # Assert
    assert completed.returncode == 0, completed.stderr
    with (tmp_path / "out" / "comparison.csv").open(encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["mean"] == "1.0"


def test_compare_reports_missing_metric(tmp_path: Path) -> None:
    # Arrange
    run_dirs = write_metric_runs(tmp_path, "other/key")
    # Act
    completed = run_script("scripts/compare.py", *run_dirs, "--metric", "absent/key")
    # Assert
    assert completed.returncode == 2
    assert "absent/key" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("script", ["train", "evaluate", "compare"])
def test_cli_help_has_russian_descriptions(script: str) -> None:
    # Arrange / Act
    completed = run_script(f"scripts/{script}.py", "--help")
    # Assert
    assert completed.returncode == 0
    assert any("а" <= character <= "я" for character in completed.stdout)
