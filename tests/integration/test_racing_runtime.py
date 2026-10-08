import json
import os
import subprocess
import sys
from pathlib import Path


def run_script(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *map(str, args)], capture_output=True, text=True,
        encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120, check=False,
    )


def racing_config(tmp_path: Path) -> Path:
    values = json.loads(Path("configs/racing-random.json").read_text(encoding="utf-8"))
    values["run"]["output_root"] = str(tmp_path / "runs")
    values["run"]["total_steps"] = 50
    values["evaluation"]["max_episode_steps"] = 50
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_train_runs_random_policy_on_racing_environment(tmp_path: Path):
    result = run_script("scripts/train.py", racing_config(tmp_path))

    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout), dict)
    summaries = list((tmp_path / "runs").glob("racing-random/seed-*/summary.json"))
    assert len(summaries) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "success" for path in summaries
    )


def test_evaluate_runs_random_policy_on_racing_environment(tmp_path: Path):
    result = run_script("scripts/evaluate.py", racing_config(tmp_path), "--seed", 11)

    assert result.returncode == 0, result.stderr
