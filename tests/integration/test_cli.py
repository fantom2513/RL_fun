import json
import subprocess
import sys


def test_train_script_completes_smoke_run(tmp_path):
    # Arrange
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "cli-smoke",
                "algorithm": "epsilon_greedy",
                "seeds": [1, 2],
                "steps": 5,
                "workers": 1,
                "output_root": str(tmp_path / "runs"),
                "parameters": {"arms": 2, "epsilon": 0.1, "reward_std": 0.0},
            }
        ),
        encoding="utf-8",
    )

    # Act
    completed = subprocess.run(
        [sys.executable, "scripts/train.py", str(config_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Assert
    assert completed.returncode == 0


def test_compare_script_writes_csv_and_png_without_display(tmp_path):
    # Arrange
    run_directories = [tmp_path / "run-one", tmp_path / "run-two"]
    for index, run_directory in enumerate(run_directories, start=1):
        run_directory.mkdir()
        run_directory.joinpath("metrics.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "step": step,
                        "metrics": {"cumulative_reward": float(step * index)},
                    }
                )
                for step in (1, 2)
            ),
            encoding="utf-8",
        )
    output_directory = tmp_path / "comparison"

    # Act
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare.py",
            *(str(path) for path in run_directories),
            "--output-dir",
            str(output_directory),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Assert
    assert completed.returncode == 0
    assert output_directory.joinpath("comparison.csv").is_file()
    assert output_directory.joinpath("comparison.png").is_file()
