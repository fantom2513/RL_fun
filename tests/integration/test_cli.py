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
