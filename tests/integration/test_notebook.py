import subprocess
import sys
from pathlib import Path


def test_bandit_notebook_executes(tmp_path):
    # Arrange
    path = Path("notebooks/01_bandits/01_epsilon_greedy.ipynb")

    # Act
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=120",
            "--output",
            str(tmp_path / "executed.ipynb"),
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr
