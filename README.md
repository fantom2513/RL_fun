# RL Fun

RL Fun is a small, testable reinforcement-learning workbench. Its first milestone
implements stationary multi-armed bandits and compares random, greedy, and
epsilon-greedy action selection.

## Setup and run

Run these commands from the repository root:

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check src scripts tests
uv run python scripts/train.py configs/bandit-epsilon-greedy.json
uv run jupyter lab
```

The last command opens Jupyter Lab; start with
`notebooks/01_bandits/01_epsilon_greedy.ipynb`.

## Project layout

- `src/` contains the reusable environments, algorithms, experiment runner, and tracking code.
- `scripts/` contains command-line entry points for training and comparing runs.
- `notebooks/` contains explanatory, executable learning material that calls the package code.
- `configs/` contains repeatable experiment configurations.
- `runs/` is generated output: one isolated directory per run with configuration, metrics, and summary artifacts.

PyTorch, Arena, and Neural Inspector are planned for subsequent milestones; they are
not part of this first bandit workbench.
