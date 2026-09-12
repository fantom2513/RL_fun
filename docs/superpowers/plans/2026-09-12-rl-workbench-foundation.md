# RL Workbench Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight, reproducible local RL experiment workbench and validate it end to end with stationary-bandit experiments using random, greedy, and epsilon-greedy algorithms.

**Architecture:** Notebooks and scripts call the same importable package. Basic algorithms keep their own explicit loops, while a small runner handles configuration, independent seeds, local metrics, artifacts, and CPU process parallelism. Gymnasium is the task-environment contract; generated runs are isolated by directory.

**Tech Stack:** Python 3.12, uv, Gymnasium, NumPy, pytest, Ruff, Jupyter, Matplotlib

**Spec:** `docs/superpowers/specs/2026-09-12-rl-workbench-design.md`

## Global Constraints

- Target Python 3.12 on Windows 11.
- Use `uv`, `pyproject.toml`, and a committed `uv.lock` for reproducible dependencies.
- Keep algorithm implementations short and explicit; do not introduce a universal Trainer hierarchy or plugin framework.
- Use Gymnasium as the public environment contract.
- Use JSON for input configuration and saved configuration snapshots.
- Every seeded run writes to an independent directory under `runs/`.
- CPU batches may use process parallelism compatible with Windows `spawn` semantics.
- Do not install PyTorch, Stable-Baselines3, TensorBoard, Pygame, Hydra, or MLflow in this milestone.
- Keep notebooks as clients of `src/rl_fun`; do not duplicate canonical algorithm implementations in notebooks.
- Test infrastructure behavior and canonical reference code without building a learner autograder.
- Follow strict RED/GREEN TDD: commit failing tests before implementation, and do not alter those tests during GREEN.

---

## Planned File Map

- `pyproject.toml` — package metadata, dependency groups, pytest, and Ruff configuration.
- `uv.lock` — exact resolved dependency graph.
- `.python-version` — Python 3.12 selection for uv.
- `.gitignore` — local environment, generated runs, notebook checkpoints, caches, and brainstorming artifacts.
- `src/rl_fun/__init__.py` — package version and public package marker.
- `src/rl_fun/py.typed` — marker for the package's inline type annotations.
- `src/rl_fun/experiments/config.py` — typed JSON configuration and validation.
- `src/rl_fun/tracking/metrics.py` — in-memory and JSONL metric sinks.
- `src/rl_fun/tracking/artifacts.py` — isolated run directories, JSON artifacts, and runtime metadata.
- `src/rl_fun/environments/bandit.py` — stationary Gaussian multi-armed bandit as a Gymnasium environment.
- `src/rl_fun/algorithms/bandits/types.py` — compact shared outcome type and incremental estimate update.
- `src/rl_fun/algorithms/bandits/random_agent.py` — random baseline.
- `src/rl_fun/algorithms/bandits/greedy.py` — greedy reference algorithm.
- `src/rl_fun/algorithms/bandits/epsilon_greedy.py` — epsilon-greedy reference algorithm.
- `src/rl_fun/experiments/bandit.py` — construction and execution of one configured bandit run.
- `src/rl_fun/experiments/runner.py` — serial or process-parallel multi-seed execution and failure aggregation.
- `src/rl_fun/experiments/compare.py` — aggregate curves and summary statistics across runs.
- `scripts/train.py` — command-line entry point for a JSON experiment.
- `scripts/compare.py` — command-line summary and plot generation.
- `configs/bandit-epsilon-greedy.json` — first reproducible example configuration.
- `notebooks/01_bandits/01_epsilon_greedy.ipynb` — explanation and interactive comparison using package code.
- `tests/` — focused tests mirroring the source responsibilities above.

---

### Task 1: Bootstrap the package and typed experiment configuration

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/rl_fun/__init__.py`
- Create: `src/rl_fun/py.typed`
- Create: `src/rl_fun/experiments/__init__.py`
- Create: `src/rl_fun/experiments/config.py`
- Create: `configs/bandit-epsilon-greedy.json`
- Test: `tests/experiments/test_config.py`

**Interfaces:**
- Consumes: JSON files and `pathlib.Path` values.
- Produces: `ExperimentConfig.from_json(path: Path) -> ExperimentConfig`, `ExperimentConfig.to_dict() -> dict[str, object]`, and `ExperimentConfig.validate() -> None`.

- [ ] **Step 1: Create package metadata and resolve only first-milestone dependencies**

Run:

```powershell
uv init --lib --name rl-fun --python 3.12 .
uv add gymnasium numpy
uv add --group notebook jupyter matplotlib
uv add --dev pytest ruff
```

Add these tool settings to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Set `.gitignore` to:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
runs/
.superpowers/
```

- [ ] **Step 2: Verify the environment and commit the non-logic bootstrap**

Run:

```powershell
uv sync --all-groups
uv run python -c "import gymnasium, numpy; print(gymnasium.__version__, numpy.__version__)"
git add pyproject.toml uv.lock .python-version .gitignore README.md src/rl_fun
git commit -m "chore: bootstrap RL workbench package"
```

Expected: dependency synchronization succeeds and the import command prints both versions.

- [ ] **Step 3: Write failing configuration tests**

Create `tests/experiments/test_config.py`:

```python
import json

import pytest

from rl_fun.experiments.config import ExperimentConfig


def test_config_loads_explicit_seeds(tmp_path):
    # Arrange
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "name": "bandit-demo",
                "algorithm": "epsilon_greedy",
                "seeds": [11, 22],
                "steps": 100,
                "workers": 2,
                "output_root": "runs",
                "parameters": {"arms": 10, "epsilon": 0.1},
            }
        ),
        encoding="utf-8",
    )

    # Act
    config = ExperimentConfig.from_json(path)

    # Assert
    assert config.seeds == (11, 22)


@pytest.mark.parametrize("steps", [0, -1])
def test_config_rejects_non_positive_steps(steps):
    # Arrange
    config = ExperimentConfig(
        name="invalid",
        algorithm="random",
        seeds=(1,),
        steps=steps,
        workers=1,
        output_root="runs",
        parameters={},
    )

    # Act / Assert
    with pytest.raises(ValueError, match="steps must be positive"):
        config.validate()


def test_config_round_trip_preserves_values():
    # Arrange
    config = ExperimentConfig(
        name="bandit-demo",
        algorithm="greedy",
        seeds=(3, 5),
        steps=50,
        workers=1,
        output_root="runs",
        parameters={"arms": 4},
    )

    # Act
    restored = ExperimentConfig.from_dict(config.to_dict())

    # Assert
    assert restored == config
```

- [ ] **Step 4: Run RED and commit the failing contract**

Run:

```powershell
uv run pytest tests/experiments/test_config.py -v
```

Expected: FAIL during import because `rl_fun.experiments.config` does not exist.

Commit:

```powershell
git add tests/experiments/test_config.py
git commit -m "test: define experiment configuration contract"
```

- [ ] **Step 5: Implement the minimal typed configuration**

Create `src/rl_fun/experiments/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    algorithm: str
    seeds: tuple[int, ...]
    steps: int
    workers: int = 1
    output_root: str = "runs"
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.algorithm.strip():
            raise ValueError("algorithm must not be empty")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        config = cls(
            name=str(values["name"]),
            algorithm=str(values["algorithm"]),
            seeds=tuple(int(seed) for seed in values["seeds"]),
            steps=int(values["steps"]),
            workers=int(values.get("workers", 1)),
            output_root=str(values.get("output_root", "runs")),
            parameters=dict(values.get("parameters", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
```

Create `configs/bandit-epsilon-greedy.json`:

```json
{
  "name": "bandit-epsilon-greedy",
  "algorithm": "epsilon_greedy",
  "seeds": [11, 22, 33, 44],
  "steps": 1000,
  "workers": 4,
  "output_root": "runs",
  "parameters": {
    "arms": 10,
    "epsilon": 0.1,
    "reward_std": 1.0
  }
}
```

- [ ] **Step 6: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/experiments/test_config.py -v
uv run ruff check src tests
```

Expected: all configuration tests pass and Ruff reports no violations.

Commit:

```powershell
git add src/rl_fun/experiments configs/bandit-epsilon-greedy.json tests/experiments/test_config.py
git commit -m "feat: add typed experiment configuration"
```

---

### Task 2: Add metric sinks and isolated run artifacts

**Files:**
- Create: `src/rl_fun/tracking/__init__.py`
- Create: `src/rl_fun/tracking/metrics.py`
- Create: `src/rl_fun/tracking/artifacts.py`
- Test: `tests/tracking/test_metrics.py`
- Test: `tests/tracking/test_artifacts.py`

**Interfaces:**
- Consumes: scalar metric mappings, an output root, experiment name, run identifier, seed, and configuration mapping.
- Produces: `MetricSink.log(step: int, metrics: Mapping[str, float]) -> None`, `MemoryMetricSink.events`, `JsonlMetricSink`, `create_run_dir(...) -> Path`, `write_json_atomic(...) -> None`, and `collect_metadata(seed: int) -> dict[str, object]`.

- [ ] **Step 1: Write failing metric and artifact tests**

Create `tests/tracking/test_metrics.py`:

```python
import json

from rl_fun.tracking.metrics import JsonlMetricSink, MemoryMetricSink


def test_memory_sink_keeps_metric_event():
    # Arrange
    sink = MemoryMetricSink()

    # Act
    sink.log(3, {"reward": 1.25})

    # Assert
    assert sink.events[0].metrics == {"reward": 1.25}


def test_jsonl_sink_writes_one_json_object_per_event(tmp_path):
    # Arrange
    path = tmp_path / "metrics.jsonl"

    # Act
    with JsonlMetricSink(path) as sink:
        sink.log(7, {"regret": 0.5})

    # Assert
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "step": 7,
        "metrics": {"regret": 0.5},
    }
```

Create `tests/tracking/test_artifacts.py`:

```python
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
```

- [ ] **Step 2: Run RED and commit the failing contracts**

Run:

```powershell
uv run pytest tests/tracking -v
```

Expected: FAIL during import because the tracking modules do not exist.

Commit:

```powershell
git add tests/tracking
git commit -m "test: define metrics and artifact contracts"
```

- [ ] **Step 3: Implement metric sinks**

Create `src/rl_fun/tracking/metrics.py` with:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Mapping, Protocol, Self


@dataclass(frozen=True, slots=True)
class MetricEvent:
    step: int
    metrics: dict[str, float]


class MetricSink(Protocol):
    def log(self, step: int, metrics: Mapping[str, float]) -> None: ...


class MemoryMetricSink:
    def __init__(self) -> None:
        self.events: list[MetricEvent] = []

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        self.events.append(MetricEvent(step, {key: float(value) for key, value in metrics.items()}))


class JsonlMetricSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        event = MetricEvent(step, {key: float(value) for key, value in metrics.items()})
        self._handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
```

- [ ] **Step 4: Implement artifact helpers**

Create `src/rl_fun/tracking/artifacts.py`:

```python
from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def create_run_dir(root: Path, experiment: str, run_id: str) -> Path:
    path = root / experiment / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run_text(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_metadata(seed: int) -> dict[str, object]:
    commit = _run_text(["git", "rev-parse", "HEAD"])
    status = _run_text(["git", "status", "--porcelain"])
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
    return {
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            "gymnasium": importlib.metadata.version("gymnasium"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "gpu": gpu,
        "cuda_driver_available": gpu is not None,
    }
```

- [ ] **Step 5: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/tracking -v
uv run ruff check src tests
```

Expected: all tracking tests pass and Ruff reports no violations.

Commit:

```powershell
git add src/rl_fun/tracking tests/tracking
git commit -m "feat: add local metrics and run artifacts"
```

---

### Task 3: Implement the stationary bandit Gymnasium environment

**Files:**
- Create: `src/rl_fun/environments/__init__.py`
- Create: `src/rl_fun/environments/bandit.py`
- Test: `tests/environments/test_bandit.py`

**Interfaces:**
- Consumes: `arms: int`, `horizon: int`, `reward_std: float`, and Gymnasium reset seeds.
- Produces: `StationaryBanditEnv(gym.Env[np.ndarray, int])` with `Discrete(arms)` actions, constant one-element observations, Gaussian rewards, truncation at the horizon, and diagnostic `optimal_action`/`instant_regret` info.

- [ ] **Step 1: Write failing environment tests**

Create `tests/environments/test_bandit.py`:

```python
import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_fun.environments.bandit import StationaryBanditEnv


def test_bandit_passes_gymnasium_contract():
    # Arrange
    env = StationaryBanditEnv(arms=4, horizon=20, reward_std=1.0)

    # Act / Assert
    check_env(env)


def test_same_seed_reproduces_arm_means():
    # Arrange
    first = StationaryBanditEnv(arms=3, horizon=5)
    second = StationaryBanditEnv(arms=3, horizon=5)

    # Act
    first.reset(seed=42)
    second.reset(seed=42)

    # Assert
    assert np.array_equal(first.arm_means, second.arm_means)


def test_bandit_truncates_at_horizon():
    # Arrange
    env = StationaryBanditEnv(arms=2, horizon=2)
    env.reset(seed=1)

    # Act
    env.step(0)
    _, _, _, truncated, _ = env.step(0)

    # Assert
    assert truncated is True
```

- [ ] **Step 2: Run RED and commit the failing environment contract**

Run:

```powershell
uv run pytest tests/environments/test_bandit.py -v
```

Expected: FAIL during import because `StationaryBanditEnv` does not exist.

Commit:

```powershell
git add tests/environments/test_bandit.py
git commit -m "test: define stationary bandit environment"
```

- [ ] **Step 3: Implement the environment**

Create `src/rl_fun/environments/bandit.py` with this behavior:

```python
class StationaryBanditEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, arms: int = 10, horizon: int = 1000, reward_std: float = 1.0):
        if arms < 2:
            raise ValueError("arms must be at least 2")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if reward_std < 0:
            raise ValueError("reward_std must be non-negative")
        self.arms = arms
        self.horizon = horizon
        self.reward_std = reward_std
        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)
        self.arm_means = np.zeros(arms, dtype=np.float64)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.arm_means = self.np_random.normal(0.0, 1.0, size=self.arms)
        self._step = 0
        return np.zeros(1, dtype=np.float32), {"optimal_action": int(self.arm_means.argmax())}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        reward = float(self.np_random.normal(self.arm_means[action], self.reward_std))
        regret = float(self.arm_means.max() - self.arm_means[action])
        self._step += 1
        truncated = self._step >= self.horizon
        info = {"optimal_action": int(self.arm_means.argmax()), "instant_regret": regret}
        return np.zeros(1, dtype=np.float32), reward, False, truncated, info
```

Use explicit return annotations matching Gymnasium's five-value `step` result and two-value `reset` result.

- [ ] **Step 4: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/environments/test_bandit.py -v
uv run ruff check src tests
```

Expected: the environment tests and Gymnasium checker pass.

Commit:

```powershell
git add src/rl_fun/environments tests/environments/test_bandit.py
git commit -m "feat: add stationary bandit environment"
```

---

### Task 4: Add short random, greedy, and epsilon-greedy algorithms

**Files:**
- Create: `src/rl_fun/algorithms/__init__.py`
- Create: `src/rl_fun/algorithms/bandits/__init__.py`
- Create: `src/rl_fun/algorithms/bandits/types.py`
- Create: `src/rl_fun/algorithms/bandits/random_agent.py`
- Create: `src/rl_fun/algorithms/bandits/greedy.py`
- Create: `src/rl_fun/algorithms/bandits/epsilon_greedy.py`
- Test: `tests/algorithms/bandits/test_algorithms.py`

**Interfaces:**
- Consumes: `StationaryBanditEnv`, `steps: int`, `numpy.random.Generator`, and `MetricSink`.
- Produces: `BanditOutcome(estimates: np.ndarray, counts: np.ndarray, cumulative_reward: float, cumulative_regret: float)` and three functions with the shared call shape `run_*(env, steps, rng, metrics) -> BanditOutcome`.

- [ ] **Step 1: Write failing focused algorithm tests**

Create `tests/algorithms/bandits/test_algorithms.py`:

```python
import numpy as np

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.random_agent import run_random
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MemoryMetricSink


def test_random_agent_takes_requested_number_of_steps():
    # Arrange
    env = StationaryBanditEnv(arms=3, horizon=12, reward_std=0.0)
    env.reset(seed=4)

    # Act
    outcome = run_random(env, 12, np.random.default_rng(4), MemoryMetricSink())

    # Assert
    assert int(outcome.counts.sum()) == 12


def test_greedy_updates_incremental_sample_mean():
    # Arrange
    env = StationaryBanditEnv(arms=2, horizon=4, reward_std=0.0)
    env.reset(seed=8)

    # Act
    outcome = run_greedy(env, 4, np.random.default_rng(8), MemoryMetricSink())

    # Assert
    assert np.allclose(outcome.estimates[outcome.counts > 0], env.arm_means[outcome.counts > 0])


def test_epsilon_one_explores_more_than_one_arm():
    # Arrange
    env = StationaryBanditEnv(arms=4, horizon=100, reward_std=0.0)
    env.reset(seed=2)

    # Act
    outcome = run_epsilon_greedy(
        env,
        100,
        np.random.default_rng(2),
        MemoryMetricSink(),
        epsilon=1.0,
    )

    # Assert
    assert np.count_nonzero(outcome.counts) > 1
```

- [ ] **Step 2: Run RED and commit the failing algorithm contract**

Run:

```powershell
uv run pytest tests/algorithms/bandits/test_algorithms.py -v
```

Expected: FAIL during import because the bandit algorithm modules do not exist.

Commit:

```powershell
git add tests/algorithms/bandits/test_algorithms.py
git commit -m "test: define basic bandit algorithms"
```

- [ ] **Step 3: Implement the compact shared result and update helper**

Create `src/rl_fun/algorithms/bandits/types.py`:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BanditOutcome:
    estimates: np.ndarray
    counts: np.ndarray
    cumulative_reward: float
    cumulative_regret: float


def update_estimate(estimates: np.ndarray, counts: np.ndarray, action: int, reward: float) -> None:
    counts[action] += 1
    estimates[action] += (reward - estimates[action]) / counts[action]


def random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates))
```

- [ ] **Step 4: Implement each explicit learning loop**

Create `src/rl_fun/algorithms/bandits/random_agent.py`:

```python
import numpy as np

from rl_fun.algorithms.bandits.types import BanditOutcome, update_estimate
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MetricSink


def run_random(
    env: StationaryBanditEnv,
    steps: int,
    rng: np.random.Generator,
    metrics: MetricSink,
) -> BanditOutcome:
    if steps > env.horizon:
        raise ValueError("steps must not exceed environment horizon")
    estimates = np.zeros(env.arms, dtype=np.float64)
    counts = np.zeros(env.arms, dtype=np.int64)
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_actions = 0
    for step in range(1, steps + 1):
        action = int(rng.integers(env.arms))
        _, reward, terminated, truncated, info = env.step(action)
        update_estimate(estimates, counts, action, reward)
        cumulative_reward += reward
        cumulative_regret += float(info["instant_regret"])
        optimal_actions += int(action == info["optimal_action"])
        metrics.log(
            step,
            {
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "regret": cumulative_regret,
                "optimal_action_rate": optimal_actions / step,
            },
        )
        if terminated or truncated:
            break
    return BanditOutcome(estimates, counts, cumulative_reward, cumulative_regret)
```

Create `src/rl_fun/algorithms/bandits/greedy.py` with the same explicit update loop and this complete entry point:

```python
import numpy as np

from rl_fun.algorithms.bandits.types import BanditOutcome, random_argmax, update_estimate
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MetricSink


def run_greedy(
    env: StationaryBanditEnv,
    steps: int,
    rng: np.random.Generator,
    metrics: MetricSink,
) -> BanditOutcome:
    if steps > env.horizon:
        raise ValueError("steps must not exceed environment horizon")
    estimates = np.zeros(env.arms, dtype=np.float64)
    counts = np.zeros(env.arms, dtype=np.int64)
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_actions = 0
    for step in range(1, steps + 1):
        action = random_argmax(estimates, rng)
        _, reward, terminated, truncated, info = env.step(action)
        update_estimate(estimates, counts, action, reward)
        cumulative_reward += reward
        cumulative_regret += float(info["instant_regret"])
        optimal_actions += int(action == info["optimal_action"])
        metrics.log(
            step,
            {
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "regret": cumulative_regret,
                "optimal_action_rate": optimal_actions / step,
            },
        )
        if terminated or truncated:
            break
    return BanditOutcome(estimates, counts, cumulative_reward, cumulative_regret)
```

Create `src/rl_fun/algorithms/bandits/epsilon_greedy.py`:

```python
import numpy as np

from rl_fun.algorithms.bandits.types import BanditOutcome, random_argmax, update_estimate
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MetricSink


def run_epsilon_greedy(
    env: StationaryBanditEnv,
    steps: int,
    rng: np.random.Generator,
    metrics: MetricSink,
    epsilon: float,
) -> BanditOutcome:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if steps > env.horizon:
        raise ValueError("steps must not exceed environment horizon")
    estimates = np.zeros(env.arms, dtype=np.float64)
    counts = np.zeros(env.arms, dtype=np.int64)
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_actions = 0
    for step in range(1, steps + 1):
        exploring = rng.random() < epsilon
        action = int(rng.integers(env.arms)) if exploring else random_argmax(estimates, rng)
        _, reward, terminated, truncated, info = env.step(action)
        update_estimate(estimates, counts, action, reward)
        cumulative_reward += reward
        cumulative_regret += float(info["instant_regret"])
        optimal_actions += int(action == info["optimal_action"])
        metrics.log(
            step,
            {
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "regret": cumulative_regret,
                "optimal_action_rate": optimal_actions / step,
            },
        )
        if terminated or truncated:
            break
    return BanditOutcome(estimates, counts, cumulative_reward, cumulative_regret)
```

Keep all three loops visible. Repetition here is intentional: each introductory algorithm should be readable without jumping through a general Trainer abstraction.

- [ ] **Step 5: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/algorithms/bandits/test_algorithms.py -v
uv run ruff check src tests
```

Expected: all three algorithm tests pass and Ruff reports no violations.

Commit:

```powershell
git add src/rl_fun/algorithms tests/algorithms/bandits/test_algorithms.py
git commit -m "feat: add readable bandit algorithms"
```

---

### Task 5: Execute and persist one configured bandit run

**Files:**
- Create: `src/rl_fun/experiments/result.py`
- Create: `src/rl_fun/experiments/bandit.py`
- Test: `tests/experiments/test_bandit_run.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, one seed, and an optional fixed run identifier for tests.
- Produces: `RunSummary(status, seed, run_dir, cumulative_reward, cumulative_regret, elapsed_seconds, error)` and `run_bandit_once(config, seed, run_id=None) -> RunSummary`.

- [ ] **Step 1: Write the failing end-to-end single-run test**

Create `tests/experiments/test_bandit_run.py`:

```python
import json

from rl_fun.experiments.bandit import run_bandit_once
from rl_fun.experiments.config import ExperimentConfig


def test_one_run_writes_reproducible_artifacts(tmp_path):
    # Arrange
    config = ExperimentConfig(
        name="smoke-bandit",
        algorithm="epsilon_greedy",
        seeds=(17,),
        steps=10,
        workers=1,
        output_root=str(tmp_path),
        parameters={"arms": 3, "epsilon": 0.1, "reward_std": 0.0},
    )

    # Act
    summary = run_bandit_once(config, seed=17, run_id="fixed-run")

    # Assert
    assert json.loads((summary.run_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "success"


def test_same_seed_produces_same_numeric_summary(tmp_path):
    # Arrange
    config = ExperimentConfig(
        name="repeatable",
        algorithm="greedy",
        seeds=(9,),
        steps=20,
        workers=1,
        output_root=str(tmp_path),
        parameters={"arms": 4, "reward_std": 0.0},
    )

    # Act
    first = run_bandit_once(config, seed=9, run_id="first")
    second = run_bandit_once(config, seed=9, run_id="second")

    # Assert
    assert (first.cumulative_reward, first.cumulative_regret) == (
        second.cumulative_reward,
        second.cumulative_regret,
    )
```

- [ ] **Step 2: Run RED and commit the failing run contract**

Run:

```powershell
uv run pytest tests/experiments/test_bandit_run.py -v
```

Expected: FAIL during import because `run_bandit_once` does not exist.

Commit:

```powershell
git add tests/experiments/test_bandit_run.py
git commit -m "test: define persisted bandit run"
```

- [ ] **Step 3: Implement the summary and single-run orchestration**

Define in `src/rl_fun/experiments/result.py`:

```python
@dataclass(frozen=True, slots=True)
class RunSummary:
    status: Literal["success", "failure", "cancelled"]
    seed: int
    run_dir: Path
    cumulative_reward: float | None
    cumulative_regret: float | None
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["run_dir"] = str(self.run_dir)
        return result
```

Create `src/rl_fun/experiments/bandit.py`:

```python
from __future__ import annotations

import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.random_agent import run_random
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.result import RunSummary
from rl_fun.tracking.artifacts import collect_metadata, create_run_dir, write_json_atomic
from rl_fun.tracking.metrics import JsonlMetricSink


def run_bandit_once(
    config: ExperimentConfig,
    seed: int,
    run_id: str | None = None,
) -> RunSummary:
    config.validate()
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not present in the experiment configuration")
    identifier = run_id or f"seed-{seed}-{uuid.uuid4().hex[:12]}"
    run_dir = create_run_dir(Path(config.output_root), config.name, identifier)
    write_json_atomic(run_dir / "config.json", config.to_dict())
    write_json_atomic(run_dir / "metadata.json", collect_metadata(seed))
    started = time.perf_counter()
    env: StationaryBanditEnv | None = None
    try:
        arms = int(config.parameters.get("arms", 10))
        reward_std = float(config.parameters.get("reward_std", 1.0))
        env = StationaryBanditEnv(arms=arms, horizon=config.steps, reward_std=reward_std)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        with JsonlMetricSink(run_dir / "metrics.jsonl") as metrics:
            match config.algorithm:
                case "random":
                    outcome = run_random(env, config.steps, rng, metrics)
                case "greedy":
                    outcome = run_greedy(env, config.steps, rng, metrics)
                case "epsilon_greedy":
                    epsilon = float(config.parameters.get("epsilon", 0.1))
                    outcome = run_epsilon_greedy(
                        env,
                        config.steps,
                        rng,
                        metrics,
                        epsilon=epsilon,
                    )
                case unknown:
                    raise ValueError(f"unsupported bandit algorithm: {unknown}")
        summary = RunSummary(
            status="success",
            seed=seed,
            run_dir=run_dir,
            cumulative_reward=outcome.cumulative_reward,
            cumulative_regret=outcome.cumulative_regret,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception:
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
        summary = RunSummary(
            status="failure",
            seed=seed,
            run_dir=run_dir,
            cumulative_reward=None,
            cumulative_regret=None,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
        )
    finally:
        if env is not None:
            env.close()
    write_json_atomic(run_dir / "summary.json", summary.to_dict())
    return summary
```

The explicit `match` is the complete first-milestone dispatch mechanism; do not create a generic algorithm registry.

- [ ] **Step 4: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/experiments/test_bandit_run.py -v
uv run ruff check src tests
```

Expected: both single-run tests pass and Ruff reports no violations.

Commit:

```powershell
git add src/rl_fun/experiments tests/experiments/test_bandit_run.py
git commit -m "feat: persist configured bandit runs"
```

---

### Task 6: Add serial and process-parallel multi-seed execution

**Files:**
- Create: `src/rl_fun/experiments/runner.py`
- Test: `tests/experiments/test_runner.py`

**Interfaces:**
- Consumes: `ExperimentConfig`.
- Produces: `BatchSummary(runs: tuple[RunSummary, ...])`, `run_many(config: ExperimentConfig) -> BatchSummary`, and a top-level picklable `_run_seed(config_dict: dict[str, object], seed: int) -> RunSummary` worker.

- [ ] **Step 1: Write failing batch tests**

Create `tests/experiments/test_runner.py`:

```python
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many


def make_config(tmp_path, workers):
    return ExperimentConfig(
        name="batch-bandit",
        algorithm="random",
        seeds=(101, 202),
        steps=8,
        workers=workers,
        output_root=str(tmp_path),
        parameters={"arms": 3, "reward_std": 0.0},
    )


def test_serial_batch_returns_every_seed(tmp_path):
    # Arrange
    config = make_config(tmp_path, workers=1)

    # Act
    batch = run_many(config)

    # Assert
    assert {run.seed for run in batch.runs} == {101, 202}


def test_parallel_batch_uses_isolated_directories(tmp_path):
    # Arrange
    config = make_config(tmp_path, workers=2)

    # Act
    batch = run_many(config)

    # Assert
    assert len({run.run_dir for run in batch.runs}) == 2


def test_batch_summary_reports_success_count(tmp_path):
    # Arrange
    config = make_config(tmp_path, workers=2)

    # Act
    batch = run_many(config)

    # Assert
    assert batch.success_count == 2
```

- [ ] **Step 2: Run RED and commit the failing batch contract**

Run:

```powershell
uv run pytest tests/experiments/test_runner.py -v
```

Expected: FAIL during import because `run_many` does not exist.

Commit:

```powershell
git add tests/experiments/test_runner.py
git commit -m "test: define multi-seed runner"
```

- [ ] **Step 3: Implement Windows-safe parallel execution**

Define:

```python
@dataclass(frozen=True, slots=True)
class BatchSummary:
    runs: tuple[RunSummary, ...]

    @property
    def success_count(self) -> int:
        return sum(run.status == "success" for run in self.runs)

    @property
    def failure_count(self) -> int:
        return sum(run.status == "failure" for run in self.runs)
```

`run_many` runs directly when `workers == 1`. Otherwise use:

```python
context = multiprocessing.get_context("spawn")
with ProcessPoolExecutor(max_workers=min(config.workers, len(config.seeds)), mp_context=context) as pool:
    futures = [pool.submit(_run_seed, config.to_dict(), seed) for seed in config.seeds]
    runs = tuple(future.result() for future in futures)
```

The top-level `_run_seed` reconstructs `ExperimentConfig` and calls `run_bandit_once`. Catch `KeyboardInterrupt` in the parent, cancel pending futures, and re-raise so the command-line entry point can return exit code 130. Sort the returned summaries by seed before constructing `BatchSummary`.

- [ ] **Step 4: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/experiments/test_runner.py -v
uv run ruff check src tests
```

Expected: serial and spawned-process tests pass on Windows.

Commit:

```powershell
git add src/rl_fun/experiments/runner.py tests/experiments/test_runner.py
git commit -m "feat: run independent seeds in parallel"
```

---

### Task 7: Compare runs and expose script entry points

**Files:**
- Create: `src/rl_fun/experiments/compare.py`
- Create: `scripts/train.py`
- Create: `scripts/compare.py`
- Test: `tests/experiments/test_compare.py`
- Test: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: `BatchSummary`, run directories, and a JSON configuration path.
- Produces: `aggregate_batch(batch: BatchSummary) -> dict[str, float]`, a CSV summary, a PNG comparison plot, and command exit codes `0` for complete success, `1` for run failures, `2` for invalid input, and `130` for interruption.

- [ ] **Step 1: Write failing comparison and CLI smoke tests**

Create `tests/experiments/test_compare.py`:

```python
from pathlib import Path

from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.result import RunSummary
from rl_fun.experiments.runner import BatchSummary


def test_aggregate_batch_calculates_mean_reward():
    # Arrange
    runs = (
        RunSummary("success", 1, Path("one"), 2.0, 1.0, 0.1),
        RunSummary("success", 2, Path("two"), 4.0, 3.0, 0.2),
    )

    # Act
    aggregate = aggregate_batch(BatchSummary(runs))

    # Assert
    assert aggregate["mean_cumulative_reward"] == 3.0
```

Create `tests/integration/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run RED and commit the failing public behavior**

Run:

```powershell
uv run pytest tests/experiments/test_compare.py tests/integration/test_cli.py -v
```

Expected: FAIL because `aggregate_batch` and the scripts do not exist.

Commit:

```powershell
git add tests/experiments/test_compare.py tests/integration/test_cli.py
git commit -m "test: define experiment comparison CLI"
```

- [ ] **Step 3: Implement aggregation and scripts**

Create `src/rl_fun/experiments/compare.py`. `aggregate_batch` ignores failed runs for numeric aggregation:

```python
import numpy as np

from rl_fun.experiments.runner import BatchSummary


def aggregate_batch(batch: BatchSummary) -> dict[str, float]:
    rewards = [
        run.cumulative_reward
        for run in batch.runs
        if run.status == "success" and run.cumulative_reward is not None
    ]
    regrets = [
        run.cumulative_regret
        for run in batch.runs
        if run.status == "success" and run.cumulative_regret is not None
    ]
    return {
        "run_count": float(len(batch.runs)),
        "success_count": float(batch.success_count),
        "failure_count": float(batch.failure_count),
        "mean_cumulative_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "std_cumulative_reward": float(np.std(rewards)) if rewards else float("nan"),
        "mean_cumulative_regret": float(np.mean(regrets)) if regrets else float("nan"),
    }
```

Create `scripts/train.py`:

```python
import argparse
import json
from pathlib import Path

from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an RL Fun experiment")
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    try:
        config = ExperimentConfig.from_json(arguments.config)
        batch = run_many(config)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(aggregate_batch(batch), indent=2, sort_keys=True))
    return 0 if batch.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/compare.py` with two focused helpers:

```python
def read_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = np.asarray([event["step"] for event in events], dtype=np.int64)
    rewards = np.asarray(
        [event["metrics"]["cumulative_reward"] for event in events],
        dtype=np.float64,
    )
    return steps, rewards


def common_curves(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    curves = [read_curve(path) for path in run_dirs]
    common_steps = set(curves[0][0].tolist())
    for steps, _ in curves[1:]:
        common_steps.intersection_update(steps.tolist())
    ordered = np.asarray(sorted(common_steps), dtype=np.int64)
    matrix = np.asarray(
        [[rewards[np.flatnonzero(steps == step)[0]] for step in ordered] for steps, rewards in curves]
    )
    return ordered, matrix
```

Its `main()` accepts `run_dirs` with `nargs="+"` and optional `--output-dir`, writes one CSV row per common step with `mean_cumulative_reward` and `std_cumulative_reward`, and plots the mean with `fill_between(mean - std, mean + std)`. It returns `2` when no successful curves or no common steps exist. It guards execution with `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 4: Run GREEN checks and commit**

Run:

```powershell
uv run pytest tests/experiments/test_compare.py tests/integration/test_cli.py -v
uv run ruff check src scripts tests
```

Expected: comparison and CLI tests pass; Ruff reports no violations.

Commit:

```powershell
git add src/rl_fun/experiments/compare.py scripts tests/experiments/test_compare.py tests/integration/test_cli.py
git commit -m "feat: add experiment scripts and comparison"
```

---

### Task 8: Add the first explanatory notebook and user documentation

**Files:**
- Create: `notebooks/01_bandits/01_epsilon_greedy.ipynb`
- Modify: `README.md`
- Test: `tests/integration/test_notebook.py`

**Interfaces:**
- Consumes: the public bandit environment, three algorithm functions, `MemoryMetricSink`, `ExperimentConfig`, and `run_many`.
- Produces: an executable notebook with explanations and plots, plus setup/run instructions in `README.md`.

- [ ] **Step 1: Write the failing notebook smoke test**

Create `tests/integration/test_notebook.py`:

```python
from pathlib import Path
import subprocess
import sys

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
```

- [ ] **Step 2: Run RED and commit the failing notebook contract**

Run:

```powershell
uv run pytest tests/integration/test_notebook.py -v
```

Expected: FAIL because the notebook file does not exist.

Commit:

```powershell
git add tests/integration/test_notebook.py
git commit -m "test: define executable bandit notebook"
```

- [ ] **Step 3: Create the notebook with these cells**

1. Markdown: the stationary k-armed bandit definition and the objective of maximizing expected cumulative reward.
2. Markdown: greedy action selection and epsilon-greedy action selection, including `A_t = argmax_a Q_t(a)` and the incremental update `Q_{n+1}=Q_n+(R_n-Q_n)/N_n`.
3. Code: imports from `rl_fun`, constructs one `StationaryBanditEnv`, and demonstrates one seeded reward:

```python
import numpy as np
import matplotlib.pyplot as plt

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.random_agent import run_random
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many
from rl_fun.tracking.metrics import MemoryMetricSink

demo_env = StationaryBanditEnv(arms=5, horizon=20)
observation, info = demo_env.reset(seed=7)
observation, reward, terminated, truncated, info = demo_env.step(0)
reward, info
```

4. Code: runs random, greedy, and epsilon-greedy with the same seed into separate `MemoryMetricSink` instances:

```python
def run_for_notebook(name, seed=42, steps=500):
    env = StationaryBanditEnv(arms=10, horizon=steps)
    env.reset(seed=seed)
    sink = MemoryMetricSink()
    rng = np.random.default_rng(seed)
    if name == "random":
        outcome = run_random(env, steps, rng, sink)
    elif name == "greedy":
        outcome = run_greedy(env, steps, rng, sink)
    else:
        outcome = run_epsilon_greedy(env, steps, rng, sink, epsilon=0.1)
    env.close()
    return outcome, sink


results = {name: run_for_notebook(name) for name in ("random", "greedy", "epsilon_greedy")}
```

5. Code: plots cumulative reward and cumulative regret from the recorded metric events:

```python
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, (_, sink) in results.items():
    steps = [event.step for event in sink.events]
    axes[0].plot(steps, [event.metrics["cumulative_reward"] for event in sink.events], label=name)
    axes[1].plot(steps, [event.metrics["regret"] for event in sink.events], label=name)
axes[0].set_title("Cumulative reward")
axes[1].set_title("Cumulative regret")
for axis in axes:
    axis.set_xlabel("Step")
    axis.legend()
plt.show()
```
6. Markdown: explains why one seed is not evidence of algorithm quality.
7. Code: constructs `ExperimentConfig` for at least ten seeds with `workers=1`, calls `run_many`, and prints `aggregate_batch`:

```python
batch_config = ExperimentConfig(
    name="notebook-epsilon-greedy",
    algorithm="epsilon_greedy",
    seeds=tuple(range(10)),
    steps=500,
    workers=1,
    output_root="../../runs",
    parameters={"arms": 10, "epsilon": 0.1, "reward_std": 1.0},
)
aggregate_batch(run_many(batch_config))
```
8. Markdown: gives three suggested manual experiments: vary epsilon, reward noise, and number of arms.

Keep canonical action-selection and estimate-update code in `src/rl_fun`; notebook cells call it rather than copying it.

- [ ] **Step 4: Write concise README usage**

Document these exact commands:

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check src scripts tests
uv run python scripts/train.py configs/bandit-epsilon-greedy.json
uv run jupyter lab
```

Explain the roles of `src/`, `scripts/`, `notebooks/`, `configs/`, and generated `runs/`. State that PyTorch, Arena, and Neural Inspector belong to subsequent milestones.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
uv run pytest -v
uv run ruff check src scripts tests
uv run python scripts/train.py configs/bandit-epsilon-greedy.json
```

Expected: the full suite passes, Ruff reports no violations, and the example command produces four isolated successful runs plus an aggregate summary.

- [ ] **Step 6: Commit the completed first milestone**

```powershell
git add notebooks/01_bandits/01_epsilon_greedy.ipynb README.md tests/integration/test_notebook.py
git commit -m "docs: add epsilon-greedy learning notebook"
```

---

## Final Verification Gate

- [ ] Run the complete automated suite: `uv run pytest -v`.
- [ ] Run static checks: `uv run ruff check src scripts tests`.
- [ ] Run the committed example config from a clean shell.
- [ ] Confirm `git status --short` contains no generated `runs/`, `.venv/`, cache, notebook checkpoint, or `.superpowers/` entries.
- [ ] Inspect one generated run and confirm it contains `config.json`, `metadata.json`, `metrics.jsonl`, and `summary.json`.
- [ ] Repeat one seed and compare numeric summaries for exact equality on the same machine and locked environment.
- [ ] Confirm the notebook executes from first cell to last without relying on uncommitted local state.
- [ ] Compare the implementation against `docs/superpowers/specs/2026-09-12-rl-workbench-design.md` and record any intentionally deferred acceptance criterion in the milestone handoff.
