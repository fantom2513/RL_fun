# Gymnasium Runtime Integration impl Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Отвязать runtime RL Fun от bandit-задачи и запускать стандартные и локальные Gymnasium-среды через одну конфигурацию, общий runner, evaluation и playback.

**Architecture:** Gymnasium registry отвечает за создание сред, а небольшой локальный registry сопоставляет ID алгоритма с функцией и проверкой совместимости. Runner управляет конфигурацией, артефактами, процессами и закрытием ресурсов; алгоритм сам управляет `reset/step`, а общий `Policy`/rollout используется только для оценки и просмотра.

**Tech Stack:** Python 3.12+, Gymnasium 1.3+, NumPy, pytest, Ruff, uv; Matplotlib/Jupyter остаются в существующей notebook-группе.

**Spec:** `docs/superpowers/specs/2026-09-13-gymnasium-runtime-design.md`

## Глобальные ограничения

- Перед реализацией прочитать спек целиком.
- Не добавлять генератор сред, чат, игровой движок, Arena, Neural Inspector или универсальный Trainer.
- Не добавлять PyTorch, DQN, PPO, Stable-Baselines3, TensorBoard, Hydra, MLflow или PettingZoo.
- Все notebook-пояснения, README, пользовательский CLI help и подписи графиков писать на русском; Python API и metric keys оставлять английскими.
- Не сохранять `render_mode` в `EnvironmentConfig.kwargs`: режим задаётся вызывающим сценарием.
- Алгоритм, а не runner, владеет `env.reset()` и `env.step()`.
- Каждый проектный environment ID имеет вид `RLFun/<Name>-vN`.
- Каждый новый логический блок проходит RED → отдельный `test:` commit → GREEN → отдельный `feat:`/`refactor:` commit.
- Нельзя менять RED-тест ради прохождения GREEN; ошибочный тест сначала обсуждается с пользователем.
- До Task 8 получить отдельное разрешение на добавление `gymnasium[classic-control]`; без разрешения остановиться после Task 7.

## Карта файлов

Создаются:

- `src/rl_fun/environments/factory.py` — единственная фабрика Gymnasium-сред.
- `src/rl_fun/algorithms/registry.py` — алгоритмические ID, adapters и compatibility checks.
- `src/rl_fun/policies.py` — `Policy` protocol и `RandomPolicy`.
- `src/rl_fun/rollouts.py` — один environment-neutral эпизод.
- `src/rl_fun/experiments/run.py` — общий `run_once` и запись артефактов.
- `scripts/evaluate.py` — headless evaluation.
- `scripts/play.py` — human playback одной среды.
- `configs/cartpole-random.json` — интеграционный пример.
- `docs/environment-authoring.md` — русская памятка по новой среде.
- focused-тесты рядом с существующими test-модулями.

Изменяются:

- `src/rl_fun/experiments/config.py` — nested config v2.
- `src/rl_fun/environments/__init__.py` — безопасная локальная регистрация.
- `src/rl_fun/algorithms/bandits/*.py` — seed/reset contract и namespace метрик.
- `src/rl_fun/experiments/result.py` — общий `RunSummary`.
- `src/rl_fun/experiments/runner.py` — multi-seed поверх `run_once`.
- `src/rl_fun/experiments/compare.py` — aggregation по имени метрики.
- `scripts/train.py`, `scripts/compare.py` — CLI общего runtime.
- `configs/bandit-epsilon-greedy.json`, `README.md`, `notebooks/01_bandits/01_epsilon_greedy.ipynb`.

После миграции удаляется только если больше нет callers:

- `src/rl_fun/experiments/bandit.py` — прежняя bandit-specific orchestration.

## Task 1: Typed-конфигурация версии 2

- Modify: `src/rl_fun/experiments/config.py`
- Modify: `tests/experiments/test_config.py`
- Produces: `JSONValue`, `EnvironmentConfig`, `AlgorithmConfig`, `RunConfig`, `EvaluationConfig`, `ExperimentConfig`.

- [ ] **Step 1: заменить тесты конфигурации на v2 и добавить строгие ошибки**

```python
def valid_values() -> dict[str, object]:
    return {
        "name": "cartpole-random",
        "environment": {"id": "CartPole-v1", "kwargs": {}},
        "algorithm": {"id": "random", "kwargs": {}},
        "run": {
            "seeds": [11, 22],
            "total_steps": 100,
            "workers": 2,
            "output_root": "runs",
        },
        "evaluation": {"episodes": 5, "max_episode_steps": 500},
    }


def test_config_v2_round_trip_preserves_sections():
    config = ExperimentConfig.from_dict(valid_values())
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_old_flat_schema_has_actionable_error():
    with pytest.raises(ValueError, match="configuration version 2"):
        ExperimentConfig.from_dict({"name": "old", "algorithm": "random"})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["run"].update(total_steps=0), "total_steps must be positive"),
        (lambda value: value["run"].update(seeds=[1, 1]), "seeds must be unique"),
        (lambda value: value["algorithm"].update(id=""), "algorithm.id must not be empty"),
        (lambda value: value.update(extra=True), "unknown top-level keys"),
    ],
)
def test_config_rejects_invalid_values(mutate, message):
    values = valid_values()
    mutate(values)
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_dict(values)
```

- [ ] **Step 2: запустить RED-тесты**

Run: `uv run pytest tests/experiments/test_config.py -v`
Expected: FAIL, потому что старый `ExperimentConfig` не понимает nested sections.

- [ ] **Step 3: зафиксировать RED**

Run: `git add tests/experiments/test_config.py`
Run: `git commit -m "test: specify experiment config v2"`

- [ ] **Step 4: реализовать dataclass и строгий parser**

```python
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    id: str
    kwargs: dict[str, JSONValue] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    id: str
    kwargs: dict[str, JSONValue] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class RunConfig:
    seeds: tuple[int, ...]
    total_steps: int
    workers: int = 1
    output_root: str = "runs"

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes: int = 5
    max_episode_steps: int | None = None

@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    run: RunConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
```

Реализовать `_require_keys`, `_reject_unknown_keys`, `_require_positive_int`, `_require_json_mapping`; отдельно отклонять `bool` там, где ожидается `int`. `from_dict()` принимает только v2, `to_dict()` выдаёт JSON-совместимые списки и словари, `from_json()` сохраняет текущий UTF-8 путь.

- [ ] **Step 5: запустить GREEN и lint**

Run: `uv run pytest tests/experiments/test_config.py -v`
Expected: PASS.
Run: `uv run ruff check src/rl_fun/experiments/config.py tests/experiments/test_config.py`
Expected: exit 0.

- [ ] **Step 6: зафиксировать GREEN**

Run: `git add src/rl_fun/experiments/config.py`
Run: `git commit -m "feat: add typed experiment config v2"`

## Task 2: Gymnasium registration и фабрика

- Create: `src/rl_fun/environments/factory.py`
- Modify: `src/rl_fun/environments/__init__.py`
- Modify: `tests/environments/test_bandit.py`
- Create: `tests/environments/test_factory.py`
- Consumes: `EnvironmentConfig` из Task 1.
- Produces: `register_environments() -> None`, `make_environment(config, render_mode=None) -> gym.Env`.

- [ ] **Step 1: написать contract-тесты регистрации и фабрики**

```python
def test_project_bandit_is_registered_and_checked():
    register_environments()
    register_environments()
    env = gym.make("RLFun/StationaryBandit-v0", arms=3, horizon=5, reward_std=0.0)
    try:
        check_env(env.unwrapped)
    finally:
        env.close()


def test_factory_creates_cartpole_without_render_mode():
    env = make_environment(EnvironmentConfig(id="CartPole-v1"))
    try:
        assert env.spec is not None and env.spec.id == "CartPole-v1"
    finally:
        env.close()


def test_factory_error_contains_environment_id():
    with pytest.raises(ValueError, match="Missing-v0"):
        make_environment(EnvironmentConfig(id="Missing-v0"))
```

Добавить тест, который временно регистрирует тот же локальный ID с другим entry point и ожидает явную ошибку от `register_environments()`; после теста восстановить registry через `monkeypatch`, не оставляя process-global загрязнения.

- [ ] **Step 2: запустить RED и закоммитить тесты**

Run: `uv run pytest tests/environments/test_bandit.py tests/environments/test_factory.py -v`
Expected: FAIL на отсутствующих `register_environments` и `make_environment`.
Run: `git add tests/environments/test_bandit.py tests/environments/test_factory.py`
Run: `git commit -m "test: specify Gymnasium environment integration"`

- [ ] **Step 3: реализовать безопасную регистрацию**

```python
LOCAL_ENVIRONMENTS = {
    "RLFun/StationaryBandit-v0": "rl_fun.environments.bandit:StationaryBanditEnv",
}

def register_environments() -> None:
    for environment_id, entry_point in LOCAL_ENVIRONMENTS.items():
        existing = gym.registry.get(environment_id)
        if existing is None:
            gym.register(id=environment_id, entry_point=entry_point)
        elif existing.entry_point != entry_point:
            raise RuntimeError(
                f"environment {environment_id!r} is already registered "
                f"with entry point {existing.entry_point!r}"
            )
```

`rl_fun.environments.__init__` вызывает `register_environments()` при импорте. Фабрика импортирует пакет регистрации, копирует `config.kwargs`, добавляет `render_mode` только когда он не `None`, вызывает `gym.make()` и оборачивает исходное исключение с ID и отсортированными именами kwargs через exception chaining.

- [ ] **Step 4: запустить GREEN, env checker и lint**

Run: `uv run pytest tests/environments -v`
Expected: PASS.
Run: `uv run ruff check src/rl_fun/environments tests/environments`
Expected: exit 0.

- [ ] **Step 5: зафиксировать GREEN**

Run: `git add src/rl_fun/environments tests/environments`
Run: `git commit -m "feat: register and create Gymnasium environments"`

## Task 3: Policy и воспроизводимый rollout

- Create: `src/rl_fun/policies.py`
- Create: `src/rl_fun/rollouts.py`
- Create: `tests/test_rollouts.py`
- Produces: `Policy`, `RandomPolicy`, `EpisodeResult`, `rollout_episode`.

- [ ] **Step 1: написать RED-тесты случайной policy и CartPole rollout**

```python
def test_random_policy_repeats_actions_for_same_seed():
    first = RandomPolicy()
    second = RandomPolicy()
    space_a = gym.spaces.Discrete(4)
    space_b = gym.spaces.Discrete(4)
    first.reset(17, space_a)
    second.reset(17, space_b)
    assert [first.act(None, False) for _ in range(8)] == [
        second.act(None, False) for _ in range(8)
    ]


def test_cartpole_rollout_is_reproducible():
    first_env = gym.make("CartPole-v1")
    second_env = gym.make("CartPole-v1")
    try:
        first = rollout_episode(first_env, RandomPolicy(), seed=23, max_episode_steps=20)
        second = rollout_episode(second_env, RandomPolicy(), seed=23, max_episode_steps=20)
        assert first == second
    finally:
        first_env.close()
        second_env.close()


def test_rollout_obeys_safety_limit():
    env = gym.make("CartPole-v1")
    try:
        result = rollout_episode(env, RandomPolicy(), seed=5, max_episode_steps=1)
        assert result.length == 1
    finally:
        env.close()
```

- [ ] **Step 2: запустить RED и закоммитить тест**

Run: `uv run pytest tests/test_rollouts.py -v`
Expected: FAIL из-за отсутствующих модулей.
Run: `git add tests/test_rollouts.py`
Run: `git commit -m "test: specify reproducible policy rollouts"`

- [ ] **Step 3: реализовать минимальные contracts**

```python
class Policy(Protocol):
    def reset(self, seed: int, action_space: gym.Space) -> None: ...
    def act(self, observation: object, deterministic: bool) -> object: ...

@dataclass(frozen=True, slots=True)
class EpisodeResult:
    reward: float
    length: int
    terminated: bool
    truncated: bool
    reached_safety_limit: bool
```

`RandomPolicy.reset()` вызывает `action_space.seed(seed)` и сохраняет space; `act()` падает с понятной ошибкой до `reset()` и иначе возвращает `sample()`. `rollout_episode()` сбрасывает policy и env одним seed, суммирует scalar reward, корректно различает Gymnasium `terminated`/`truncated` и собственный safety limit. `max_episode_steps=None` означает отсутствие дополнительного лимита.

- [ ] **Step 4: запустить GREEN и lint**

Run: `uv run pytest tests/test_rollouts.py -v`
Expected: PASS.
Run: `uv run ruff check src/rl_fun/policies.py src/rl_fun/rollouts.py tests/test_rollouts.py`
Expected: exit 0.

- [ ] **Step 5: зафиксировать GREEN**

Run: `git add src/rl_fun/policies.py src/rl_fun/rollouts.py`
Run: `git commit -m "feat: add reproducible policy rollouts"`

## Task 4: Общая граница алгоритмов и registry

- Create: `src/rl_fun/algorithms/registry.py`
- Modify: `src/rl_fun/algorithms/bandits/random_agent.py`
- Modify: `src/rl_fun/algorithms/bandits/greedy.py`
- Modify: `src/rl_fun/algorithms/bandits/epsilon_greedy.py`
- Modify: `tests/algorithms/bandits/test_algorithms.py`
- Create: `tests/algorithms/test_registry.py`
- Consumes: `JSONValue`, `RandomPolicy`, `rollout_episode`.
- Produces: `AlgorithmResult`, `AlgorithmDefinition`, `get_algorithm`, IDs `random`, `bandit_greedy`, `bandit_epsilon`.

- [ ] **Step 1: написать RED-тесты registry и совместимости**

```python
def test_unknown_algorithm_lists_supported_ids():
    with pytest.raises(ValueError, match="bandit_epsilon.*bandit_greedy.*random"):
        get_algorithm("missing")


def test_bandit_algorithm_rejects_cartpole():
    env = gym.make("CartPole-v1")
    try:
        definition = get_algorithm("bandit_greedy")
        with pytest.raises(ValueError, match="bandit_greedy.*CartPole-v1"):
            definition.validate(env, "CartPole-v1")
    finally:
        env.close()


def test_random_algorithm_finishes_requested_steps():
    env = gym.make("CartPole-v1")
    sink = MemoryMetricSink()
    try:
        result = get_algorithm("random").run(
            env, 25, 7, np.random.default_rng(7), sink, {}
        )
        assert result.metrics["train/steps"] == 25.0
    finally:
        env.close()
```

В bandit tests добавить проверку, что первый reset передаёт seed и последний event содержит `train/cumulative_reward`, `train/cumulative_regret`, `train/optimal_action_rate`.

- [ ] **Step 2: запустить RED и закоммитить тесты**

Run: `uv run pytest tests/algorithms -v`
Expected: FAIL на отсутствующем registry и старых metric keys.
Run: `git add tests/algorithms`
Run: `git commit -m "test: specify algorithm registry contract"`

- [ ] **Step 3: реализовать registry и adapters**

```python
AlgorithmFunction = Callable[
    [gym.Env, int, int, np.random.Generator, MetricSink, Mapping[str, JSONValue]],
    "AlgorithmResult",
]

@dataclass(frozen=True, slots=True)
class AlgorithmResult:
    metrics: dict[str, float]
    policy: Policy | None = None

@dataclass(frozen=True, slots=True)
class AlgorithmDefinition:
    run: AlgorithmFunction
    validate: Callable[[gym.Env, str], None]
```

Каждый adapter первым действием вызывает `env.reset(seed=seed)`. Bandit adapters получают `StationaryBanditEnv` через `env.unwrapped`, валидируют тип и параметры (`epsilon` обязателен только для `bandit_epsilon`; неизвестные kwargs запрещены), затем вызывают учебную функцию. Учебные функции не получают config и registry, но пишут новые `train/...` keys.

Generic random algorithm выполняет эпизоды до суммарного `total_steps`, сидирует каждый эпизод через RNG, не превышает budget и логирует завершённые эпизоды. Финальные метрики: `train/steps`, `train/episodes`, `train/cumulative_reward`; policy в результате — `RandomPolicy()`.

- [ ] **Step 4: запустить GREEN и lint**

Run: `uv run pytest tests/algorithms -v`
Expected: PASS.
Run: `uv run ruff check src/rl_fun/algorithms tests/algorithms`
Expected: exit 0.

- [ ] **Step 5: зафиксировать GREEN**

Run: `git add src/rl_fun/algorithms tests/algorithms`
Run: `git commit -m "feat: add explicit algorithm registry"`

## Task 5: Общий `run_once` и `RunSummary`

- Create: `src/rl_fun/experiments/run.py`
- Modify: `src/rl_fun/experiments/result.py`
- Replace: `tests/experiments/test_bandit_run.py` → `tests/experiments/test_run.py`
- Consumes: config v2, `make_environment`, `get_algorithm`, `JsonlMetricSink`, artifact helpers.
- Produces: `run_once(config, seed, run_id=None) -> RunSummary`.

- [ ] **Step 1: написать RED-тесты общего запуска**

```python
def bandit_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict({
        "name": "bandit-smoke",
        "environment": {
            "id": "RLFun/StationaryBandit-v0",
            "kwargs": {"arms": 3, "horizon": 10, "reward_std": 0.0},
        },
        "algorithm": {"id": "bandit_epsilon", "kwargs": {"epsilon": 0.1}},
        "run": {
            "seeds": [17], "total_steps": 10, "workers": 1,
            "output_root": str(tmp_path),
        },
        "evaluation": {"episodes": 2, "max_episode_steps": 10},
    })


def test_run_once_writes_generic_success_summary(tmp_path):
    summary = run_once(bandit_config(tmp_path), 17, run_id="fixed")
    written = json.loads((summary.run_dir / "summary.json").read_text("utf-8"))
    assert written["metrics"] == summary.metrics


def test_run_once_is_reproducible(tmp_path):
    config = bandit_config(tmp_path)
    assert run_once(config, 17, "one").metrics == run_once(config, 17, "two").metrics


def test_run_once_records_environment_failure(tmp_path):
    values = bandit_config(tmp_path).to_dict()
    values["environment"] = {"id": "Missing-v0", "kwargs": {}}
    summary = run_once(ExperimentConfig.from_dict(values), 17, "failed")
    assert summary.status == "failure" and "Missing-v0" in (summary.error or "")
```

- [ ] **Step 2: запустить RED и закоммитить тестовую миграцию**

Run: `uv run pytest tests/experiments/test_run.py -v`
Expected: FAIL на отсутствующем `run_once` и generic summary.
Run: `git add tests/experiments/test_run.py tests/experiments/test_bandit_run.py`
Run: `git commit -m "test: specify generic single-run orchestration"`

- [ ] **Step 3: реализовать общий результат и orchestration**

```python
@dataclass(frozen=True, slots=True)
class RunSummary:
    status: Literal["success", "failure", "cancelled"]
    seed: int
    run_dir: Path
    metrics: dict[str, float]
    elapsed_seconds: float
    error: str | None = None
```

`run_once()` проверяет принадлежность seed конфигу, создаёт уникальный `seed-{seed}-<12 hex>` ID при отсутствии явного ID, пишет config/metadata, создаёт env, запускает compatibility check и algorithm, пишет traceback в `error.txt` на `Exception`, всегда закрывает env и атомарно пишет `summary.json`. `KeyboardInterrupt` не превращается в failure: среда закрывается, исключение идёт наверх.

- [ ] **Step 4: запустить GREEN и lint**

Run: `uv run pytest tests/experiments/test_run.py tests/tracking -v`
Expected: PASS.
Run: `uv run ruff check src/rl_fun/experiments/run.py src/rl_fun/experiments/result.py tests/experiments/test_run.py`
Expected: exit 0.

- [ ] **Step 5: зафиксировать GREEN**

Run: `git add src/rl_fun/experiments/run.py src/rl_fun/experiments/result.py`
Run: `git commit -m "feat: generalize single-run orchestration"`

## Task 6: Multi-seed runner и aggregation выбранной метрики

- Modify: `src/rl_fun/experiments/runner.py`
- Modify: `src/rl_fun/experiments/compare.py`
- Modify: `tests/experiments/test_runner.py`
- Modify: `tests/experiments/test_compare.py`
- Delete: `src/rl_fun/experiments/bandit.py` после проверки отсутствия imports.
- Consumes: `run_once`, generic `RunSummary`, config v2.
- Produces: `run_many(config) -> BatchSummary`, `aggregate_batch(batch, metric) -> dict[str, float]`.

- [ ] **Step 1: мигрировать RED-тесты**

```python
def test_aggregate_batch_uses_requested_metric():
    runs = (
        RunSummary("success", 1, Path("one"), {"score": 2.0}, 0.1),
        RunSummary("success", 2, Path("two"), {"score": 4.0}, 0.2),
    )
    aggregate = aggregate_batch(BatchSummary(runs), "score")
    assert aggregate["mean"] == 3.0


def test_aggregate_batch_rejects_missing_metric():
    batch = BatchSummary((RunSummary("success", 1, Path("one"), {}, 0.1),))
    with pytest.raises(ValueError, match="missing"):
        aggregate_batch(batch, "missing")
```

Переписать helper `make_config()` в `test_runner.py` на `RLFun/StationaryBandit-v0` и `bandit_greedy`; сохранить проверки serial/parallel, seed sorting, уникальных директорий и success/failure counts.

- [ ] **Step 2: запустить RED и закоммитить тесты**

Run: `uv run pytest tests/experiments/test_runner.py tests/experiments/test_compare.py -v`
Expected: FAIL, потому что runner вызывает `run_bandit_once`, а aggregation ждёт fixed fields.
Run: `git add tests/experiments/test_runner.py tests/experiments/test_compare.py`
Run: `git commit -m "test: specify generic batch aggregation"`

- [ ] **Step 3: переключить runner и compare**

`_run_seed()` восстанавливает config v2 и вызывает `run_once()`. Serial и spawn paths используют одну функцию. `aggregate_batch(batch, metric)` берёт только successful runs с данным key и возвращает `run_count`, `success_count`, `failure_count`, `metric`, `sample_count`, `mean`, `std`.

Run: `rg "run_bandit_once|experiments\.bandit" src tests scripts`
Expected: no matches. После этого удалить `src/rl_fun/experiments/bandit.py` через patch.

- [ ] **Step 4: запустить GREEN и lint**

Run: `uv run pytest tests/experiments -v`
Expected: PASS.
Run: `uv run ruff check src/rl_fun/experiments tests/experiments`
Expected: exit 0.

- [ ] **Step 5: зафиксировать GREEN**

Run: `git add src/rl_fun/experiments tests/experiments`
Run: `git commit -m "refactor: run batches through generic runtime"`

## Task 7: Train, evaluate и generic compare CLI

- Create: `scripts/evaluate.py`
- Modify: `scripts/train.py`
- Modify: `scripts/compare.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `configs/bandit-epsilon-greedy.json`
- Create: `configs/cartpole-random.json`
- Consumes: config v2, `run_many`, `aggregate_batch`, `make_environment`, `RandomPolicy`, `rollout_episode`.

- [ ] **Step 1: написать RED integration tests**

Добавить subprocess-тесты:

```python
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


def run_script(script: str, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, str(config)],
        capture_output=True, text=True, timeout=30, check=False,
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


def test_train_script_runs_cartpole_config(tmp_path):
    config = cartpole_values(tmp_path)
    completed = run_script("scripts/train.py", write_config(tmp_path, config))
    assert completed.returncode == 0


def test_evaluate_script_writes_episode_metrics(tmp_path):
    config = cartpole_values(tmp_path)
    completed = run_script("scripts/evaluate.py", write_config(tmp_path, config))
    assert completed.returncode == 0
    assert "Средняя награда" in completed.stdout


def test_compare_accepts_metric_name(tmp_path):
    run_dirs = write_metric_runs(tmp_path, "train/cumulative_reward")
    completed = subprocess.run(
        [sys.executable, "scripts/compare.py", *map(str, run_dirs),
         "--metric", "train/cumulative_reward", "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0
```

Обновить существующий bandit CLI smoke на schema v2 и проверить `returncode == 0` плюс generic JSON aggregate.

- [ ] **Step 2: запустить RED и закоммитить tests/config fixtures**

Run: `uv run pytest tests/integration/test_cli.py -v`
Expected: FAIL на v2 CLI и отсутствующем `evaluate.py`.
Run: `git add tests/integration/test_cli.py configs/bandit-epsilon-greedy.json configs/cartpole-random.json`
Run: `git commit -m "test: specify generic experiment CLIs"`

- [ ] **Step 3: реализовать CLI**

`train.py`: русский parser description, optional `--metric` с default `train/cumulative_reward`, печать JSON aggregation; missing aggregate metric — parser error.
`evaluate.py`: config path и optional `--seed`; создаёт директорию `output_root/name/eval-seed-{seed}-<12 hex>`, сохраняет туда config/metadata и `metrics.jsonl`; создаёт env без render, для каждого эпизода получает производный seed и вызывает `rollout_episode`; печатает число эпизодов, путь артефактов и среднюю награду; env закрывается в `finally`.
`compare.py`: обязательный `--metric`; читает этот key, пишет neutral columns `step`, `mean`, `std`; plot label равен metric key, заголовок/оси/легенда на русском.

Оба committed config используют v2; CartPole задаёт `random`, 2 seed, короткий step budget и headless evaluation.

- [ ] **Step 4: запустить GREEN и lint**

Run: `uv run pytest tests/integration/test_cli.py -v`
Expected: PASS.
Run: `uv run ruff check scripts tests/integration/test_cli.py`
Expected: exit 0.

- [ ] **Step 5: зафиксировать GREEN**

Run: `git add scripts configs`
Run: `git commit -m "feat: add generic train evaluate and compare CLIs"`

## Task 8: Human playback CartPole — dependency approval gate

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `scripts/play.py`
- Create: `tests/integration/test_play_cli.py`
- Consumes: config v2, factory, `RandomPolicy`, rollout interaction rules.
- Produces: ручной playback одной среды через Gymnasium `human` render mode.

- [ ] **Step 1: получить разрешение пользователя**

Спросить: «Разрешаешь заменить базовую зависимость на `gymnasium[classic-control]>=1.3.0`, чтобы `play.py` мог открыть human-rendering CartPole?» Не выполнять `uv add` без явного согласия.

- [ ] **Step 2: написать RED-тест CLI без открытия GUI**

```python
class UnsupportedHumanEnv:
    metadata = {"render_modes": []}

    def close(self) -> None:
        pass


def unsupported_human_env(config, render_mode=None):
    assert render_mode == "human"
    return UnsupportedHumanEnv()


def write_play_config(tmp_path: Path) -> Path:
    values = {
        "name": "play-test",
        "environment": {"id": "NoHuman-v0", "kwargs": {}},
        "algorithm": {"id": "random", "kwargs": {}},
        "run": {
            "seeds": [7], "total_steps": 1, "workers": 1,
            "output_root": str(tmp_path / "runs"),
        },
        "evaluation": {"episodes": 1, "max_episode_steps": 1},
    }
    path = tmp_path / "play.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_play_reports_unsupported_render_mode(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(play, "make_environment", unsupported_human_env)
    exit_code = play.main([str(write_play_config(tmp_path))])
    assert exit_code == 2
    assert "human" in capsys.readouterr().err
```

Добавить unit-level seam `main(argv: Sequence[str] | None = None)`; тест подменяет фабрику и не создаёт окно.

- [ ] **Step 3: запустить RED и закоммитить тест**

Run: `uv run pytest tests/integration/test_play_cli.py -v`
Expected: FAIL, потому что `scripts/play.py` отсутствует.
Run: `git add tests/integration/test_play_cli.py`
Run: `git commit -m "test: specify human playback CLI"`

- [ ] **Step 4: добавить approved dependency и реализовать playback**

Run: `uv add "gymnasium[classic-control]>=1.3.0"`
Expected: `pyproject.toml` и `uv.lock` обновлены, `pygame` доступен транзитивно.

`play.py` проверяет наличие `"human"` в `env.metadata["render_modes"]`, использует первый либо указанный `--seed` и вызывает `rollout_episode(env, RandomPolicy(), seed, max_episode_steps)`. Среда закрывается в `finally`. Gymnasium renderer управляет pacing; скрипт не вызывает `sleep()`.

- [ ] **Step 5: запустить автоматическую проверку и ручной smoke**

Run: `uv run pytest tests/integration/test_play_cli.py -v`
Expected: PASS.
Run: `uv run ruff check scripts/play.py tests/integration/test_play_cli.py`
Expected: exit 0.
Manual run: `uv run python scripts/play.py configs/cartpole-random.json --seed 11`
Expected: открывается окно CartPole, эпизод завершается, окно закрывается, процесс возвращает 0.

- [ ] **Step 6: зафиксировать GREEN**

Run: `git add pyproject.toml uv.lock scripts/play.py`
Run: `git commit -m "feat: add CartPole human playback"`

## Task 9: Русская документация и финальная проверка

- Create: `docs/environment-authoring.md`
- Modify: `README.md`
- Modify: `notebooks/01_bandits/01_epsilon_greedy.ipynb`
- Modify: `tests/integration/test_notebook.py`
- Create: `tests/integration/test_russian_docs.py`
- Consumes: все публичные interfaces Tasks 1–8.

- [ ] **Step 1: написать RED checks учебного текста**

```python
def test_readme_and_authoring_guide_use_russian_explanations():
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/environment-authoring.md").read_text(encoding="utf-8")
    assert "Запуск эксперимента" in readme
    assert "Как добавить новую среду" in guide


def test_notebook_markdown_contains_russian_learning_sections():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "Жадный алгоритм" in markdown
    assert "ε-жадный алгоритм" in markdown
```

- [ ] **Step 2: запустить RED и закоммитить тесты**

Run: `uv run pytest tests/integration/test_notebook.py tests/integration/test_russian_docs.py -v`
Expected: FAIL на отсутствующих русских разделах.
Run: `git add tests/integration/test_notebook.py tests/integration/test_russian_docs.py`
Run: `git commit -m "test: require Russian learning materials"`

- [ ] **Step 3: написать конкретную документацию**

README должен объяснять установку через `uv sync --all-groups`, команды train/evaluate/play/compare, структуру config v2, расположение артефактов и границу между runtime и конкретной средой.

`docs/environment-authoring.md` должен содержать: минимальный `gym.Env` skeleton; правила `super().reset(seed=seed)`; observation/action spaces; `terminated` против `truncated`; render modes; регистрацию `RLFun/<Name>-v0`; `check_env`; unit smoke; чек-лист конструктора, episode boundary и manual render smoke.

Notebook сохраняет исполняемый код и результаты первого этапа, но Markdown, заголовки графиков, оси и легенды переводятся на русский. Не добавлять новые алгоритмы или учебный тестовый harness.

- [ ] **Step 4: выполнить focused GREEN**

Run: `uv run pytest tests/integration/test_notebook.py tests/integration/test_russian_docs.py -v`
Expected: PASS.
Run: `uv run ruff check tests/integration`
Expected: exit 0.

- [ ] **Step 5: выполнить полную verification gate**

Run: `uv run pytest -v`
Expected: все тесты PASS, 0 failed.
Run: `uv run ruff check .`
Expected: exit 0.
Run: `uv run python scripts/train.py configs/bandit-epsilon-greedy.json`
Expected: exit 0 и несколько успешных seed.
Run: `uv run python scripts/train.py configs/cartpole-random.json --metric train/cumulative_reward`
Expected: exit 0.
Run: `uv run python scripts/evaluate.py configs/cartpole-random.json --seed 11`
Expected: exit 0 и русская строка со средней наградой.

- [ ] **Step 6: проверить отсутствие старой связности и placeholders**

Run: `rg "run_bandit_once|cumulative_reward: float \| None|TODO|TBD|FIXME" src tests scripts docs/environment-authoring.md README.md`
Expected: no matches, кроме осмысленного metric key `train/cumulative_reward`, который не совпадает с шаблоном поля.

- [ ] **Step 7: зафиксировать документацию**

Run: `git add README.md docs/environment-authoring.md notebooks tests/integration`
Run: `git commit -m "docs: explain Gymnasium workflow in Russian"`

- [ ] **Step 8: проверить итоговый diff**

Run: `git status --short`
Expected: empty.
Run: `git log --oneline --decorate -20`
Expected: для каждой задачи видны отдельные RED и GREEN commits, а последние проверки относятся к текущему HEAD.

## Итог этапа

После Task 9 репозиторий умеет запускать bandit и CartPole через один runtime, оценивать random policy headlessly, показывать одну среду вручную и сравнивать произвольную числовую метрику. Следующая отдельная design-задача выбирает первую собственную визуальную среду; рекомендуемый кандидат — top-down гонка.
