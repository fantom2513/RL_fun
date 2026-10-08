# Racing Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Флот машинок, прогон поколений, отрисовка флота с панелью сети, эталонная эволюция с зафиксированным бенчмарком и ноутбук-заготовка с пояснениями.

**Architecture:** Векторизованные numpy-примитивы (трасса, лучи, динамика) → `RacingFleet` (N машинок) → `run_generation` (вызывает пользовательскую `forward`) → `FleetView`/`NetworkPanel` (pygame). Эталон (`reference.py`) использует тот же `run_generation`, как и ноутбук пользователя. Спека: `docs/superpowers/specs/2026-10-08-racing-evolution-design.md`. Базовая среда: `docs/superpowers/plans/2026-10-08-racing-env.md` (уже реализована).

**Tech Stack:** Python 3.12, numpy, pygame, matplotlib (группа `notebook`), pytest, ruff, uv. **Новых зависимостей нет.** Всё считается на CPU.

**Формат плана:** в отличие от первого плана здесь тесты даны полностью (они определяют поведение), а реализация описана точными сигнатурами и правилами. Реализующий пишет её сам по тестам. Если тест кажется неверным — остановиться и сообщить, не менять молча.

**Conventions:** TDD — красные тесты отдельным коммитом перед реализацией; Conventional Commits; AAA; аннотации типов; ruff (line-length 100); ветка `feat/racing-env`. Трейлер коммита: пустая строка и `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>` (или тот, что соответствует модели исполнителя). Команды из `D:\projects\RL_fun`. Не трогать неотслеживаемый `docs/reference/`. Системы координат те же, что в базовой среде (y вверх, руль «+» — влево).

---

## File Structure

| Файл | Ответственность |
|---|---|
| `src/rl_fun/racing/track.py` (изм.) | `project_many`, `contains_many`, `progress_delta_many` |
| `src/rl_fun/racing/sensors.py` (изм.) | `cast_rays_many` |
| `src/rl_fun/racing/dynamics.py` (изм.) | `KinematicBicycle.step_arrays`, протокол |
| `src/rl_fun/racing/fleet.py` | `RacingFleet` |
| `src/rl_fun/racing/generation.py` | `GenerationResult`, `run_generation` |
| `src/rl_fun/racing/render.py` (изм.) | `Renderer.draw_fleet` |
| `src/rl_fun/racing/fleet_view.py` | `NetworkPanel`, `FleetView`, `ViewClosed` |
| `src/rl_fun/racing/reference.py` | эталонная сеть и эволюция |
| `src/rl_fun/racing/benchmark.py` | `load_benchmark`, `compare`, `plot_comparison` |
| `src/rl_fun/racing/benchmarks/oval.json` | зафиксированный бенчмарк |
| `scripts/make_reference.py` | генерация бенчмарка |
| `notebooks/02_racing/01_evolution.ipynb` | ноутбук-заготовка |
| `tests/racing/test_vectorized.py`, `test_fleet.py`, `test_generation.py`, `test_reference.py`, `test_benchmark.py`, `test_fleet_view.py` | тесты по модулям |
| `tests/integration/test_racing_notebook.py` | структура и выполнение ноутбука |
| `README.md` (изм.) | русский раздел |

---

### Task 1: Векторные примитивы

**Files:** Create `tests/racing/test_vectorized.py`; Modify `track.py`, `sensors.py`, `dynamics.py`.

- [ ] **Step 1: Тесты** — `tests/racing/test_vectorized.py`

```python
import numpy as np
import pytest

from rl_fun.racing.dynamics import KinematicBicycle, VehicleState
from rl_fun.racing.sensors import cast_rays, cast_rays_many
from rl_fun.racing.track import load_track

POINTS = np.array([[60.0, 0.0], [0.0, 35.0], [10.0, 3.0], [100.0, 100.0]])


def test_project_many_matches_scalar_projection():
    track = load_track("oval")

    progress, distance = track.project_many(POINTS)

    for index, point in enumerate(POINTS):
        expected_progress, expected_distance = track.project(point)
        assert progress[index] == pytest.approx(expected_progress)
        assert distance[index] == pytest.approx(expected_distance)


def test_contains_many_matches_scalar_check():
    track = load_track("oval")

    inside = track.contains_many(POINTS)

    assert inside.dtype == bool
    assert inside.tolist() == [track.contains(point) for point in POINTS]


def test_progress_delta_many_wraps_around_start():
    track = load_track("oval")
    length = track.length

    delta = track.progress_delta_many(
        np.array([length - 1.0, 5.0, 10.0]), np.array([1.0, length - 1.0, 20.0])
    )

    assert delta == pytest.approx([2.0, -6.0, 10.0])


def test_cast_rays_many_matches_single_origin_casts():
    track = load_track("wavy")
    origins = np.array([track.centerline[0], track.centerline[10], track.centerline[30]])
    headings = np.array([0.3, 1.0, -2.0])
    angles = np.radians([-90.0, -30.0, 0.0, 30.0, 90.0])

    many = cast_rays_many(track.boundary_segments, origins, headings, angles, 40.0)

    assert many.shape == (3, 5)
    for index in range(3):
        single = cast_rays(
            track.boundary_segments, origins[index], headings[index], angles, 40.0
        )
        assert many[index] == pytest.approx(single)


def test_step_arrays_matches_scalar_step():
    model = KinematicBicycle()
    x = np.array([0.0, 5.0, -3.0])
    y = np.array([0.0, 1.0, 2.0])
    heading = np.array([0.0, 1.0, -2.0])
    speed = np.array([0.0, 10.0, 19.9])
    steer = np.array([1.0, -0.5, 0.0])
    throttle = np.array([1.0, -1.0, 0.3])

    nx, ny, nheading, nspeed, nyaw = model.step_arrays(x, y, heading, speed, steer, throttle, 0.05)

    for i in range(3):
        after = model.step(VehicleState(x[i], y[i], heading[i], v_long=speed[i]), steer[i], throttle[i], 0.05)
        assert nx[i] == pytest.approx(after.x)
        assert ny[i] == pytest.approx(after.y)
        assert nheading[i] == pytest.approx(after.heading)
        assert nspeed[i] == pytest.approx(after.v_long)
        assert nyaw[i] == pytest.approx(after.yaw_rate)
```

- [ ] **Step 2:** `uv run --all-groups pytest tests/racing/test_vectorized.py -q` → падает (нет методов). Закоммитить: `git add tests/racing/test_vectorized.py` / `test: specify vectorized racing primitives`.

- [ ] **Step 3: Реализация.**
  - `Track.project_many(points: np.ndarray (N,2)) -> tuple[np.ndarray, np.ndarray]` — та же математика, что в `project`, с осью N (форма `(N, S, 2)`), результаты `(N,)`.
  - `Track.contains_many(points) -> np.ndarray[bool]` — `distance <= width/2`.
  - `Track.progress_delta_many(previous, current)` — как `progress_delta`, через `np.where`.
  - `cast_rays_many(segments, origins (N,2), headings (N,), angles (R,), max_range) -> (N, R)` в `sensors.py` — та же формула пересечения, что в `cast_rays`, форма `(N, R, S)`; `cast_rays` не менять.
  - `KinematicBicycle.step_arrays(x, y, heading, v_long, steer, throttle, dt) -> (x, y, heading, v_long, yaw_rate)` — те же формулы, что `step`, на numpy-массивах (`np.clip`, `np.where` для газа/тормоза). Добавить метод в протокол `VehicleDynamics`.
  - Скалярные методы не менять.

- [ ] **Step 4:** тесты зелёные, `uv run --all-groups ruff check src tests` чист, весь набор `uv run --all-groups pytest -q` зелёный. Commit: `feat: add vectorized racing primitives`.

---

### Task 2: Флот `RacingFleet`

**Files:** Create `tests/racing/test_fleet.py`, `src/rl_fun/racing/fleet.py`.

- [ ] **Step 1: Тесты** — `tests/racing/test_fleet.py`

```python
import math

import numpy as np
import pytest

from rl_fun.racing.env import RacingEnv
from rl_fun.racing.fleet import RacingFleet

FORWARD = np.array([0.0, 1.0])
IDLE = np.array([0.0, 0.0])


def repeat(action: np.ndarray, count: int) -> np.ndarray:
    return np.tile(action, (count, 1))


def test_reset_returns_observation_per_car():
    fleet = RacingFleet(4)

    observation = fleet.reset()

    assert observation.shape == (4, 8)
    assert observation.dtype == np.float32
    assert fleet.observation_size == 8
    assert fleet.alive.all()
    assert fleet.progress == pytest.approx(np.zeros(4))
    assert not fleet.done


def test_ray_angles_define_observation_size():
    fleet = RacingFleet(2, ray_angles_deg=(-90, 0, 90))

    assert fleet.reset().shape == (2, 6)


def test_single_car_matches_racing_env():
    fleet = RacingFleet(1)
    env = RacingEnv()
    observation_fleet = fleet.reset()
    observation_env, _ = env.reset(seed=0)
    assert observation_fleet[0] == pytest.approx(observation_env, abs=1e-5)

    for step in range(40):
        action = np.array([0.3 * math.sin(step / 5), 1.0], dtype=np.float32)
        observation_fleet, rewards = fleet.step(action[None, :])
        observation_env, reward, terminated, _, info = env.step(action)

        assert not terminated
        assert observation_fleet[0] == pytest.approx(observation_env, abs=1e-5)
        assert rewards[0] == pytest.approx(reward, abs=1e-6)
        assert fleet.progress[0] == pytest.approx(info["progress"], abs=1e-6)
    env.close()


def test_cars_with_different_actions_diverge():
    fleet = RacingFleet(2)
    fleet.reset()

    for _ in range(10):
        fleet.step(np.array([[0.0, 1.0], [0.5, 1.0]]))

    assert fleet.heading[0] != fleet.heading[1]


def test_crashed_car_dies_and_freezes_while_others_continue():
    fleet = RacingFleet(2)
    fleet.reset()

    for _ in range(600):
        fleet.step(np.array([FORWARD, IDLE]))
        if not fleet.alive[0]:
            break
    frozen_x, frozen_y = fleet.x[0], fleet.y[0]
    _, rewards = fleet.step(np.array([FORWARD, IDLE]))

    assert not fleet.alive[0]
    assert fleet.alive[1]
    assert (fleet.x[0], fleet.y[0]) == (frozen_x, frozen_y)
    assert rewards[0] == 0.0
    assert fleet.steps_alive[0] < fleet.steps_alive[1]
    assert not fleet.done


def test_done_after_max_steps():
    fleet = RacingFleet(2, max_steps=3)
    fleet.reset()

    for _ in range(3):
        fleet.step(repeat(IDLE, 2))

    assert fleet.done
    assert fleet.alive.all()


def test_done_when_every_car_has_crashed():
    fleet = RacingFleet(2)
    fleet.reset()

    for _ in range(600):
        fleet.step(repeat(FORWARD, 2))
        if fleet.done:
            break

    assert fleet.done
    assert not fleet.alive.any()


def test_finishing_the_lap_marks_car_finished():
    fleet = RacingFleet(1)
    fleet.reset()
    fleet._travelled[:] = fleet.track.length - 0.001

    fleet.step(repeat(FORWARD, 1))

    assert fleet.finished[0]
    assert not fleet.alive[0]
    assert fleet.progress[0] >= 1.0
    assert fleet.lap_steps[0] == 1.0


def test_lap_steps_is_nan_for_unfinished_cars():
    fleet = RacingFleet(2)
    fleet.reset()
    fleet.step(repeat(FORWARD, 2))

    assert np.isnan(fleet.lap_steps).all()


def test_actions_are_clipped_and_shape_checked():
    clipped, limit = RacingFleet(2), RacingFleet(2)
    clipped.reset()
    limit.reset()

    observation_a, _ = clipped.step(np.full((2, 2), 5.0))
    observation_b, _ = limit.step(np.full((2, 2), 1.0))

    assert np.array_equal(observation_a, observation_b)
    with pytest.raises(ValueError):
        clipped.step(np.zeros((3, 2)))


@pytest.mark.parametrize(
    "kwargs",
    [{"n_cars": 0}, {"n_cars": 2, "ray_angles_deg": ()}, {"n_cars": 2, "max_steps": 0},
     {"n_cars": 2, "track": "missing"}, {"n_cars": 2, "dt": 0.0}],
)
def test_invalid_arguments_raise(kwargs: dict):
    with pytest.raises(ValueError):
        RacingFleet(**kwargs)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset"):
        RacingFleet(2).step(repeat(IDLE, 2))
```

- [ ] **Step 2:** убедиться, что падает (`ModuleNotFoundError: rl_fun.racing.fleet`); закоммитить RED: `test: specify racing fleet`.

- [ ] **Step 3: Реализация** `src/rl_fun/racing/fleet.py` — `RacingFleet`:
  - Конструктор: `(n_cars, track="oval", dynamics="kinematic", ray_angles_deg=(-90, -30, 0, 30, 90), ray_range=40.0, dt=1/30, max_steps=1500, laps=1)`. Валидация как в тестах (`ValueError`). Атрибуты: `track`, `dynamics`, `n_cars`, `ray_range`, `dt`, `max_steps`, `laps`, `observation_size = len(angles) + 3`.
  - Состояние — массивы `(N,)`: `x`, `y`, `heading`, `v_long`, `yaw_rate`, `alive`, `finished`, `steps_alive` (int), `lap_steps` (float, `nan` пока не финишировал), приватный `_travelled` (метры, накопительно), `_arclength` (предыдущая проекция), `_distances` `(N, R)`; публично `distances`, `steps`, `progress = _travelled / (track.length)` в кругах, `done`.
  - `reset() -> (N, R+3) float32`: все на старте трассы, `alive=True`, остальное обнулить.
  - `step(actions) -> (observation, rewards)`: `RuntimeError("call reset() before step()")` до `reset`; форма `(N, 2)` иначе `ValueError`; обрезка до `[-1, 1]`. Обновляются **только живые**: `dynamics.step_arrays`, проекция через `project_many`, `progress_delta_many`, `contains_many`. Награда живой машинки = `delta / track.length`, у остальных `0.0`. Выезд за границу → `alive=False`. Если `_travelled >= laps * track.length` → `finished=True`, `alive=False`, `lap_steps = steps` (число шагов флота на этот момент). `steps_alive` у машинки увеличивается на шаг, пока она жива (включая шаг гибели/финиша). `done = not alive.any() or steps >= max_steps`.
  - Наблюдение строится как в `RacingEnv._observe` (нормировка лучей на `ray_range`, скорость на `max_speed`, угловая на `max_yaw_rate`, обрезка в `[0, 1]`/`[-1, 1]`, `float32`) для всех машинок; у мёртвых наблюдение остаётся прежним (состояние заморожено).
  - Поведение для `n_cars=1` должно совпадать с `RacingEnv` (тест).

- [ ] **Step 4:** `uv run --all-groups pytest tests/racing/test_fleet.py -q`, весь набор, ruff — зелёные. Commit: `feat: add vectorized racing fleet`.

---

### Task 3: Прогон поколения

**Files:** Create `tests/racing/test_generation.py`, `src/rl_fun/racing/generation.py`.

- [ ] **Step 1: Тесты** — `tests/racing/test_generation.py`

```python
import numpy as np
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import GenerationResult, run_generation


def forward_straight(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
    return np.array([0.0, 1.0])


def forward_idle(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
    return np.array([0.0, 0.0])


def test_returns_one_result_per_car():
    fleet = RacingFleet(3, max_steps=50)

    result = run_generation(fleet, np.zeros((3, 4)), forward_straight)

    assert isinstance(result, GenerationResult)
    for values in (result.progress, result.finished, result.steps_alive, result.lap_steps):
        assert values.shape == (3,)
    assert not result.finished.any()
    assert np.isnan(result.lap_steps).all()


def test_forward_receives_weight_row_and_observation():
    seen: list[tuple[np.ndarray, tuple[int, ...]]] = []

    def forward(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
        seen.append((weights.copy(), observation.shape))
        return np.array([0.0, 0.0])

    population = np.arange(6, dtype=float).reshape(2, 3)

    run_generation(RacingFleet(2, max_steps=2), population, forward)

    assert len(seen) == 4
    assert seen[0][1] == (8,)
    assert np.array_equal(seen[0][0], population[0])
    assert np.array_equal(seen[1][0], population[1])


def test_dead_cars_are_not_queried():
    calls = [0, 0]

    def forward(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
        calls[int(weights[0])] += 1
        return np.array([0.0, 1.0]) if weights[0] == 0 else np.array([0.0, 0.0])

    run_generation(RacingFleet(2, max_steps=800), np.array([[0.0], [1.0]]), forward)

    assert calls[1] == 800
    assert calls[0] < calls[1]


def test_run_resets_the_fleet_each_time():
    fleet = RacingFleet(2, max_steps=20)
    population = np.zeros((2, 1))

    first = run_generation(fleet, population, forward_straight)
    second = run_generation(fleet, population, forward_straight)

    assert first.progress == pytest.approx(second.progress)


def test_population_size_must_match_fleet():
    with pytest.raises(ValueError, match="population"):
        run_generation(RacingFleet(3), np.zeros((2, 4)), forward_straight)


def test_driving_beats_standing_still():
    fleet = RacingFleet(2, max_steps=60)
    population = np.array([[0.0], [1.0]])

    def forward(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0 - weights[0]])

    result = run_generation(fleet, population, forward)

    assert result.progress[0] > result.progress[1]
    assert result.best_index == 0
```

- [ ] **Step 2:** убедиться, что падает; RED-коммит `test: specify generation runner`.

- [ ] **Step 3: Реализация** `src/rl_fun/racing/generation.py`:
  - `@dataclass(frozen=True, slots=True) class GenerationResult`: `progress`, `finished`, `steps_alive`, `lap_steps` (`np.ndarray (N,)`); свойство `best_index -> int = int(np.argmax(progress))`.
  - `run_generation(fleet, population, forward, *, view=None, inspect=None, label="") -> GenerationResult`. Проверить `population.shape[0] == fleet.n_cars` → `ValueError("population size ...")`. `observation = fleet.reset()`. Пока не `fleet.done`: для каждой **живой** машинки `forward(population[i], observation[i])` → строка действия `(2,)` (остальные получают нули), `fleet.step(actions)`. Если `view` задан — после шага вызвать `view.draw(fleet, lines, network)`: `lines = [label, f"Живых: {n}/{N}", f"Шаг: {fleet.steps}"]`, `network` = `inspect(population[leader], observation[leader])` для живого лидера с максимальным прогрессом, если `inspect` задан и есть живые, иначе `None`. Вернуть результат из атрибутов флота (копии массивов).
  - Модуль не импортирует pygame (тип `view` — Protocol/`Any`).

- [ ] **Step 4:** тесты, весь набор, ruff зелёные. Commit: `feat: add generation runner`.

---

### Task 4: Эталонная эволюция

**Files:** Create `tests/racing/test_reference.py`, `src/rl_fun/racing/reference.py`.

- [ ] **Step 1: Тесты** — `tests/racing/test_reference.py`

```python
import numpy as np
import pytest

from rl_fun.racing import reference as ref


def test_layer_sizes_and_weight_count():
    sizes = ref.layer_sizes(8, (6, 5))

    assert sizes == [8, 6, 5, 2]
    assert ref.weight_count(sizes) == (8 + 1) * 6 + (6 + 1) * 5 + (5 + 1) * 2


def test_init_population_shape_and_determinism():
    sizes = ref.layer_sizes(8, (6, 5))

    a = ref.init_population(np.random.default_rng(1), 10, sizes, 1.0)
    b = ref.init_population(np.random.default_rng(1), 10, sizes, 1.0)

    assert a.shape == (10, ref.weight_count(sizes))
    assert np.array_equal(a, b)


def test_forward_returns_bounded_action_that_depends_on_weights():
    sizes = ref.layer_sizes(8, (6, 5))
    rng = np.random.default_rng(2)
    first, second = ref.init_population(rng, 2, sizes, 1.0)
    observation = np.linspace(0.0, 1.0, 8)

    action = ref.forward(first, observation, sizes)

    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)
    assert not np.allclose(action, ref.forward(second, observation, sizes))


def test_inspect_matches_forward():
    sizes = ref.layer_sizes(8, (6, 5))
    weights = ref.init_population(np.random.default_rng(3), 1, sizes, 1.0)[0]
    observation = np.linspace(0.0, 1.0, 8)

    matrices, activations = ref.inspect(weights, observation, sizes)

    assert [m.shape for m in matrices] == [(6, 8), (5, 6), (2, 5)]
    assert [a.shape[0] for a in activations] == [8, 6, 5, 2]
    assert activations[-1] == pytest.approx(ref.forward(weights, observation, sizes))


def test_next_generation_keeps_elite_best_first():
    params = ref.EvolutionParams(population=10, elite=2)
    population = np.random.default_rng(4).normal(size=(10, 20))
    fitness = np.arange(10, dtype=float)

    new = ref.next_generation(population, fitness, np.random.default_rng(5), params)

    assert new.shape == population.shape
    assert np.array_equal(new[0], population[9])
    assert np.array_equal(new[1], population[8])


def test_next_generation_mutates_children_and_is_deterministic():
    params = ref.EvolutionParams(population=10, elite=2, mutation_rate=1.0, mutation_scale=0.5)
    population = np.random.default_rng(4).normal(size=(10, 20))
    fitness = np.arange(10, dtype=float)

    a = ref.next_generation(population, fitness, np.random.default_rng(5), params)
    b = ref.next_generation(population, fitness, np.random.default_rng(5), params)

    assert np.array_equal(a, b)
    for child in a[2:]:
        assert not any(np.array_equal(child, parent) for parent in population)


def test_reference_improves_on_oval():
    params = ref.EvolutionParams(population=24, elite=4)

    run = ref.train("oval", seed=7, generations=15, params=params, fleet_kwargs={"max_steps": 600})

    bests = [entry["best"] for entry in run.history]
    assert len(run.history) == 15
    assert max(bests) > bests[0]
    assert run.best_weights.shape == (ref.weight_count(run.sizes),)
```

- [ ] **Step 2:** падает на импорте; RED-коммит `test: specify reference evolution`.

- [ ] **Step 3: Реализация** `src/rl_fun/racing/reference.py` (numpy, без новых зависимостей):
  - `@dataclass(frozen=True) EvolutionParams`: `population=60`, `hidden=(6, 5)`, `elite=6`, `mutation_rate=0.15`, `mutation_scale=0.3`, `init_scale=1.0`.
  - `layer_sizes(n_inputs, hidden, n_outputs=2) -> list[int]`; `weight_count(sizes) -> int` (на каждый слой `(in + 1) * out`, смещения включены).
  - `init_population(rng, n, sizes, scale) -> (n, W)`: нормальные веса `scale * normal / sqrt(in)` или проще `rng.normal(0, scale, ...)` (выбрать то, что лучше обучается; смещения можно инициализировать нулями).
  - `forward(weights, observation, sizes) -> (2,)`: полносвязная сеть, скрытые слои и выход `tanh`. Раскладка весов слой за слоем: матрица `(out, in)` затем вектор смещений `(out,)`.
  - `inspect(weights, observation, sizes) -> (matrices, activations)`: матрицы `(out, in)` по слоям (без смещений) и массивы активаций `[вход, слой1, ..., выход]`.
  - `next_generation(population, fitness, rng, params)`: сортировка по убыванию фитнеса; первые `elite` строк копируются без изменений (лучший первым); остальные — потомки случайных родителей из элиты (`rng.choice`) с гауссовой мутацией: каждый вес мутирует с вероятностью `mutation_rate`, добавка `normal(0, mutation_scale)`. Размер выхода равен размеру входа. Результат детерминирован при одном состоянии `rng`.
  - `@dataclass ReferenceRun`: `history: list[dict]` (`generation`, `best`, `mean`, `finished` — число машинок с `finished`), `best_weights` (лучшие за всё время), `sizes`, `params`, `seed`, `track`.
  - `train(track="oval", seed=7, generations=50, params=EvolutionParams(), fleet_kwargs=None, view=None) -> ReferenceRun`: `rng = np.random.default_rng(seed)`, `RacingFleet(params.population, track=track, **(fleet_kwargs or {}))`, на каждом поколении `run_generation(...)` (с `view`, если задан), фитнес = `progress + бонус за скорость круга` (для финишировавших `1 - lap_steps / max_steps`, для остальных 0; обработать `nan`), запись в историю, `next_generation`.
  - **Подбор параметров:** если `test_reference_improves_on_oval` не проходит при `seed=7`, подобрать значения по умолчанию `EvolutionParams`/инициализации (не менять тест). Проверить устойчивость (seed 1, 7, 42): улучшение должно быть не случайным.

- [ ] **Step 4:** тесты, ruff, весь набор зелёные. Commit: `feat: add reference evolution`.

---

### Task 5: Бенчмарк эталона и сравнение

**Files:** Create `tests/racing/test_benchmark.py`, `src/rl_fun/racing/benchmark.py`, `src/rl_fun/racing/benchmarks/oval.json`, `scripts/make_reference.py`.

- [ ] **Step 1: Тесты** — `tests/racing/test_benchmark.py`

```python
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from rl_fun.racing import reference as ref
from rl_fun.racing.benchmark import compare, load_benchmark, plot_comparison
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import run_generation

HISTORY = [
    {"generation": 0, "best": 0.10, "mean": 0.05, "finished": 0},
    {"generation": 1, "best": 0.30, "mean": 0.12, "finished": 0},
]


def test_oval_benchmark_has_expected_fields():
    benchmark = load_benchmark("oval")

    for key in ("track", "seed", "params", "fleet", "history", "best_weights", "sizes"):
        assert key in benchmark
    assert benchmark["track"] == "oval"


def test_reference_completes_a_lap_in_benchmark():
    history = load_benchmark("oval")["history"]

    assert max(entry["best"] for entry in history) >= 1.0
    assert any(entry["finished"] > 0 for entry in history)


def test_benchmark_is_reproducible():
    benchmark = load_benchmark("oval")
    params = ref.EvolutionParams(**{**benchmark["params"], "hidden": tuple(benchmark["params"]["hidden"])})

    run = ref.train(
        benchmark["track"], seed=benchmark["seed"], generations=3,
        params=params, fleet_kwargs=benchmark["fleet"],
    )

    for new, stored in zip(run.history, benchmark["history"][:3]):
        assert new["best"] == pytest.approx(stored["best"])
        assert new["mean"] == pytest.approx(stored["mean"])


def test_stored_best_weights_still_drive_a_lap():
    benchmark = load_benchmark("oval")
    sizes = benchmark["sizes"]
    weights = np.array(benchmark["best_weights"])
    fleet = RacingFleet(1, track=benchmark["track"], **benchmark["fleet"])

    result = run_generation(
        fleet, weights[None, :], lambda w, o: ref.forward(w, o, sizes)
    )

    assert result.finished[0]


def test_compare_mentions_both_sides():
    text = compare(HISTORY, load_benchmark("oval"))

    assert "эталон" in text.lower()
    assert "0.30" in text


def test_compare_reports_lap_completion():
    history = [*HISTORY, {"generation": 2, "best": 1.0, "mean": 0.4, "finished": 3}]

    assert "круг" in compare(history, load_benchmark("oval")).lower()


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError, match="benchmark"):
        load_benchmark("nope")


def test_plot_comparison_returns_figure():
    figure = plot_comparison(HISTORY, load_benchmark("oval"))

    assert figure.axes
```

- [ ] **Step 2:** падает на импорте; RED-коммит `test: specify reference benchmark`.

- [ ] **Step 3: Реализация.**
  - `scripts/make_reference.py [--track oval] [--seed 7] [--generations 80]`: запускает `reference.train`, пишет `src/rl_fun/racing/benchmarks/<track>.json` с ключами `track`, `seed`, `params` (`dataclasses.asdict`, `hidden` как список), `fleet` (kwargs флота, например `{"max_steps": 1500}`), `history`, `best_weights` (список float), `sizes`. Скрипт печатает по одной строке на поколение.
  - Запустить скрипт и закоммитить `oval.json`. **Критерий приёмки:** эталон проходит круг (`best >= 1.0`, `finished > 0`) в пределах заданных поколений. Если нет — подобрать `EvolutionParams`/число поколений/seed (вернуться к Task 4) и сообщить, что изменилось. Размер файла разумный (веса ~100 чисел, история ≤ 100 записей).
  - `benchmark.py`: `load_benchmark(track) -> dict` (`ValueError("unknown benchmark ...")` для отсутствующего файла); `compare(history, benchmark) -> str` — русский текст: лучший прогресс пользователя против эталона (формат `{:.2f}`), на каком поколении эталон прошёл круг и прошёл ли его пользователь, понятный вывод «ты на X% от эталона»; `plot_comparison(history, benchmark)` — `matplotlib.figure.Figure` с кривыми лучшего и среднего прогресса пользователя и эталона по поколениям, подписи на русском; matplotlib импортируется внутри функции (он в группе `notebook`).

- [ ] **Step 4:** тесты, ruff, весь набор зелёные. Commit: `feat: add reference benchmark and comparison helpers` (включая `oval.json` и скрипт).

---

### Task 6: Отрисовка флота и панель сети

**Files:** Create `tests/racing/test_fleet_view.py`, `src/rl_fun/racing/fleet_view.py`; Modify `render.py`.

- [ ] **Step 1: Тесты** — `tests/racing/test_fleet_view.py`

```python
import numpy as np
import pygame
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.fleet_view import FleetView, NetworkPanel, ViewClosed
from rl_fun.racing.generation import run_generation
from rl_fun.racing.render import GAME_HEIGHT, GAME_WIDTH


def network_state() -> tuple[list[np.ndarray], list[np.ndarray]]:
    matrices = [np.ones((3, 2)), -np.ones((2, 3))]
    activations = [np.array([0.5, -0.5]), np.zeros(3), np.array([0.1, 0.9])]
    return matrices, activations


def test_network_panel_draws_something_once_updated():
    panel = NetworkPanel(["a", "b"], ["x", "y"], width=300)
    panel.update(*network_state())
    surface = pygame.Surface((300, GAME_HEIGHT))
    surface.fill((0, 0, 0))

    panel.draw(surface, surface.get_rect())

    assert pygame.surfarray.array3d(surface).any()


def test_network_panel_without_state_does_not_crash():
    panel = NetworkPanel(["a"], ["x"], width=200)
    surface = pygame.Surface((200, GAME_HEIGHT))

    panel.draw(surface, surface.get_rect())


def test_fleet_view_frame_includes_network_panel():
    fleet = RacingFleet(5)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=True)
    try:
        frame = view.draw(fleet, ["Поколение 1"], None)
    finally:
        view.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH + view.panel.width, 3)


def test_fleet_view_without_network_has_game_size():
    fleet = RacingFleet(5)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=False)
    try:
        frame = view.draw(fleet, [], None)
    finally:
        view.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH, 3)


def test_dead_cars_are_drawn_differently_from_living_ones():
    fleet = RacingFleet(3)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=False)
    try:
        alive_frame = view.draw(fleet, [], None)
        fleet.alive[1:] = False
        dead_frame = view.draw(fleet, [], None)
    finally:
        view.close()

    assert not np.array_equal(alive_frame, dead_frame)


def test_run_generation_draws_every_step_and_passes_network_state():
    fleet = RacingFleet(3, max_steps=5)
    view = FleetView(fleet, mode="rgb_array", show_network=True)
    networks: list[object] = []
    original = view.draw

    def spy(fleet_arg, lines, network):
        networks.append(network)
        return original(fleet_arg, lines, network)

    view.draw = spy
    try:
        run_generation(
            fleet, np.zeros((3, 2)), lambda w, o: np.array([0.0, 1.0]),
            view=view, inspect=lambda w, o: network_state(), label="Поколение 0",
        )
    finally:
        view.close()

    assert len(networks) == 5
    assert all(network is not None for network in networks)


def test_closed_window_error_propagates_from_run_generation():
    fleet = RacingFleet(2, max_steps=5)

    class ClosingView:
        def draw(self, fleet, lines, network):
            raise ViewClosed

    with pytest.raises(ViewClosed):
        run_generation(
            fleet, np.zeros((2, 1)), lambda w, o: np.zeros(2), view=ClosingView()
        )
```

- [ ] **Step 2:** падает на импорте; RED-коммит `test: specify fleet view and network panel`.

- [ ] **Step 3: Реализация.**
  - `Renderer.draw_fleet(x, y, heading, alive, leader, ray_points, lines) -> np.ndarray | None` в `render.py`: общий с `draw` код (фон, HUD, панель, режимы `human`/`rgb_array`) вынести в приватные методы; живые машинки яркие, мёртвые приглушённые, лидер выделен контуром; лучи рисуются только у лидера. `draw` для одной машинки сохранить без изменения поведения (существующие тесты `test_render.py` остаются зелёными).
  - `fleet_view.py`: `class ViewClosed(Exception)`; `NetworkPanel(input_labels, output_labels, width=360)` реализует `SidePanel` (`width`, `draw(surface, rect)`) и `update(matrices, activations)`; рисует слои узлов колонками, связи зелёным при положительном весе и красным при отрицательном (толщина и прозрачность по модулю, слабые связи пропускать), узлы цветом по знаку активации; подписи входов и выходов со значениями у выходов; без состояния рисует только фон и заголовок. `FleetView(fleet, mode="human", show_network=True, input_labels=None, output_labels=("Руль", "Газ"))`: подписи входов по умолчанию строятся из углов лучей (`"-90°"`, ..., `"Скор."`, `"Бок."`, `"Угл."`), атрибут `panel` (`None`, если сеть выключена); `draw(fleet, lines, network) -> frame | None` обновляет панель из `network` (`(matrices, activations)`), вычисляет лидера (живая машинка с максимальным `progress`, иначе максимальный `progress` среди всех), точки лучей лидера из `fleet.distances` и вызывает `Renderer.draw_fleet`; в режиме `human` при событии `pygame.QUIT` поднимает `ViewClosed`; `close()`.

- [ ] **Step 4:** `uv run --all-groups pytest tests/racing -q`, весь набор, ruff зелёные. Commit: `feat: add fleet rendering and network panel`.

---

### Task 7: Ноутбук-заготовка

**Files:** Create `tests/integration/test_racing_notebook.py`, `notebooks/02_racing/01_evolution.ipynb`.

- [ ] **Step 1: Тесты** — `tests/integration/test_racing_notebook.py`

```python
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path("notebooks/02_racing/01_evolution.ipynb")
STUBS = ("init_population", "forward", "next_generation", "inspect")
REFERENCE_CELLS = {
    "init_population": (
        "from rl_fun.racing import reference as ref\n"
        "def init_population(rng, n, n_weights):\n"
        "    return ref.init_population(rng, n, SIZES, 1.0)\n"
    ),
    "forward": (
        "def forward(weights, observation):\n"
        "    return ref.forward(weights, observation, SIZES)\n"
    ),
    "next_generation": (
        "def next_generation(population, fitness, rng):\n"
        "    params = ref.EvolutionParams(population=len(population), elite=2)\n"
        "    return ref.next_generation(population, fitness, rng, params)\n"
    ),
    "inspect": (
        "def inspect(weights, observation):\n"
        "    return ref.inspect(weights, observation, SIZES)\n"
    ),
}


def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def source(cell: dict) -> str:
    return "".join(cell["source"])


def stub_cell(notebook: dict, name: str) -> dict:
    matches = [
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and re.search(rf"def {name}\(", source(cell))
    ]
    assert len(matches) == 1, f"expected exactly one cell defining {name}"
    return matches[0]


def test_every_stub_cell_is_a_stub():
    notebook = load()

    for name in STUBS:
        assert "NotImplementedError" in source(stub_cell(notebook, name))


def test_stub_cells_do_not_contain_a_solution():
    notebook = load()

    for name in STUBS:
        text = source(stub_cell(notebook, name))
        for giveaway in ("tanh", "argsort", "rng.normal", "np.dot", "@ "):
            assert giveaway not in text


def test_every_code_cell_follows_a_russian_explanation():
    cells = load()["cells"]
    markdown = [source(c) for c in cells if c["cell_type"] == "markdown"]

    assert len(markdown) >= 8
    assert any(re.search("[А-Яа-я]", text) for text in markdown)
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert index > 0 and cells[index - 1]["cell_type"] == "markdown", index


def test_reference_is_used_only_in_comparison_sections():
    heading = ""
    for cell in load()["cells"]:
        text = source(cell)
        if cell["cell_type"] == "markdown":
            headings = [line for line in text.splitlines() if line.startswith("#")]
            heading = headings[0].lower() if headings else heading
        elif "reference" in text or "benchmark" in text:
            assert "эталон" in heading, heading


def test_notebook_runs_end_to_end_with_reference_solution(tmp_path: Path):
    notebook = load()
    for name in STUBS:
        stub_cell(notebook, name)["source"] = REFERENCE_CELLS[name].splitlines(keepends=True)
    for cell in notebook["cells"]:
        text = source(cell)
        if "GENERATIONS =" in text:
            text = re.sub(r"GENERATIONS = \d+", "GENERATIONS = 2", text)
            text = re.sub(r"POPULATION = \d+", "POPULATION = 8", text)
            text = re.sub(r"MAX_STEPS = \d+", "MAX_STEPS = 100", text)
            text = re.sub(r"SHOW_WINDOW = \w+", "SHOW_WINDOW = False", text)
            cell["source"] = text.splitlines(keepends=True)
    prepared = tmp_path / "prepared.ipynb"
    prepared.write_text(json.dumps(notebook), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--ExecutePreprocessor.timeout=300", "--output", str(tmp_path / "executed.ipynb"),
         str(prepared)],
        capture_output=True, text=True, timeout=360, check=False,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
    )

    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2:** падает (нет ноутбука); RED-коммит `test: specify racing evolution notebook`.

- [ ] **Step 3: Создать ноутбук** `notebooks/02_racing/01_evolution.ipynb` (нужны ID ячеек, как в существующем ноутбуке; тест подменяет заготовки ячеек по `def имя(`). Стиль: короткие русские пояснения, формулы в `$...$`. Обязательная структура, у **каждой** кодовой ячейки перед ней markdown-ячейка:
  1. `# Эволюция машинок` — идея цикла «оценить, отобрать, мутировать» и схема поколения; что пишет пользователь, а что даёт библиотека.
  2. `## Параметры` + код: `TRACK = "oval"`, `RAY_ANGLES = (-90, -30, 0, 30, 90)`, `POPULATION = 50`, `GENERATIONS = 60`, `MAX_STEPS = 1500`, `HIDDEN = (6, 5)`, `SEED = 7`, `SHOW_WINDOW = True`, `SHOW_NETWORK = False`. Все в одной ячейке, формат `NAME = value` (тест правит их регулярными выражениями). Пояснение, что каждый параметр меняет.
  3. `## Флот и наблюдения` — что такое наблюдение (лучи, скорости), действие, фитнес. Код создаёт `fleet = RacingFleet(...)`, печатает форму наблюдения; затем ячейка, считающая `SIZES = [fleet.observation_size, *HIDDEN, 2]` и `N_WEIGHTS = sum((a + 1) * b for a, b in zip(SIZES[:-1], SIZES[1:]))` с пояснением про веса и смещения.
  4. `## Заготовка 1: начальная популяция` — код `def init_population(rng, n, n_weights): ...` с `raise NotImplementedError("впиши здесь ...")`, в тексте подсказки о форме `(n, n_weights)` и о масштабе.
  5. `## Заготовка 2: сеть` — `def forward(weights, observation): ...` (вход — вектор длины `SIZES[0]`, выход — `[руль, газ]` в `[-1, 1]`), подсказки без реализации.
  6. `## Заготовка 3: отбор и мутация` — `def next_generation(population, fitness, rng): ...`.
  7. `## Заготовка 4 (необязательно): взгляд внутрь сети` — `def inspect(weights, observation): ...` для панели, объяснение формата `(matrices, activations)`; пояснение, что нужно поставить `SHOW_NETWORK = True`.
  8. `## Обучение` — цикл поколений из спеки/плана (создание `RacingFleet`, `FleetView` только если `SHOW_WINDOW`, `run_generation`, `fitness = result.progress  # можно изменить`, `history.append({...generation, best, mean, finished})`, печать строки, `next_generation`, перехват `ViewClosed`, `finally: view.close()`; `inspect` передаётся только если `SHOW_NETWORK`).
  9. `## Кривая обучения` — график лучшего и среднего прогресса (matplotlib).
  10. `## Сравнение с эталоном` — `from rl_fun.racing.benchmark import compare, load_benchmark, plot_comparison`; печать `compare(history, benchmark)` и график `plot_comparison`.
  11. `## Посмотреть лучшую машинку и эталон` — под `if SHOW_WINDOW:`: показать прогон лучшей особи пользователя и эталонных весов через `FleetView`.
  12. `## Что попробовать дальше` (markdown, без кода) — идеи: другое число лучей, архитектура сети, форма фитнеса, трасса `wavy`, скорость мутации.
  Ячейки сравнения/просмотра — единственные, где встречаются слова `reference`/`benchmark`, под заголовками с «эталон».

- [ ] **Step 4:** `uv run --all-groups pytest tests/integration/test_racing_notebook.py -v` зелёный; весь набор и ruff.
- [ ] **Step 5:** Commit: `docs: add racing evolution notebook`.

---

### Task 8: README и финальная проверка

- [ ] **Step 1:** В `README.md` добавить русский раздел «Эволюция машинок» (после «Гоночная среда»): что такое флот, как открыть ноутбук `notebooks/02_racing/01_evolution.ipynb`, смысл `SHOW_WINDOW` и `SHOW_NETWORK`, как сравниваться с эталоном, как перегенерировать бенчмарк (`scripts/make_reference.py`), что всё считается на CPU. Дополнить список «Структура проекта».
- [ ] **Step 2:** `uv run --all-groups pytest -q` и `uv run --all-groups ruff check .` — всё зелёное.
- [ ] **Step 3:** Commit: `docs: document racing evolution workflow`.
- [ ] **Step 4 (пользователь):** открыть ноутбук с `SHOW_WINDOW = True` и убедиться, что окно показывает пачку машинок и (с реализованной `inspect`) панель сети.

---

## Self-Review

- **Покрытие спеки:** §4.1 — Task 1; §4.2 — Task 2; §4.3 — Task 3 (+ вызов `view` в Task 6); §4.4 — Task 6; §4.5 — Tasks 4–5; §4.6 — Task 7; README — Task 8.
- **Согласованность:** `RacingFleet` атрибуты (`alive`, `finished`, `progress`, `steps_alive`, `lap_steps`, `x`, `y`, `heading`, `distances`, `done`, `_travelled`) и сигнатуры `run_generation(fleet, population, forward, *, view, inspect, label)`, `FleetView.draw(fleet, lines, network)`, `reference.*` одинаковы во всех задачах. Бенчмарк хранит `params`/`fleet`/`sizes`/`best_weights` в тех же именах, что читают тесты.
- **Риск:** обучаемость эталона (Task 4–5) проверяется на практике; при провале подбираем параметры, не тесты.
