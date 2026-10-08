# Model Spec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Описать модель машинки данными (`ModelSpec`), сделать входы и выходы флота настраиваемыми и добавить составной фитнес — без поломки текущего поведения.

**Architecture:** `ModelSpec` (чистые данные + проверка) → `controls.map_controls` (сырые выходы сети в руль/газ/буст) → `RacingFleet(model=...)` собирает наблюдение в порядке `model.inputs` и копит статистики → `FitnessSpec.score(GenerationResult)` → `reference.train(model=..., fitness=...)`. Спека: `docs/superpowers/specs/2026-10-08-model-spec-design.md` (раздел 4 — точные контракты).

**Tech Stack:** Python 3.12, numpy, pytest, ruff, uv. Новых зависимостей нет.

**Формат:** тесты даны полностью (они определяют поведение), реализация описана сигнатурами и правилами. Если тест кажется неверным — остановиться и сообщить, не менять молча.

**Conventions:** TDD — красные тесты отдельным коммитом; Conventional Commits; AAA; аннотации типов; ruff (line-length 100); ветка `feat/racing-env`; в каждый коммит добавляются только свои файлы (`git add <пути>`). Трейлер: пустая строка и `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>` (или модель исполнителя). Команды из `D:\projects\RL_fun` (PowerShell). Не трогать `docs/reference/`. Все прежние тесты должны оставаться зелёными (прежнее поведение при значениях по умолчанию не меняется).

## File Structure

| Файл | Ответственность |
|---|---|
| `src/rl_fun/racing/model_spec.py` | `ModelSpec`, каталоги входов и выходов |
| `src/rl_fun/racing/controls.py` | `map_controls` |
| `src/rl_fun/racing/dynamics.py` (изм.) | `boost` в `step_arrays` |
| `src/rl_fun/racing/fleet.py` (изм.) | `model`, `action_size`, новые входы, статистики |
| `src/rl_fun/racing/fitness.py` | `FitnessSpec`, `PRESETS` |
| `src/rl_fun/racing/generation.py` (изм.) | новые поля `GenerationResult`, размер действий |
| `src/rl_fun/racing/reference.py` (изм.) | `activation`, `train(model=, fitness=)` |
| `src/rl_fun/racing/fleet_view.py` (изм.) | подписи из `fleet.model` |
| `tests/racing/test_model_spec.py`, `test_controls.py`, `test_fleet_model.py`, `test_fitness.py`, `test_reference_model.py` | тесты |

---

### Task 1: `ModelSpec`

- [ ] **Step 1: Тесты** — `tests/racing/test_model_spec.py`

```python
import json

import pytest

from rl_fun.racing.model_spec import ModelSpec, available_inputs, available_outputs

DEFAULT_INPUTS = (
    "ray:-90", "ray:-30", "ray:0", "ray:30", "ray:90", "speed", "lateral_speed", "yaw_rate",
)


def test_default_spec_matches_current_model():
    spec = ModelSpec()

    assert spec.inputs == DEFAULT_INPUTS
    assert spec.hidden == (6, 5)
    assert spec.outputs == ("steer", "throttle")
    assert spec.activation == "tanh"
    assert spec.layer_sizes == [8, 6, 5, 2]
    assert spec.weight_count == (8 + 1) * 6 + (6 + 1) * 5 + (5 + 1) * 2


def test_ray_angles_follow_input_order():
    spec = ModelSpec(inputs=("speed", "ray:30", "ray:-30", "yaw_rate"))

    assert spec.ray_angles_deg == (30.0, -30.0)


def test_labels_for_default_spec():
    spec = ModelSpec()

    assert spec.input_labels == [
        "↑ -90°", "↑ -30°", "↑ 0°", "↑ 30°", "↑ 90°", "Скор.", "Бок.", "Угл.",
    ]
    assert spec.output_labels == ["Руль", "Газ"]


def test_labels_for_extended_spec():
    spec = ModelSpec(
        inputs=("ray:0", "acceleration", "steering_angle"),
        outputs=("steer", "accelerate", "brake", "boost"),
    )

    assert spec.input_labels == ["↑ 0°", "Уск.", "Колёса"]
    assert spec.output_labels == ["Руль", "Газ", "Тормоз", "Буст"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inputs": ()},
        {"inputs": ("speed", "speed")},
        {"inputs": ("warp",)},
        {"inputs": ("ray:abc",)},
        {"inputs": ("ray:200",)},
        {"hidden": (0,)},
        {"hidden": (-3,)},
        {"outputs": ("steer",)},
        {"outputs": ("throttle",)},
        {"outputs": ("steer", "throttle", "accelerate")},
        {"outputs": ("steer", "throttle", "brake")},
        {"outputs": ("steer", "warp")},
        {"outputs": ("steer", "throttle", "throttle")},
        {"activation": "swish"},
    ],
)
def test_invalid_specs_raise(kwargs: dict):
    with pytest.raises(ValueError):
        ModelSpec(**kwargs)


def test_empty_hidden_layers_are_allowed():
    spec = ModelSpec(hidden=())

    assert spec.layer_sizes == [8, 2]


def test_dict_round_trip_through_json():
    spec = ModelSpec(
        inputs=("ray:-45", "ray:45", "speed", "acceleration"),
        hidden=(4,),
        outputs=("steer", "accelerate", "brake", "boost"),
        activation="relu",
    )

    restored = ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    assert restored == spec


def test_from_dict_uses_defaults_and_rejects_unknown_keys():
    assert ModelSpec.from_dict({}) == ModelSpec()
    with pytest.raises(ValueError, match="bogus"):
        ModelSpec.from_dict({"bogus": 1})


def test_catalogs_list_choices():
    assert {"speed", "lateral_speed", "yaw_rate", "acceleration", "steering_angle"} <= set(
        available_inputs()
    )
    assert available_outputs() == ("steer", "throttle", "accelerate", "brake", "boost")
```

- [ ] **Step 2:** запустить, убедиться, что падает на импорте; RED-коммит `test: specify model spec`.

- [ ] **Step 3: Реализация** `src/rl_fun/racing/model_spec.py` по разделу 4.1 спеки: замороженный `@dataclass(frozen=True)` с `__post_init__`-проверкой (поля-кортежи; принимать списки из `from_dict` и приводить к кортежам через `object.__setattr__`), свойства `layer_sizes`, `weight_count` (`(in + 1) * out` по слоям), `ray_angles_deg` (кортеж `float` в порядке входов), `input_labels` (луч: `"↑ {угол:g}°"` — целые углы без `.0`), `output_labels`; `to_dict` (списки), `from_dict` (неизвестные ключи → `ValueError` с их именами), `available_inputs()` (кортеж без лучей: `speed`, `lateral_speed`, `yaw_rate`, `acceleration`, `steering_angle`), `available_outputs()`. Луч допустим как `ray:<число>` в диапазоне −180..180.

- [ ] **Step 4:** тесты, `uv run --all-groups pytest -q`, `uv run --all-groups ruff check src tests` зелёные. Commit `feat: add model spec`.

---

### Task 2: Управление и буст

- [ ] **Step 1: Тесты** — `tests/racing/test_controls.py`

```python
import numpy as np
import pytest

from rl_fun.racing.controls import map_controls
from rl_fun.racing.dynamics import KinematicBicycle


def test_default_outputs_pass_steer_and_throttle_through():
    raw = np.array([[0.5, -0.25], [-1.0, 1.0]])

    steer, throttle, boost = map_controls(("steer", "throttle"), raw)

    assert steer == pytest.approx([0.5, -1.0])
    assert throttle == pytest.approx([-0.25, 1.0])
    assert boost == pytest.approx([0.0, 0.0])


def test_accelerate_and_brake_are_rescaled_and_combined():
    raw = np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])

    _, throttle, _ = map_controls(("steer", "accelerate", "brake"), raw)

    assert throttle == pytest.approx([1.0, -1.0, 0.0])


def test_boost_is_rescaled_to_unit_interval():
    raw = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    _, _, boost = map_controls(("steer", "throttle", "boost"), raw)

    assert boost == pytest.approx([1.0, 0.0])


def test_values_are_clipped():
    steer, throttle, _ = map_controls(("steer", "throttle"), np.array([[5.0, -5.0]]))

    assert steer == pytest.approx([1.0])
    assert throttle == pytest.approx([-1.0])


def test_wrong_width_raises():
    with pytest.raises(ValueError):
        map_controls(("steer", "throttle"), np.zeros((2, 3)))


def step(boost: float, speed: float = 0.0) -> tuple[float, float]:
    model = KinematicBicycle()
    x = np.zeros(1)
    result = model.step_arrays(
        x, x, x, np.array([speed]), x, np.ones(1), 0.1, boost=np.array([boost])
    )
    return float(result[3][0]), model.max_speed


def test_boost_accelerates_faster():
    boosted, _ = step(1.0)
    plain, _ = step(0.0)

    assert boosted > plain


def test_boost_raises_top_speed():
    top = KinematicBicycle().max_speed

    boosted, _ = step(1.0, speed=top)

    assert boosted > top


def test_zero_boost_keeps_old_behaviour():
    model = KinematicBicycle()
    arrays = [np.array([1.0]), np.array([2.0]), np.array([0.3]), np.array([10.0])]
    steer, throttle = np.array([0.4]), np.array([0.7])

    with_default = model.step_arrays(*arrays, steer, throttle, 0.05)
    explicit = model.step_arrays(*arrays, steer, throttle, 0.05, boost=np.zeros(1))

    for a, b in zip(with_default, explicit, strict=True):
        assert a == pytest.approx(b)
```

- [ ] **Step 2:** падает (нет `controls`, нет параметра `boost`); RED-коммит `test: specify controls and boost`.

- [ ] **Step 3: Реализация.**
  - `src/rl_fun/racing/controls.py`: `map_controls(outputs, raw) -> (steer, throttle, boost)` по разделу 4.2 (проверить ширину `raw.shape[1] == len(outputs)`; сырые значения `clip(-1, 1)`; отсутствующие каналы: `boost` нули; `throttle` строится из `throttle` + `accelerate` − `brake`).
  - `KinematicBicycle.step_arrays(..., dt, boost=0.0)`: параметр `boost` (число или массив `(N,)`, значения в `[0, 1]`); при `throttle > 0` разгон умножается на `1 + 0.8 * boost`; потолок скорости `max_speed * (1 + 0.3 * boost)`. Для `boost = 0` результаты прежние. Протокол `VehicleDynamics` дополнить параметром.

- [ ] **Step 4:** тесты и весь набор зелёные, ruff чистый. Commit `feat: add control mapping and boost`.

---

### Task 3: Флот с моделью

- [ ] **Step 1: Тесты** — `tests/racing/test_fleet_model.py`

```python
import numpy as np
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.model_spec import ModelSpec


def run(fleet: RacingFleet, action: list[float], steps: int) -> np.ndarray:
    observation = fleet.reset()
    for _ in range(steps):
        observation, _ = fleet.step(np.tile(action, (fleet.n_cars, 1)))
    return observation


def test_default_fleet_uses_default_model():
    fleet = RacingFleet(2)

    assert fleet.model == ModelSpec()
    assert fleet.action_size == 2


def test_model_and_ray_angles_are_mutually_exclusive():
    with pytest.raises(ValueError):
        RacingFleet(2, model=ModelSpec(), ray_angles_deg=(0,))


def test_observation_follows_input_order():
    fleet = RacingFleet(2, model=ModelSpec(inputs=("speed", "ray:0", "yaw_rate")))
    default = RacingFleet(2)

    observation = fleet.reset()
    default_observation = default.reset()

    assert observation.shape == (2, 3)
    assert observation[:, 0] == pytest.approx([0.0, 0.0])
    assert observation[:, 1] == pytest.approx(default_observation[:, 2])


def test_selected_columns_match_the_full_default_observation():
    spec = ModelSpec(inputs=("ray:30", "speed", "yaw_rate"))
    selected = run(RacingFleet(2, model=spec), [0.2, 1.0], 8)
    full = run(RacingFleet(2), [0.2, 1.0], 8)

    assert selected[:, 0] == pytest.approx(full[:, 3])
    assert selected[:, 1] == pytest.approx(full[:, 5])
    assert selected[:, 2] == pytest.approx(full[:, 7])


def test_acceleration_input_is_positive_when_speeding_up_and_negative_when_braking():
    spec = ModelSpec(inputs=("acceleration",))
    fleet = RacingFleet(1, model=spec)
    fleet.reset()
    for _ in range(10):
        accelerating, _ = fleet.step(np.array([[0.0, 1.0]]))
    for _ in range(2):
        braking, _ = fleet.step(np.array([[0.0, -1.0]]))

    assert accelerating[0, 0] > 0
    assert braking[0, 0] < 0


def test_steering_angle_input_reports_the_last_steer_command():
    spec = ModelSpec(inputs=("steering_angle",))
    fleet = RacingFleet(1, model=spec)
    assert fleet.reset()[0, 0] == pytest.approx(0.0)

    observation, _ = fleet.step(np.array([[0.5, 0.2]]))

    assert observation[0, 0] == pytest.approx(0.5)


def test_action_size_follows_outputs_and_width_is_checked():
    spec = ModelSpec(outputs=("steer", "accelerate", "brake", "boost"))
    fleet = RacingFleet(2, model=spec)
    fleet.reset()

    assert fleet.action_size == 4
    fleet.step(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        fleet.step(np.zeros((2, 2)))


def test_brake_output_slows_the_car():
    spec = ModelSpec(outputs=("steer", "accelerate", "brake"))
    fleet = RacingFleet(1, model=spec)
    fleet.reset()
    for _ in range(15):
        fleet.step(np.array([[0.0, 1.0, -1.0]]))
    speed_before = fleet.v_long[0]

    for _ in range(5):
        fleet.step(np.array([[0.0, -1.0, 1.0]]))

    assert fleet.v_long[0] < speed_before


def test_boost_output_makes_the_car_faster():
    spec = ModelSpec(outputs=("steer", "throttle", "boost"))
    boosted, plain = RacingFleet(1, model=spec), RacingFleet(1, model=spec)
    boosted.reset()
    plain.reset()

    for _ in range(30):
        boosted.step(np.array([[0.0, 1.0, 1.0]]))
        plain.step(np.array([[0.0, 1.0, -1.0]]))

    assert boosted.v_long[0] > plain.v_long[0]


def test_stats_start_at_zero_and_track_driving_quality():
    fleet = RacingFleet(2)
    fleet.reset()
    assert fleet.mean_speed.tolist() == [0.0, 0.0]

    for step in range(30):
        steer = 0.0 if step % 2 == 0 else 0.6
        fleet.step(np.array([[0.0, 1.0], [steer, 1.0]]))

    assert fleet.mean_speed[0] > 0
    assert fleet.centering[0] > 0.9
    assert fleet.smoothness[0] == pytest.approx(1.0)
    assert fleet.smoothness[1] < fleet.smoothness[0]
    assert np.all((0.0 <= fleet.centering) & (fleet.centering <= 1.0))
```

- [ ] **Step 2:** падает; RED-коммит `test: specify configurable fleet model`.

- [ ] **Step 3: Реализация** в `src/rl_fun/racing/fleet.py` по разделам 4.2–4.4 спеки:
  - Параметр `model: ModelSpec | None = None`. Если задан вместе с `ray_angles_deg` (когда пользователь передал его явно) — `ValueError`; для этого значением по умолчанию `ray_angles_deg` сделать `None`, а для старого поведения подставлять умолчание `(-90, -30, 0, 30, 90)`. Если передан только `ray_angles_deg`, строить `ModelSpec` с входами `ray:<угол>` + `speed`, `lateral_speed`, `yaw_rate`. Существующий публичный атрибут `ray_angles` (в радианах) и `observation_size` сохранить.
  - `fleet.model`, `fleet.action_size = len(model.outputs)`. `step` принимает `(N, action_size)`, иначе `ValueError`; управление через `map_controls`, буст передаётся в `step_arrays`.
  - Сборка наблюдения: словарь признаков (все лучи для всех углов модели, `speed`, `lateral_speed`, `yaw_rate`, `acceleration`, `steering_angle`) и сборка столбцов в порядке `model.inputs`. Нормировки прежние (лучи на `ray_range`, скорости на `max_speed`, угловая на `max_yaw_rate`, обрезка). `acceleration = clip((v_new − v_old) / dt / dynamics.acceleration, −1, 1)`; `steering_angle` — последний применённый руль (0 после `reset`).
  - Статистики по живым шагам (обновляются только у машинок, живых в начале шага): суммы скорости/`max_speed`, центрирования (`1 − clip(distance/(width/2), 0, 1)`), изменения руля (`|steer − предыдущий|/2`) и счётчик шагов; публичные свойства `mean_speed`, `centering`, `smoothness` (`smoothness = 1 − среднее изменение`, а при нуле шагов все три равны 0).
  - Поведение при умолчаниях прежнее (все существующие тесты флота остаются зелёными).

- [ ] **Step 4:** тесты и весь набор зелёные, ruff чистый. Commit `feat: configurable fleet inputs, outputs and driving stats`.

---

### Task 4: Составной фитнес и результат поколения

- [ ] **Step 1: Тесты** — `tests/racing/test_fitness.py`

```python
import math

import numpy as np
import pytest

from rl_fun.racing.fitness import PRESETS, FitnessSpec, available_terms
from rl_fun.racing.generation import GenerationResult


def make_result() -> GenerationResult:
    return GenerationResult(
        progress=np.array([0.5, 1.0]),
        finished=np.array([False, True]),
        steps_alive=np.array([100, 200]),
        lap_steps=np.array([np.nan, 150.0]),
        mean_speed=np.array([0.4, 0.8]),
        centering=np.array([0.9, 0.5]),
        smoothness=np.array([0.7, 0.9]),
        max_steps=400,
    )


def test_progress_only_matches_progress():
    assert FitnessSpec({"progress": 1.0}).score(make_result()) == pytest.approx([0.5, 1.0])


def test_weighted_sum_of_terms():
    spec = FitnessSpec({"progress": 1.0, "speed": 0.5})

    assert spec.score(make_result()) == pytest.approx([0.7, 1.4])


def test_lap_bonus_rewards_finished_cars_only():
    score = FitnessSpec({"lap_bonus": 1.0}).score(make_result())

    assert score == pytest.approx([0.0, 1 - 150 / 400])


def test_survival_term():
    assert FitnessSpec({"survival": 1.0}).score(make_result()) == pytest.approx([0.25, 0.5])


def test_negative_weight_acts_as_penalty():
    assert FitnessSpec({"speed": -1.0}).score(make_result()) == pytest.approx([-0.4, -0.8])


@pytest.mark.parametrize(
    "weights", [{}, {"warp": 1.0}, {"progress": math.nan}, {"progress": math.inf}]
)
def test_invalid_weights_raise(weights: dict):
    with pytest.raises(ValueError):
        FitnessSpec(weights)


def test_presets_are_valid_and_cover_known_terms():
    assert {"racer", "careful", "balanced"} <= set(PRESETS)
    for spec in PRESETS.values():
        assert set(spec.weights) <= set(available_terms())
        assert spec.score(make_result()).shape == (2,)


def test_dict_round_trip():
    spec = FitnessSpec({"progress": 1.0, "centering": 0.25})

    assert FitnessSpec.from_dict(spec.to_dict()) == spec


def test_result_without_stats_cannot_score_stat_terms():
    bare = GenerationResult(
        progress=np.zeros(2), finished=np.zeros(2, bool),
        steps_alive=np.zeros(2, int), lap_steps=np.full(2, np.nan),
    )

    assert FitnessSpec({"progress": 1.0}).score(bare) == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError):
        FitnessSpec({"speed": 1.0}).score(bare)
```

Добавить в `tests/racing/test_generation.py` (в конец) тест, что прогон заполняет новые поля:

```python
def test_result_carries_driving_stats():
    fleet = RacingFleet(2, max_steps=30)

    result = run_generation(fleet, np.zeros((2, 1)), forward_straight)

    assert result.max_steps == 30
    assert result.mean_speed.shape == (2,) and result.mean_speed.min() > 0
    assert result.centering.shape == (2,)
    assert result.smoothness.shape == (2,)
```

- [ ] **Step 2:** падает; RED-коммит `test: specify composite fitness`.

- [ ] **Step 3: Реализация.**
  - `GenerationResult`: добавить необязательные поля `mean_speed`, `centering`, `smoothness` (`np.ndarray | None = None`) и `max_steps: int | None = None`. `run_generation` заполняет их из флота (копии) и теперь формирует массив действий формы `(N, fleet.action_size)` вместо `(N, 2)`; `forward` по-прежнему возвращает вектор длины `action_size`.
  - `src/rl_fun/racing/fitness.py`: `available_terms()` → `("progress", "lap_bonus", "speed", "survival", "centering", "smoothness")`; `FitnessSpec` — замороженный dataclass с `weights: dict[str, float]` (копия в `__post_init__`; `ValueError` для пустого набора, неизвестного слагаемого, нефинитного веса), `score(result) -> ndarray (N,)` по разделу 4.5 (слагаемое, требующее отсутствующих данных, — `ValueError` с понятным текстом; `lap_bonus` обрабатывает `nan`), `to_dict`/`from_dict`. Профили: `racer = {progress: 1, lap_bonus: 0.5, speed: 0.3}`, `careful = {progress: 1, centering: 0.5, smoothness: 0.3, survival: 0.2}`, `balanced = {progress: 1, lap_bonus: 0.3, speed: 0.15, centering: 0.15}`. Модуль `fitness` не импортирует `generation` напрямую (тип результата — `Protocol` или `TYPE_CHECKING`).

- [ ] **Step 4:** тесты и весь набор зелёные, ruff чистый. Commit `feat: add composite fitness`.

---

### Task 5: Эталон и подписи под модель

- [ ] **Step 1: Тесты** — `tests/racing/test_reference_model.py`

```python
import numpy as np
import pytest

from rl_fun.racing import reference as ref
from rl_fun.racing.fitness import FitnessSpec
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.fleet_view import FleetView
from rl_fun.racing.model_spec import ModelSpec


def test_relu_and_sigmoid_hidden_layers_change_the_output():
    sizes = [4, 5, 2]
    weights = ref.init_population(np.random.default_rng(1), 1, sizes, 1.0)[0]
    observation = np.array([0.3, -0.7, 0.9, 0.1])

    tanh = ref.forward(weights, observation, sizes)
    relu = ref.forward(weights, observation, sizes, activation="relu")
    sigmoid = ref.forward(weights, observation, sizes, activation="sigmoid")

    assert not np.allclose(tanh, relu)
    assert not np.allclose(tanh, sigmoid)
    for action in (tanh, relu, sigmoid):
        assert np.all(np.abs(action) <= 1.0)


def test_inspect_uses_the_same_activation():
    sizes = [4, 5, 2]
    weights = ref.init_population(np.random.default_rng(1), 1, sizes, 1.0)[0]
    observation = np.array([0.3, -0.7, 0.9, 0.1])

    _, activations = ref.inspect(weights, observation, sizes, activation="relu")

    assert activations[-1] == pytest.approx(
        ref.forward(weights, observation, sizes, activation="relu")
    )
    assert np.all(activations[1] >= 0.0)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        ref.forward(np.zeros(10), np.zeros(2), [2, 2], activation="swish")


def test_train_accepts_a_custom_model_and_fitness():
    model = ModelSpec(
        inputs=("ray:-45", "ray:0", "ray:45", "speed"),
        hidden=(4,),
        outputs=("steer", "accelerate", "brake"),
    )
    params = ref.EvolutionParams(population=12, elite=3)

    run = ref.train(
        "oval", seed=3, generations=3, params=params, fleet_kwargs={"max_steps": 150},
        model=model, fitness=FitnessSpec({"progress": 1.0, "centering": 0.2}),
    )

    assert run.sizes == model.layer_sizes
    assert run.best_weights.shape == (model.weight_count,)
    assert len(run.history) == 3


def test_default_training_is_unchanged():
    params = ref.EvolutionParams(population=12, elite=3)

    run = ref.train("oval", seed=3, generations=2, params=params, fleet_kwargs={"max_steps": 150})

    assert run.sizes == ref.layer_sizes(8, params.hidden)


def test_fleet_view_takes_labels_from_the_fleet_model():
    model = ModelSpec(
        inputs=("ray:0", "speed"), hidden=(3,), outputs=("steer", "accelerate", "brake")
    )
    fleet = RacingFleet(3, model=model)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=True)
    try:
        assert view.panel is not None
        assert list(view.panel.input_labels) == model.input_labels
        assert list(view.panel.output_labels) == model.output_labels
    finally:
        view.close()
```

- [ ] **Step 2:** падает; RED-коммит `test: specify model-aware reference and view labels`.

- [ ] **Step 3: Реализация.**
  - `reference.forward(weights, observation, sizes, activation="tanh")` и `reference.inspect(..., activation="tanh")`: скрытые слои с `tanh` / `relu` (`max(0, x)`) / `sigmoid` (`1/(1+exp(-x))`), выходной слой всегда `tanh`; неизвестное имя — `ValueError`. Прежнее поведение при `activation="tanh"`.
  - `reference.train(..., model=None, fitness=None)`: при заданной `model` флот создаётся с `model=model`, размеры слоёв берутся из `model.layer_sizes` (а `params.hidden` игнорируется, о чём сказать в docstring), активация — `model.activation`; `fitness` по умолчанию `FitnessSpec({"progress": 1.0, "lap_bonus": 1.0})` (равносильно прежней формуле; бенчмарк `oval` остаётся воспроизводимым), счёт считается через `fitness.score(result)`. `history` по-прежнему содержит `best`/`mean`/`finished` по прогрессу.
  - `FleetView`: подписи входов и выходов по умолчанию берутся из `fleet.model.input_labels` / `output_labels`; у `NetworkPanel` должны быть публичные атрибуты `input_labels` и `output_labels`. Явно переданные подписи имеют приоритет. Если у `NetworkPanel` такие атрибуты называются иначе, переименовать с сохранением совместимости.

- [ ] **Step 4:** тесты, весь набор (включая `test_benchmark_is_reproducible` и тест выполнения ноутбука), ruff зелёные. Commit `feat: model-aware reference training and view labels`.

---

### Task 6: README и финальная проверка

- [ ] В `README.md` в раздел «Эволюция машинок» добавить абзац (по-русски): описание `ModelSpec` (входы, выходы, слои, активация), список доступных входов и выходов, составной фитнес и профили `racer` / `careful` / `balanced`, пример кода (`RacingFleet(..., model=ModelSpec(...))`, `reference.train(model=..., fitness=PRESETS["careful"])`) и замечание, что это основа будущего интерфейса.
- [ ] `uv run --all-groups pytest -q` и `uv run --all-groups ruff check .` — зелёные. Commit `docs: document model spec and composite fitness`.

---

## Self-Review

- **Покрытие спеки:** §4.1 — Task 1; §4.2 и §4.3 — Task 2; §4.4 — Task 3; §4.5 и поля результата — Task 4; §4.6 (`reference`, `FleetView`) — Task 5; README — Task 6.
- **Согласованность имён:** `ModelSpec` (`inputs`, `hidden`, `outputs`, `activation`, `layer_sizes`, `weight_count`, `ray_angles_deg`, `input_labels`, `output_labels`), `map_controls`, `fleet.model` / `action_size` / `mean_speed` / `centering` / `smoothness`, `GenerationResult(..., mean_speed, centering, smoothness, max_steps)`, `FitnessSpec(weights)`, `PRESETS` везде одинаковы.
- **Риски:** совместимость умолчаний с прежним флотом и бенчмарком — проверяется прежними тестами.
