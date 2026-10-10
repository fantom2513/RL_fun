"""Parameter schemas of the learners, as plain data for the catalog.

The interface builds its forms from these descriptions, so a new learner needs only an entry here.
A parameter is `{key, label, type, min, max, step, default, log, live, hint}`: `key` is the path of
the field in `RunConfig` (`elite`, `ppo.learning_rate`), `live` tells whether it can change while
the run is going, `log` that a slider should be logarithmic.
"""

from __future__ import annotations

import copy
from typing import Any


def _param(
    key: str,
    label: str,
    kind: str,
    low: float,
    high: float,
    step: float,
    default: float,
    hint: str,
    *,
    live: bool,
    log: bool = False,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": kind,
        "min": low,
        "max": high,
        "step": step,
        "default": default,
        "log": log,
        "live": live,
        "hint": hint,
    }


def _max_steps() -> dict[str, Any]:
    return _param(
        "max_steps",
        "Шагов на заезд",
        "int",
        50,
        5000,
        50,
        1500,
        "Сколько шагов симуляции даётся машинке на один заезд (30 шагов — одна секунда). "
        "Чем больше, тем дольше можно ехать, но тем медленнее идёт обучение.",
        live=False,
    )


EVOLUTION_SCHEMA: dict[str, Any] = {
    "id": "evolution",
    "label": "Эволюция",
    "description": (
        "Популяция сетей ездит наперегонки, лучшие остаются, а из них с небольшими случайными "
        "изменениями рождается следующее поколение. Не требует вычислений градиентов и хорошо "
        "работает на простых трассах."
    ),
    "groups": [
        {
            "label": "Популяция",
            "params": [
                _param(
                    "population",
                    "Размер популяции",
                    "int",
                    2,
                    200,
                    1,
                    60,
                    "Сколько машинок ездит одновременно в каждом поколении. Больше машинок — "
                    "шире поиск, но каждое поколение считается дольше.",
                    live=False,
                ),
                _param(
                    "elite",
                    "Элита",
                    "int",
                    1,
                    100,
                    1,
                    6,
                    "Сколько лучших машинок переходит в следующее поколение без изменений. "
                    "Они же служат родителями остальных. Должно быть меньше размера популяции.",
                    live=True,
                ),
                _max_steps(),
            ],
        },
        {
            "label": "Мутация",
            "params": [
                _param(
                    "mutation_rate",
                    "Доля мутирующих весов",
                    "float",
                    0.01,
                    1.0,
                    0.01,
                    0.15,
                    "Какая часть весов сети меняется у потомка. Мало — потомки похожи на "
                    "родителей, много — почти случайный поиск.",
                    live=True,
                ),
                _param(
                    "mutation_scale",
                    "Сила мутации",
                    "float",
                    0.01,
                    2.0,
                    0.01,
                    0.3,
                    "Насколько сильно меняется каждый мутирующий вес. Большие значения дают "
                    "резкие скачки, малые — тонкую подстройку.",
                    live=True,
                    log=True,
                ),
                _param(
                    "init_scale",
                    "Масштаб начальных весов",
                    "float",
                    0.1,
                    3.0,
                    0.05,
                    1.0,
                    "Размах случайных весов в самом первом поколении. Влияет только на старт.",
                    live=False,
                ),
            ],
        },
    ],
}

PPO_SCHEMA: dict[str, Any] = {
    "id": "ppo",
    "label": "PPO",
    "description": (
        "Обучение с подкреплением (policy gradient): сеть-водитель постепенно подстраивается под "
        "награду, собирая опыт заездов и делая небольшие аккуратные шаги улучшения. Учится плавнее "
        "эволюции, но чувствительна к параметрам."
    ),
    "groups": [
        {
            "label": "Запуск",
            "params": [
                _param(
                    "population",
                    "Число машинок",
                    "int",
                    2,
                    200,
                    1,
                    60,
                    "Сколько машинок собирают опыт параллельно. Больше машинок — больше опыта "
                    "за итерацию и стабильнее обучение, но итерация длится дольше.",
                    live=False,
                ),
                _max_steps(),
            ],
        },
        {
            "label": "Обучение",
            "params": [
                _param(
                    "ppo.learning_rate",
                    "Скорость обучения",
                    "float",
                    1e-5,
                    1e-2,
                    1e-5,
                    3e-4,
                    "Размер шага, которым сеть меняет веса после каждой порции опыта. Слишком "
                    "большой — обучение скачет и срывается, слишком маленький — почти не "
                    "продвигается.",
                    live=True,
                    log=True,
                ),
                _param(
                    "ppo.clip_epsilon",
                    "Ограничение шага (clip)",
                    "float",
                    0.05,
                    0.5,
                    0.01,
                    0.2,
                    "Насколько за одно обновление может измениться поведение водителя. Малое "
                    "значение — осторожное, но медленное обучение.",
                    live=True,
                ),
                _param(
                    "ppo.entropy_coef",
                    "Любопытство (энтропия)",
                    "float",
                    0.0,
                    0.05,
                    0.001,
                    0.005,
                    "Награда за разнообразие действий. Выше — водитель дольше пробует новое, "
                    "ниже — быстрее закрепляет найденное.",
                    live=True,
                ),
                _param(
                    "ppo.value_coef",
                    "Вес оценки ситуации",
                    "float",
                    0.0,
                    2.0,
                    0.05,
                    0.5,
                    "Насколько важно учить сеть-оценщика, предсказывающую будущую награду. Она "
                    "помогает водителю понять, какие действия были удачными.",
                    live=True,
                ),
                _param(
                    "ppo.max_grad_norm",
                    "Предел градиента",
                    "float",
                    0.1,
                    5.0,
                    0.05,
                    0.5,
                    "Ограничение размера одного обновления весов, защита от внезапных скачков.",
                    live=False,
                ),
                _param(
                    "ppo.initial_std",
                    "Начальный разброс действий",
                    "float",
                    0.1,
                    2.0,
                    0.05,
                    0.6,
                    "Насколько случайно водитель действует в начале. Со временем сеть "
                    "сама уменьшает разброс.",
                    live=False,
                ),
            ],
        },
        {
            "label": "Опыт и награда",
            "params": [
                _param(
                    "ppo.rollout_steps",
                    "Шагов опыта на машинку",
                    "int",
                    16,
                    1024,
                    16,
                    128,
                    "Сколько шагов каждая машинка проезжает перед очередным обновлением сети. "
                    "Больше — точнее оценка, но обновления реже.",
                    live=False,
                ),
                _param(
                    "ppo.epochs",
                    "Проходов по опыту",
                    "int",
                    1,
                    20,
                    1,
                    4,
                    "Сколько раз сеть заново просматривает собранный опыт при обновлении.",
                    live=False,
                ),
                _param(
                    "ppo.minibatch_size",
                    "Размер порции опыта",
                    "int",
                    16,
                    8192,
                    16,
                    256,
                    "На какие порции делится опыт при обновлении. Меньше порции — больше "
                    "шагов обучения и больше шума.",
                    live=False,
                ),
                _param(
                    "ppo.gamma",
                    "Дальновидность (gamma)",
                    "float",
                    0.8,
                    0.999,
                    0.001,
                    0.99,
                    "Насколько водитель ценит награду в будущем по сравнению с наградой "
                    "сейчас. Ближе к 1 — учитывает дальние последствия.",
                    live=True,
                ),
                _param(
                    "ppo.gae_lambda",
                    "Сглаживание оценки (lambda)",
                    "float",
                    0.8,
                    1.0,
                    0.01,
                    0.95,
                    "Баланс между быстрой, но неточной оценкой удачности действия и точной, но "
                    "шумной. Обычно 0.9–0.97.",
                    live=True,
                ),
                _param(
                    "ppo.crash_penalty",
                    "Штраф за аварию",
                    "float",
                    0.0,
                    10.0,
                    0.1,
                    1.0,
                    "Сколько очков награды теряет машинка, вылетев с трассы. Помогает быстрее "
                    "понять, что съезжать нельзя.",
                    live=True,
                ),
            ],
        },
    ],
}


def _cem_schema() -> dict[str, Any]:
    """Cross-entropy method: the evolution parameters that make sense, described for CEM."""
    schema = copy.deepcopy(EVOLUTION_SCHEMA)
    schema["id"] = "cem"
    schema["label"] = "Кросс-энтропия (CEM)"
    schema["description"] = (
        "Метод кросс-энтропии: вокруг одной «средней» сети разбрасывается облако вариантов, "
        "лучшие из них задают новое среднее и новый разброс весов. Нет мутаций и скрещивания: "
        "облако само сжимается к удачным весам. Очень простой метод и хороший соперник эволюции."
    )
    population, mutation = schema["groups"]
    population["params"][1]["hint"] = (
        "Сколько лучших вариантов из облака определяют новое среднее. Меньше — быстрее "
        "сходится, но легче застрять; больше — осторожнее. Должно быть меньше размера популяции."
    )
    mutation["label"] = "Облако весов"
    mutation["params"] = [param for param in mutation["params"] if param["key"] != "mutation_rate"]
    for param in mutation["params"]:
        if param["key"] == "mutation_scale":
            param["label"] = "Добавочный шум"
            param["hint"] = (
                "Что прибавляется к разбросу весов после каждого поколения (одна десятая этого "
                "числа), чтобы облако не схлопнулось слишком рано. Больше — дольше ищет."
            )
        if param["key"] == "init_scale":
            param["label"] = "Начальный разброс весов"
            param["hint"] = "Размах облака в первом поколении. Влияет только на старт."
    return schema


def _a2c_schema() -> dict[str, Any]:
    """A2C: PPO without the clip, the epochs and the minibatches."""
    schema = copy.deepcopy(PPO_SCHEMA)
    schema["id"] = "a2c"
    schema["label"] = "A2C"
    schema["description"] = (
        "Actor-critic без «предохранителей» PPO: после каждого сбора опыта делается один шаг "
        "градиента по всему опыту сразу. Это самый прямой policy gradient: быстрее и проще, но "
        "шаги ничем не ограничены, и обучение легче срывается. Сравните с PPO на том же задании."
    )
    dropped = {"ppo.clip_epsilon", "ppo.epochs", "ppo.minibatch_size"}
    # with one step per rollout, short rollouts and a bigger step are what make A2C learn: a form
    # switched to A2C starts from these values (the API defaults stay those of PPO)
    recommended = {"ppo.learning_rate": 0.003, "ppo.rollout_steps": 16}
    for group in schema["groups"]:
        group["params"] = [param for param in group["params"] if param["key"] not in dropped]
        for param in group["params"]:
            if param["key"] in recommended:
                param["default"] = recommended[param["key"]]
    schema["groups"] = [group for group in schema["groups"] if group["params"]]
    return schema


CEM_SCHEMA: dict[str, Any] = _cem_schema()
A2C_SCHEMA: dict[str, Any] = _a2c_schema()

LEARNER_SCHEMAS: list[dict[str, Any]] = [EVOLUTION_SCHEMA, PPO_SCHEMA, CEM_SCHEMA, A2C_SCHEMA]


def copy_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """An independent deep copy, safe to modify."""
    return copy.deepcopy(schema)


def build_learner_schemas() -> list[dict[str, Any]]:
    """Deep copies of all learner schemas, for the catalog."""
    return [copy_schema(schema) for schema in LEARNER_SCHEMAS]
