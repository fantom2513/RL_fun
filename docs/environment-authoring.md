# Как добавить новую среду

Среда определяет наблюдения, действия, награды и конец эпизода. Runtime создаёт
её через Gymnasium ID, передаёт seed, запускает алгоритм, сохраняет метрики и
закрывает ресурсы. Выбор подходящего алгоритма остаётся отдельной задачей:
`random` работает с action space, у которого есть `sample()`, а бандитные
алгоритмы требуют `StationaryBanditEnv` с неизменёнными действиями.

## Минимальная среда

Пример ниже — исполняемая среда-счётчик: действие 1 увеличивает состояние,
действие 0 сохраняет его. Достижение цели завершает эпизод; лимит времени
обрезает незавершённый эпизод. Сохраните класс в `src/rl_fun/environments/counter.py`.

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CounterEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, goal: int = 5, horizon: int = 20, render_mode: str | None = None):
        if goal <= 0 or horizon <= 0:
            raise ValueError("goal и horizon должны быть положительными")
        if render_mode is not None:
            raise ValueError("Эта среда работает без отображения")
        self.render_mode = render_mode
        self.goal = goal
        self.horizon = horizon
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, goal, shape=(1,), dtype=np.int64)
        self.position = 0
        self.steps = 0
        self.finished = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.position = 0
        self.steps = 0
        self.finished = False
        return np.array([self.position], dtype=np.int64), {}

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("Перед новым эпизодом вызовите reset")
        if not self.action_space.contains(action):
            raise ValueError("Недопустимое действие")
        self.position += int(action)
        self.steps += 1
        terminated = self.position >= self.goal
        truncated = self.steps >= self.horizon and not terminated
        self.finished = terminated or truncated
        return (
            np.array([self.position], dtype=np.int64),
            float(terminated),
            terminated,
            truncated,
            {},
        )

    def close(self):
        pass
```

`observation_space` должен описывать тип, форму и диапазон каждого наблюдения,
а `action_space` — допустимые действия. Не возвращайте изменяемую ссылку на
внутреннее состояние. Конструктор принимает параметры среды, но не начинает
эпизод и не открывает окно для режима без отображения.

## Seed и границы эпизода

Всегда вызывайте `super().reset(seed=seed)`. Случайность среды берите из
`self.np_random`, а не из глобального генератора NumPy. Одинаковый seed при
одинаковых действиях должен воспроизводить переходы и награды. Policy использует
свой генератор; случайный action space инициализируется отдельно runtime.

`reset` возвращает `(observation, info)`, `step` —
`(observation, reward, terminated, truncated, info)`. `terminated` означает
конечное состояние задачи, например достижение цели. `truncated` означает
внешнее ограничение, например лимит времени. После любого из этих сигналов
следующий эпизод начинается с `reset`; safety limit runtime не следует выдавать
за естественно завершённый эпизод. Награда должна быть числом, `info` — словарём.

## Регистрация и конфигурация

Добавьте запись в `LOCAL_ENVIRONMENTS` в `src/rl_fun/environments/__init__.py`:

```python
"RLFun/Counter-v0": "rl_fun.environments.counter:CounterEnv",
```

`register_environments()` регистрирует ID повторно безопасно, но отклоняет
конфликтующий entry point. Фабрика `make_environment` вызывает `gym.make`.
Для изменения несовместимого публичного поведения увеличивайте версию ID.

Конфигурация нового эксперимента:

```json
{
  "name": "counter-random",
  "environment": {"id": "RLFun/Counter-v0", "kwargs": {"goal": 5, "horizon": 20}},
  "algorithm": {"id": "random", "kwargs": {}},
  "run": {"seeds": [11, 22], "total_steps": 100, "workers": 1, "output_root": "runs"},
  "evaluation": {"episodes": 5, "max_episode_steps": 20}
}
```

Не помещайте `render_mode` в конфигурацию: этот параметр фабрика принимает
отдельно. Для добавления алгоритма расширяйте локальный реестр алгоритмов,
а не код общей фабрики или runner.

## Проверка среды

После реализации выполните проверку Gymnasium и один эпизод:

```python
from gymnasium.utils.env_checker import check_env
from rl_fun.environments.counter import CounterEnv
from rl_fun.environments.factory import make_environment
from rl_fun.experiments.config import EnvironmentConfig
from rl_fun.policies import RandomPolicy
from rl_fun.rollouts import rollout_episode

env = CounterEnv()
try:
    check_env(env, skip_render_check=True)
finally:
    env.close()

env = make_environment(EnvironmentConfig("RLFun/Counter-v0"))
try:
    episode = rollout_episode(env, RandomPolicy(), seed=11, max_episode_steps=20)
    assert 1 <= episode.length <= 20
finally:
    env.close()
```

Добавляйте тесты в `tests/environments/`: повторяемый reset, допустимость
наблюдений, награды и оба способа завершения. Сначала зафиксируйте падающие
тесты, затем реализацию. Проверьте запуск через фабрику и `random`, чтобы
поймать ошибки регистрации и взаимодействия с runtime.

## Отображение и чек-лист

Если среда поддерживает отображение, перечислите реальные режимы в
`metadata["render_modes"]` и укажите `render_fps`. Режим `human` обновляет окно,
`rgb_array` возвращает массив пикселей; поддерживать их необязательно.
`close()` освобождает окно и остальные ресурсы и допускает повторный вызов.
В текущем этапе CLI визуального воспроизведения отложен до отдельного
согласования зависимости. Автоматические проверки выполняются без окна.

- Конструктор валидирует параметры; spaces совпадают с действиями и наблюдениями.
- Reset использует `super().reset(seed=seed)` и восстанавливает весь эпизод.
- Step корректно различает `terminated` и `truncated`.
- Регистрация `RLFun/<Name>-v0` не перезаписывает чужую среду.
- `check_env`, unit smoke и запуск через фабрику проходят.
- После реализации рендеринга вручную проверьте кадры, темп и закрытие окна;
  новые зависимости предварительно согласуйте.
