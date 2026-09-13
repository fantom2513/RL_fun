# Интеграция Gymnasium в runtime RL Fun

**Дата:** 2026-09-13  
**Статус:** направление согласовано, письменный спек ожидает ревью

## 1. Зачем нужен этот этап

Первый этап подтвердил, что проект умеет воспроизводимо запускать несколько seed для стационарного многорукого бандита и сохранять независимые артефакты. Но конфигурация и runner пока привязаны именно к bandit-задаче.

Цель второго этапа — убрать эту привязку и научить проект запускать обычные Gymnasium-среды по ID. После этого новую среду — змейку, машинку, маятник или другую симуляцию — можно будет добавлять обычной задачей в Codex, не переписывая инфраструктуру экспериментов.

Это не генератор сред, не встроенный чат, не игровой движок и не универсальный Trainer. Репозиторий предоставляет понятный контракт и runtime, а каждая содержательная среда остаётся обычным Python-кодом.

Учебные пояснения, README, notebook Markdown, подписи графиков и CLI-подсказки пишутся на русском. Python-идентификаторы и стандартные термины API остаются на английском.

## 2. Что входит в этап

- Создание встроенных и локальных сред через `gymnasium.make()` по ID и параметрам конструктора.
- Регистрация собственных сред в пространстве имён `RLFun/`.
- Раздельные секции конфигурации для среды, алгоритма, запуска и оценки.
- Общая граница одного запуска без знания о конкретной задаче.
- Именованные метрики вместо обязательных bandit-полей.
- Небольшой явный registry функций-алгоритмов.
- Общий rollout для оценки и ручного просмотра.
- Проверка архитектуры на существующем bandit и стандартном `CartPole-v1`.
- Сохранение параллельных multi-seed запусков и изоляции артефактов.
- Короткая русская инструкция по добавлению следующей пользовательской среды.

## 3. Что не входит

- Генерация среды по текстовому описанию внутри проекта.
- Собственный чат, copilot или визуальный редактор уровней.
- Универсальная декларативная схема физики и игровой логики.
- Реализация гонок, змейки, GridWorld или новой физической среды.
- Сравнительная Pygame Arena и Neural Inspector.
- PyTorch, DQN, PPO, Stable-Baselines3, TensorBoard, Hydra и MLflow.
- Multi-agent API; при реальной потребности он потребует отдельного решения, вероятно на базе PettingZoo.
- Абстрактный Trainer, скрывающий внутренний цикл любого алгоритма.
- Поддержка старого JSON-формата навсегда: committed-конфиг мигрирует без compatibility layer.

## 4. Рассмотренные варианты

### 4.1. Жёсткие ветки в runner

Можно добавлять новый `if` или `match` для каждой пары «среда + алгоритм». Сначала кода мало, но центральный модуль быстро превращается в таблицу всех комбинаций. Вариант отклонён.

### 4.2. Registry Gymnasium и маленький registry алгоритмов — выбран

Gymnasium уже решает обнаружение и создание сред, wrappers, render modes и версионирование ID. Проект использует этот механизм напрямую.

Локальные алгоритмы остаются короткими функциями. Явный словарь сопоставляет короткий ID алгоритма с функцией и проверкой совместимости. Это минимальная абстракция, которую уже оправдывают два разных сценария: bandit и CartPole.

### 4.3. Полноценная plugin-система

Entry points, dependency injection, lifecycle hooks и metadata полезны публичной платформе с внешними расширениями. Для личной песочницы это лишний слой, поэтому вариант отклонён.

## 5. Конфигурация версии 2

Плоская конфигурация первого этапа заменяется вложенными секциями:

```json
{
  "name": "cartpole-random",
  "environment": {"id": "CartPole-v1", "kwargs": {}},
  "algorithm": {"id": "random", "kwargs": {}},
  "run": {
    "seeds": [11, 22, 33, 44],
    "total_steps": 1000,
    "workers": 4,
    "output_root": "runs"
  },
  "evaluation": {"episodes": 5, "max_episode_steps": 500}
}
```

Структуру представляют неизменяемые dataclass:

```text
ExperimentConfig
├── name: str
├── environment: EnvironmentConfig
│   ├── id: str
│   └── kwargs: dict[str, JSONValue]
├── algorithm: AlgorithmConfig
│   ├── id: str
│   └── kwargs: dict[str, JSONValue]
├── run: RunConfig
│   ├── seeds: tuple[int, ...]
│   ├── total_steps: int
│   ├── workers: int
│   └── output_root: str
└── evaluation: EvaluationConfig
    ├── episodes: int
    └── max_episode_steps: int | None
```

Оба `kwargs` принимают только JSON-совместимые значения. Валидация отклоняет пустые ID, отсутствие seed, дубликаты seed, неположительные количества, `bool` вместо целых чисел и неизвестные ключи секций. Проверку специфичных параметров конструктора выполняет сама среда; сообщение об ошибке дополнительно содержит её ID.

Committed bandit-конфиг мигрирует на версию 2. Старый плоский формат получает понятную ошибку с указанием на новый формат; отдельного legacy parser не будет.

`render_mode` намеренно не хранится в `environment.kwargs`. Его задаёт режим выполнения:

- обучение и headless-оценка: аргумент не передаётся;
- ручной просмотр: `"human"`;
- будущая запись или передача кадров: `"rgb_array"`.

Это соответствует Gymnasium: render mode выбирается при создании среды.

## 6. Создание и регистрация сред

`make_environment(config, render_mode=None)` — единственная фабрика runtime. Она импортирует `rl_fun.environments`, чтобы установить локальные регистрации, а затем вызывает `gymnasium.make(config.id, **kwargs)`.

Если `render_mode is None`, ключ вообще не передаётся: сторонний конструктор не обязан одинаково обрабатывать отсутствие аргумента и явный `None`.

Собственные среды получают версионированные ID под namespace `RLFun/`. Существующий bandit регистрируется как:

```python
gymnasium.register(
    id="RLFun/StationaryBandit-v0",
    entry_point="rl_fun.environments.bandit:StationaryBanditEnv",
)
```

Регистрация идемпотентна в пределах процесса. Повторный импорт не должен падать. Если тот же `RLFun/` ID уже указывает на другой entry point, это явная ошибка, а не молчаливая замена.

Новые среды остаются обычными подклассами `gymnasium.Env`: объявляют пространства действий и наблюдений, реализуют нужные `reset`, `step`, `render`, `close` и вызывают `super().reset(seed=seed)` для собственной случайности.

`gymnasium.utils.env_checker.check_env` выполняется в тестах каждой локальной среды, но не при каждом запуске эксперимента.

## 7. Граница выполнения алгоритма

Runner отвечает за оркестрацию, а не за обучающий цикл. Концептуальный контракт алгоритма:

```python
def run_algorithm(
    env: gymnasium.Env,
    total_steps: int,
    seed: int,
    rng: numpy.random.Generator,
    metrics: MetricSink,
    parameters: Mapping[str, JSONValue],
) -> AlgorithmResult:
    ...
```

Именно алгоритм владеет вызовами `reset()` и `step()`: только он знает, нужен ли один непрерывный процесс, несколько эпизодов или особая логика обновления. Первый `reset(seed=seed)` выполняет алгоритм, а последующие эпизоды получают детерминированно производные seed. Runner не выполняет скрытый пробный reset перед алгоритмом.

`AlgorithmResult` содержит:

```text
AlgorithmResult
├── metrics: dict[str, float]
└── policy: Policy | None
```

Формат внутреннего состояния модели и checkpoint пока не стандартизируется. Он появится вместе с первым реальным stateful-алгоритмом.

Явный registry содержит:

```text
random          → общий алгоритм случайных rollout
bandit_greedy   → адаптер существующего greedy
bandit_epsilon  → адаптер существующего epsilon-greedy
```

`random` работает с любой single-agent Gymnasium-средой, у которой можно вызвать `action_space.sample()`. Bandit-алгоритмы проверяют необходимые свойства unwrapped-среды. Несовместимая пара завершается до обучающего цикла с ID среды и алгоритма в сообщении.

Адаптеры остаются тонкими. Учебные файлы алгоритмов по-прежнему показывают цикл непосредственно и не импортируют registry.

## 8. Общий сценарий одного запуска

Bandit-специфичный `run_bandit_once` заменяется на `run_once`:

1. Проверить typed-конфигурацию и запрошенный seed.
2. Найти описание алгоритма в registry.
3. Создать отдельную директорию запуска.
4. Записать точную конфигурацию v2 и runtime metadata.
5. Создать среду без renderer через `make_environment`.
6. Создать независимый RNG алгоритма из seed.
7. Проверить совместимость алгоритма со средой без изменения её состояния.
8. Передать среду алгоритму, который сам управляет эпизодами, и записывать именованные метрики.
9. Закрыть среду в `finally` при любом исходе.
10. Записать общий summary и вернуть его.

`run_many` по-прежнему разворачивает список seed и вызывает picklable worker верхнего уровня. Worker теперь вызывает `run_once`; процессная модель, уникальные директории и сортировка результатов по seed сохраняются.

## 9. Результаты и метрики

В `RunSummary` больше нет обязательных `cumulative_reward` и `cumulative_regret`:

```text
RunSummary
├── status: success | failure | cancelled
├── seed: int
├── run_dir: Path
├── metrics: dict[str, float]
├── elapsed_seconds: float
└── error: str | None
```

Общие имена — соглашение, а не обязательные поля:

- `episode/reward`;
- `episode/length`;
- `train/cumulative_reward`;
- `train/cumulative_regret`;
- `train/optimal_action_rate`.

Bandit переносит текущие значения в namespace `train/`. Rollout пишет награду и длину эпизода. Сравнение принимает имя метрики явно, игнорирует неуспешные запуски и понятно сообщает, если нужной метрики нигде нет.

Metric JSONL сохраняет форму `{step, metrics}`, поэтому существующему reader достаточно работать с новыми именами.

## 10. Policy и rollout

Policy — минимальный stateful-объект для оценки и просмотра:

```python
class Policy(Protocol):
    def reset(self, seed: int, action_space: gymnasium.Space) -> None: ...

    def act(self, observation: object, deterministic: bool) -> object: ...
```

`RandomPolicy.reset()` сидирует переданное `action_space` и сохраняет ссылку на него; `act()` вызывает `action_space.sample()`. Так используется собственный RNG пространства Gymnasium, а глобальная случайность не затрагивается. Другие policy могут создавать свой RNG из того же seed.

`rollout_episode(env, policy, seed, max_episode_steps)` отвечает только за взаимодействие:

1. вызвать `policy.reset(seed, env.action_space)`;
2. вызвать `env.reset(seed=seed)`;
3. запросить действие у policy;
4. вызвать `env.step(action)`;
5. накопить награду и длину;
6. остановиться по `terminated`, `truncated` или safety limit.

Rollout ничего не обучает и не предполагает структуру observation. Среду закрывает вызывающая сторона. Состояние будущей recurrent policy может жить внутри policy без изменения rollout.

Для нескольких эпизодов вызывающая сторона получает их seed детерминированно из seed запуска; повторный запуск с тем же конфигом воспроизводим.

## 11. Режимы выполнения

### 11.1. Train

`scripts/train.py` загружает конфиг v2, выполняет один или несколько seed и печатает общий aggregate summary. Мигрированный bandit работает через тот же путь.

### 11.2. Evaluate

`scripts/evaluate.py` запускает policy на `evaluation.episodes` без human rendering. На этом этапе общая policy только одна — `RandomPolicy`. Загрузка обученной policy появится вместе с первым алгоритмом, который создаёт reusable checkpoint. Результаты эпизодов сохраняются текущей системой артефактов.

### 11.3. Play

`scripts/play.py` создаёт одну среду с `render_mode="human"`, берёт один seed из конфигурации и запускает случайную policy. Частоту кадров задаёт среда через `metadata["render_fps"]`; скрипт не добавляет задержку, если среда уже сама управляет pacing.

Это ещё не Arena: одновременно показывается только одна среда и одна policy. Для human-rendering CartPole потребуется rendering extra Gymnasium classic-control. Добавление зависимости выполняется только после отдельного подтверждения перед реализацией.

## 12. Соглашение для новых сред

Если сложность среды оправдывает разбиение, используется структура:

```text
src/rl_fun/environments/<name>/
├── __init__.py
├── env.py
├── config.py        # только при содержательной валидации параметров
└── renderer.py      # только при существенном объёме отрисовки
```

Маленькая среда может остаться одним модулем. Тесты лежат в соответствующем `tests/environments/<name>/`.

Задача на новую среду должна определить:

- назначение и границы эпизода;
- observation и action spaces;
- параметры конструктора и допустимые диапазоны;
- принадлежащую среде случайность и сидирование;
- различие `terminated` и `truncated`;
- render modes и `render_fps`;
- ручной smoke-сценарий;
- Gymnasium contract test;
- ID регистрации и версию.

В репозитории появится короткая русская инструкция. Это памятка для обычных задач Codex и ручной разработки, а не исполняемый генератор.

## 13. Язык учебных материалов

На этом этапе существующие README и bandit notebook переводятся на русский, чтобы в проекте не было двух конкурирующих соглашений.

На русском пишутся:

- Markdown-пояснения в notebooks;
- README и инструкция по созданию среды;
- заголовки, оси и легенды графиков;
- CLI help для пользователя;
- комментарии, объясняющие алгоритм.

Идентификаторы, JSON-ключи, ключи метрик, environment ID, имена исключений и стандартные Python API-термины могут оставаться английскими. При первом появлении важного понятия допустима пара: «наблюдение (`observation`)».

## 14. Ошибки и завершение

- Неизвестная среда: запрошенный ID плюс исходная диагностика Gymnasium.
- Неизвестный алгоритм: ID и список поддерживаемых локальных алгоритмов.
- Неверные constructor kwargs: ID среды, имена переданных ключей и исходное исключение.
- Несовместимая пара: отказ до записи обучающих метрик.
- Human rendering не поддерживается: ID среды и объявленные render modes.
- Ошибка одного seed: `error.txt` и failure summary; остальные seed продолжают работу.
- `close()` вызывается в `finally` для train, evaluate и play.
- `Ctrl+C`: exit code 130 с сохранением уже завершённых директорий.
- Compare: понятная ошибка, если ни один успешный запуск не содержит выбранную метрику.

## 15. Стратегия тестирования

Focused-тесты покрывают:

- parse, validation и round trip конфигурации v2;
- понятный отказ для старого плоского формата;
- создание `CartPole-v1` через фабрику;
- регистрацию и создание `RLFun/StationaryBandit-v0` через `gymnasium.make`;
- идемпотентность локальной регистрации;
- воспроизводимый reset bandit;
- случайный rollout на CartPole;
- явный отказ bandit-алгоритма на CartPole;
- сериализацию общего `RunSummary`;
- изоляцию multi-seed артефактов после перехода на `run_once`;
- агрегацию выбранной вызывающим кодом метрики;
- CLI smoke для bandit и CartPole random policy;
- наличие русских учебных пояснений в README и notebook Markdown.

Human rendering проверяется вручную: GUI-окна ненадёжны в CI. `rgb_array` проверим вместе с первой собственной визуальной средой или Arena, а не будем имитировать заранее.

## 16. Изменения по файлам

Существующие файлы:

- `src/rl_fun/experiments/config.py` — вложенные dataclass v2 и строгая JSON-валидация.
- `src/rl_fun/experiments/result.py` — общий mapping метрик.
- `src/rl_fun/experiments/runner.py` — вызов общего `run_once`.
- `src/rl_fun/experiments/compare.py` — агрегация выбранной метрики.
- `src/rl_fun/experiments/bandit.py` — только тонкие bandit-адаптеры либо удаление после миграции callers.
- `src/rl_fun/environments/__init__.py` — локальные Gymnasium-регистрации.
- текущие bandit-алгоритмы — только переход на namespace метрик.
- `scripts/train.py`, `scripts/compare.py` — конфиг v2 и выбор метрики.
- `configs/bandit-epsilon-greedy.json` — схема v2.
- `README.md` и существующий bandit notebook — русские учебные материалы.

Новые файлы:

- `src/rl_fun/environments/factory.py` — создание Gymnasium-сред.
- `src/rl_fun/algorithms/registry.py` — явный registry и проверки совместимости.
- `src/rl_fun/policies.py` — `Policy` и `RandomPolicy`.
- `src/rl_fun/rollouts.py` — общий эпизодный rollout.
- `src/rl_fun/experiments/run.py` — оркестрация `run_once`.
- `scripts/evaluate.py` — headless-оценка.
- `scripts/play.py` — human playback одной среды.
- `configs/cartpole-random.json` — пример стандартной Gymnasium-среды.
- `docs/environment-authoring.md` — русская инструкция по добавлению среды.
- соответствующие focused-тесты в `tests/`.

Пустого package tree, сгенерированных шаблонов и фиктивной custom-среды «для вида» не добавляется.

## 17. Поток данных

```text
JSON config v2
      |
      v
typed ExperimentConfig
      |
      v
run_once(seed) -----> algorithm function -----> AlgorithmResult.metrics
      |                                             |
      +---- MetricSink JSONL                        v
      +---- config + metadata                generic RunSummary
      +---- error.txt при сбое                       |
                                                    v
                                             multi-seed aggregate

evaluate/play -----> make_environment(render mode) -----> Policy -----> rollout_episode
```

## 18. Критерии готовности

- Bandit запускается общим runner через `RLFun/StationaryBandit-v0` и конфиг v2.
- CartPole со случайным алгоритмом запускается из JSON без изменения центрального runner.
- После импорта `rl_fun.environments` вызов `gymnasium.make("RLFun/StationaryBandit-v0", ...)` создаёт среду проекта.
- Параметры среды и алгоритма представлены и сохраняются независимо.
- Multi-seed CPU-запуск сохраняет изолированные success/failure артефакты.
- Summary и compare работают с именованными метриками, а не bandit-полями.
- `evaluate.py` headlessly завершает несколько эпизодов CartPole.
- После одобрения rendering dependency `play.py` открывает CartPole и корректно закрывается в ручном smoke-тесте.
- Неизвестная среда, неизвестный алгоритм, несовместимая пара и неверные kwargs дают понятные ошибки.
- Все автоматические тесты и Ruff проходят.
- README, пояснения существующего notebook, подписи графиков и новая инструкция написаны на русском.

## 19. Что идёт дальше

После этого этапа выбирается одна конкретная визуальная среда. Хороший кандидат — top-down гонка: она проверит непрерывное управление, геометрические датчики, столкновения, renderer, ручную игру и дальнейшее визуальное сравнение.

Общие geometry/rendering helpers извлекаются только из реальной реализации, а не проектируются заранее. Сравнительная Arena появляется после хотя бы одной собственной визуальной среды и двух содержательных policy. PyTorch/DQN и Neural Inspector остаются отдельными последующими этапами.

## 20. Источники

- Gymnasium Env API и render modes: <https://gymnasium.farama.org/api/env/>
- Создание и регистрация custom environment: <https://gymnasium.farama.org/main/tutorials/environment_creation/>
- Проверка среды через `check_env`: <https://gymnasium.farama.org/api/utils/>
