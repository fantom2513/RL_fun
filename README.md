# RL Fun

RL Fun — учебная лаборатория обучения с подкреплением на Gymnasium. Один runtime запускает стационарного многорукого бандита и CartPole, сохраняет метрики каждого seed и сравнивает результаты. Доступны алгоритмы `random`, `bandit_greedy` и `bandit_epsilon`. Последние два требуют StationaryBanditEnv с неизменёнными действиями.

## Установка и ноутбук

Нужны Python 3.12 или новее и uv. Выполняйте команды из корня репозитория:

```powershell
uv sync --all-groups
uv run --all-groups pytest
uv run --all-groups ruff check .
uv run --all-groups python -m ipykernel install --user --name rl-fun --display-name "Python (RL Fun)"
uv run --all-groups jupyter lab
```

Откройте `notebooks/01_bandits/01_epsilon_greedy.ipynb`, выберите ядро `Python (RL Fun)` и выполните все ячейки по порядку. При `No module named 'rl_fun'` проверьте ядро и повторите `uv sync --all-groups`. Ноутбук использует установленный пакет.

## Запуск эксперимента

```powershell
uv run --all-groups python scripts/train.py configs/bandit-epsilon-greedy.json
uv run --all-groups python scripts/train.py configs/cartpole-random.json --metric train/cumulative_reward
uv run --all-groups python scripts/evaluate.py configs/cartpole-random.json --seed 11
```

Обучение запускает все seed конфигурации в `workers` процессах. `--metric` выбирает ключ итоговой статистики. Оценка использует случайную policy, а не обученную модель; число эпизодов и ограничение длины берутся из `evaluation`. Без `--seed` выбирается первый seed конфигурации. Оба режима работают без окна.

Визуальное воспроизведение CartPole отложено до отдельного согласования зависимости для рендеринга. Команда playback пока не реализована.

## Конфигурация версии 2

```json
{
  "name": "cartpole-random",
  "environment": {"id": "CartPole-v1", "kwargs": {}},
  "algorithm": {"id": "random", "kwargs": {}},
  "run": {"seeds": [11, 22], "total_steps": 100, "workers": 1, "output_root": "runs"},
  "evaluation": {"episodes": 2, "max_episode_steps": 100}
}
```

`environment.id` — зарегистрированный Gymnasium ID; `kwargs` — параметры конструктора. `algorithm.id` выбирается из локального реестра, его `kwargs` содержат параметры алгоритма (например, `epsilon` для `bandit_epsilon`). `run.total_steps` — бюджет шагов на seed. Seed уникальны и неотрицательны; `workers` — положительное число (по умолчанию 1), `output_root` — каталог относительно рабочей директории (по умолчанию `runs`). Необязательный `evaluation` по умолчанию задаёт 5 эпизодов и `max_episode_steps: null`; для среды без естественного конца задавайте ограничение. Старый плоский формат и неизвестные поля отклоняются. `render_mode` запрещён в `environment.kwargs`: его передают отдельно фабрике среды.

## Артефакты и сравнение

Обучение создаёт `runs/<name>/seed-<seed>-<идентификатор>/`:

- `config.json` — конфигурация;
- `metadata.json` — seed, версии Python и пакетов, система, Git commit и состояние рабочей копии;
- `metrics.jsonl` — события с полями `step` и `metrics`;
- `summary.json` — статус `success`, `failure` или `cancelled`, итоговые метрики и время;
- `error.txt` — traceback ошибки или прерывания.

Оценка создаёт каталог `eval-seed-<seed>-<идентификатор>` с config, metadata и эпизодными метриками; средняя награда выводится в консоль. Основные ключи: `train/steps`, `train/cumulative_reward`, для бандитов также `train/cumulative_regret`. Алгоритм random записывает `episode/reward`, `episode/length` и `train/episodes` по завершённым эпизодам. Последний эпизод, обрезанный бюджетом, включается в итоговую награду, но не записывается как завершённый.

Для сравнения укажите реальные каталоги запусков. Например, в PowerShell:

```powershell
$runDirs = (Get-ChildItem runs/bandit-epsilon-greedy -Directory | Select-Object -Last 2).FullName
uv run --all-groups python scripts/compare.py @runDirs --metric train/cumulative_reward --output-dir comparison
```

Сравнение пропускает неуспешные запуски, использует общие шаги выбранной метрики и записывает `comparison.csv` (шаг, среднее, стандартное отклонение) и `comparison.png`. При отсутствии общих событий возвращается ошибка.

## Гоночная среда

`RLFun/Racing-v0` — машинка с лучевыми сенсорами на замкнутой 2D-трассе. Наблюдение: расстояния лучей, продольная и боковая скорость, угловая скорость. Действие: `[руль, газ]` в диапазоне `[-1, 1]`; положительный руль — поворот налево, отрицательный газ — торможение. Награда за шаг — прирост прогресса вдоль трассы в долях круга; эпизод завершается аварией (касание границы), полным кругом или лимитом шагов.

Ручное вождение (нужно окно):

```powershell
uv run --all-groups python scripts/play.py
uv run --all-groups python scripts/play.py wavy
```

Стрелки или WASD — управление, `R` — перезапуск, `Esc` — выход. Трасса — имя встроенной (`oval`, `wavy`) или путь к JSON: `{"name": "...", "width": 10, "centerline": [[x, y], ...]}`; точки осевой линии идут по замкнутому контуру без самопересечений.

Через runtime среда запускается как любая другая, например `uv run --all-groups python scripts/train.py configs/racing-random.json`. Параметры среды (`track`, `ray_angles_deg` — по умолчанию `[-90, -30, 0, 30, 90]`, `ray_range`, `dt`, `max_steps`, `laps`, `dynamics`) задаются в `environment.kwargs`. Сейчас доступна кинематическая модель `kinematic` без заноса; другие модели динамики подключаются через `make_dynamics`.

## Эволюция машинок

Режим «как в видео про ИИ за рулём»: пачка машинок (`RacingFleet`) едет по трассе одновременно, врезавшиеся «умирают», лучшие дают потомство, и так поколение за поколением. Сеть и алгоритм отбора пишешь ты сам, библиотека даёт только флот, прогон поколения, окно и панель нейросети.

Откройте `notebooks/02_racing/01_evolution.ipynb` (ядро `Python (RL Fun)`, см. раздел про установку). В ноутбуке четыре заготовки с пояснениями — начальная популяция, сеть, отбор и мутация, а также необязательный просмотр сети — и параметры в первой ячейке:

- `SHOW_WINDOW` — показывать окно pygame с пачкой машинок; на машине без окна ставьте `False`, останется только график в ноутбуке;
- `SHOW_NETWORK` — рисовать справа сеть лидера (нужна четвёртая заготовка `inspect`);
- `POPULATION`, `GENERATIONS`, `HIDDEN`, `RAY_ANGLES`, `TRACK`, `SEED` — размер популяции, число поколений, скрытые слои, углы лучей, трасса и seed.

Для сравнения с готовым решением в библиотеке лежит эталон (`rl_fun.racing.reference`) с зафиксированным бенчмарком (`src/rl_fun/racing/benchmarks/oval.json`): ноутбук в разделе «Сравнение с эталоном» накладывает вашу кривую обучения на эталонную. Бенчмарк пересоздаётся командой `uv run --all-groups python scripts/make_reference.py`. Всё считается на CPU и не требует видеокарты.

## Структура проекта

- `src/rl_fun/` — среды, алгоритмы, policy, rollout, runtime и метрики;
- `src/rl_fun/racing/` — гоночная среда, флот машинок, отрисовка, эталонная эволюция;
- `scripts/` — обучение, оценка, сравнение;
- `scripts/play.py` — ручное вождение;
- `configs/` — воспроизводимые конфигурации;
- `notebooks/` — исполняемый учебный материал;
- `tests/` — проверки поведения;
- `runs/` — создаваемые артефакты.

[Как добавить новую среду](docs/environment-authoring.md): runtime отвечает за запуск и ресурсы, среда — за наблюдения, действия, награды и границы эпизода. PyTorch и Arena остаются будущими этапами.
