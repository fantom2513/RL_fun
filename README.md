# RL Fun

RL Fun — учебная лаборатория обучения с подкреплением на Gymnasium. Один runtime запускает стационарного многорукого бандита и CartPole, сохраняет метрики каждого seed и сравнивает результаты. Доступны алгоритмы `random`, `bandit_greedy` и `bandit_epsilon`. Последние два требуют StationaryBanditEnv с неизменёнными действиями.

## Snake Lab

The local laboratory includes a Snake sandbox with manual play, DQN/Double DQN,
REINFORCE (Policy Gradient), A2C, PPO, experimental TD3 with an action adapter,
evolutionary neural-policy training, and human-left/agent-right comparison.
Launch it from the repository root:

```powershell
uv run --all-groups python scripts/lab.py
```

Open the **Змейка** navigation item (or `http://127.0.0.1:8765/#/snake`).
Choose **Я играю**, **Нейросеть**, or **Я vs нейросеть**. Click **Играть**;
arrows/WASD turn the human snake, Space pauses. Click the human board to restore
keyboard focus. The direction buttons support touch input. Losing browser focus
pauses playback. Both sides share the initial seed and tick rate; either side
can continue after the other loses. Changing the board, seed, mode or model
creates a fresh paused game.

The training panel runs a separate CPU process. **DQN** learns from replayed
experience using a target network and epsilon-greedy exploration. **Double DQN**
selects the next action with the online network and values it with the target
network. **REINFORCE** uses categorical actions and Monte Carlo episode returns.
**A2C** uses n-step advantages and a separate state-value network. **PPO** uses
GAE, fixed on-policy trajectories, clipped likelihood ratios and multiple
minibatch passes. Their exploration comes from sampled action probabilities;
entropy regularization is configurable, and replay/epsilon settings do not apply.
**TD3 · адаптер ходов** trains two Q critics and a deterministic actor with
delayed actor updates, target-action noise and soft target updates. Its actor
outputs three continuous scores in [-1, 1]; argmax maps them to the three Snake
turns. This piecewise-constant action adapter is experimental: original TD3 is
intended for continuous actions. **Эволюция**
evaluates a population on shared seeds, retains elites and mutates offspring.
The legacy observation uses 11 features and an `11 → 64 → 64 → 3` MLP. The
**Карта тела и еды** observation adds two board planes (ordered body segments
and food); on 16×16 it uses a `523 → 128 → 128 → 3` MLP. Scores count
food eaten; reward is `+10` for food, `-10` for collision and `-0.01` per ordinary
step. The score graph distinguishes training averages from greedy evaluation on
fixed seeds. A short run checks execution; substantial playing strength needs
more training and parameter tuning.

The objective is to occupy every board cell: 253 food items on 16×16. Training
and evaluation report occupancy and the fraction of completed games. A separate
configurable bonus rewards full-board victory. Evaluation respects the full
configured step budget; starvation, loops, collisions and budget exhaustion
are reported separately. Playback uses the best evaluated policy; continuation
restores the latest network, optimizer and replay. Legacy version-1 models remain
loadable, and continuation requires the same observation mode.

Disable **Анимация обучения** for full-speed simulation: no per-step snapshots,
frame transmission or animation sleeps. You can enable it again during a run.
The **Карта относительно головы** observation centers the head and rotates its
direction upward. Five planes encode the head, ordered body, food, occupied
fraction and walls; the 16×16 policy uses `4805 → 128 → 128 → 3`.
For RL algorithms, `curriculum_steps > 0` mixes ordinary starts with connected
long bodies. Half the resets retain the usual length of three; the maximum
synthetic starting length increases from 25% to 95% of the board over that
number of environment steps. These are training states, not demonstrations.
Playback and tuning/validation always start with three segments. Wins from
synthetic training starts do not establish full-game completion.
The optional `distance_reward` adds its value for each cell of Manhattan
distance gained toward the previous food, and subtracts it for moving away.
Food respawning does not change this progress calculation; collisions retain
their configured penalty. It defaults to zero. This reward shaping can favor
greedy food collection and must be tested for late-game self-trapping.

Research directions include [head-centered PPO with curriculum](https://elichen.github.io/snake/index.html),
[CNN-DQN with curriculum](https://web.stanford.edu/class/aa228/reports/2020/final16.pdf)
and [AlphaSnake's MCTS plus RL](https://arxiv.org/abs/2211.09622).
Their environments and training budgets differ. The current long-body reset
curriculum is an experimental hypothesis, not a reproduction of their reported
results. A successful configuration must be established by saved weights and
independent full-game evaluation in this environment.

The **Полное поле 16×16** preset selects Double DQN, board observations and
disabled animation. It is explicitly experimental until a trained model wins
at least 90 of 100 independent validation games at size 16 or larger.

Run a reproducible, bounded training and validation experiment:

```powershell
uv run python scripts/snake_check.py --train-seconds 60 --episodes 100000
uv run python scripts/snake_check.py --algorithm ppo --train-seconds 60 --episodes 100000
uv run python scripts/snake_check.py --train-seconds 300 --episodes 100000 --resume runs/lab/snake/<run-id>/model.pt
uv run python scripts/snake_check.py --validate-only --resume runs/lab/snake/<frozen-run-id>/model.pt
```

Training seeds, 20 tuning seeds and 100 validation seeds occupy separate ranges.
Training chunks evaluate only the 20 tuning seeds and write `evaluation.json`.
Freeze the selected model before running `--validate-only`: it performs no
training and writes the 100 independent outcomes in `validation.json`. Artifacts
include checkpoints, metrics, `record.json`, measured completion rate, elapsed
time, checkpoint hash and source version. The time budget bounds training;
greedy evaluation runs afterwards. Saved models appear in the lab model list.

Compare all six RL algorithms with a bounded, headless parameter search:

```powershell
uv run python scripts/snake_search.py --steps 100000 --seconds 180 --workers 2 --manifest runs/lab/snake/search-round1.json
```

The search records actual environment steps, updates and training time for each
experiment. The wall-clock cap may stop a slow algorithm before its step budget.
`--configs` accepts a JSON list of `{ "name": "experiment", "config": { ... },
"resume": "optional/model.pt" }` entries for follow-up parameter trials. The
manifest is updated after each finished run; weights remain in its run directory.
Ranking uses wins first, then mean occupancy on the 20 tuning games. These short
trials screen candidates and do not establish reliable full-board completion.
The **Лучший в поиске** preset loads the strongest saved tuning candidate and
its parameters. It shows measured wins and occupancy, remains experimental,
and ignores results whose checkpoint has changed since evaluation. Use
**Обновить** after a search to refresh the preset catalogue.
Under **Сохранённые модели → Лучшие предобученные**, choose the strongest
evaluated checkpoint for each algorithm. Each option shows mean occupancy and
victories on the tuning games; selection loads weights and training parameters.
**Обновить** refreshes both saved models and the preset catalogue.

Start, pause/resume, stop and save training from the panel. Models auto-save
after the first completed iteration, every 20 iterations, and on completion or
stop. Select a saved model under **Агент на поле** to load it into the game.
To resume training, select the model and check **Продолжить выбранную модель**;
the new episode/generation budget is added to the saved count. Continuation
requires the same algorithm, board size and (for evolution) population size.
DQN restores optimizer, target network and replay; evolution restores the
population and champion. On-policy algorithms restore the actor, critic,
optimizer, unfinished trajectory, PPO minibatch cursor and RNG state. TD3
restores both critics, target networks and continuous-score replay. Before training, agent play is explicitly labeled
**Необученная сеть**.

Artifacts live in `<runs-dir>/snake/<run-id>/`: `record.json`, `metrics.jsonl`
and the safely loaded tensor checkpoint `model.pt`. Use `--runs-dir` and `--port`
to launch an independent sandbox. Saved models and metric history remain
available after server restart. One Snake training job can run at a time.

For custom algorithms, use the environment directly from Python:

```python
import gymnasium as gym
import rl_fun.environments

env = gym.make("RLFun/Snake-v0", size=16)
observation, info = env.reset(seed=42)
# Relative actions: 0 = straight, 1 = left, 2 = right.
observation, reward, terminated, truncated, info = env.step(0)
env.close()
```

Learners and policies are in `src/rl_fun/snake/`; the independent environment
also exposes `snapshot()` for custom visualizations. Snake-specific API routes
are under `/api/snake`, including `catalog`, `state`, `models`, `train`,
`control`, `events`, and `games`.

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

Сеть машинки описывается данными — `ModelSpec` (`rl_fun.racing.model_spec`): входы, скрытые слои, выходы и активация скрытых слоёв (`tanh`, `relu` или `sigmoid`). Входы: лучи `ray:<угол>` (от −180 до 180 градусов), `speed`, `lateral_speed`, `yaw_rate`, `acceleration` (продольное ускорение) и `steering_angle` (последний угол руля). Выходы: `steer` (руль) обязателен; газ задаётся либо `throttle` (от −1 до 1, отрицательный — тормоз), либо парой `accelerate` и `brake` (от 0 до 1), плюс необязательный `boost` (буст, от 0 до 1). `RacingFleet(..., model=...)` собирает наблюдение в порядке `model.inputs`, а `reference.train(..., model=...)` строит сеть по этой спецификации. Награда может быть составной — `FitnessSpec({"progress": 1.0, "centering": 0.5})` — взвешенная сумма слагаемых `progress`, `lap_bonus`, `speed`, `survival`, `centering` и `smoothness` (отрицательный вес работает как штраф). Готовые профили в `PRESETS`: `racer` (прогресс, бонус круга, скорость), `careful` (прогресс, центрирование, плавность, выживание) и `balanced`. Без явных параметров всё работает как раньше; это основа будущего интерфейса для настройки сети и наград.

```python
from rl_fun.racing import reference
from rl_fun.racing.fitness import PRESETS
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.model_spec import ModelSpec

model = ModelSpec(
    inputs=("ray:-45", "ray:0", "ray:45", "speed", "acceleration"),
    hidden=(8,),
    outputs=("steer", "accelerate", "brake", "boost"),
    activation="relu",
)
fleet = RacingFleet(40, model=model)  # fleet.action_size == 4 выхода сети
run = reference.train("oval", generations=20, model=model, fitness=PRESETS["careful"])
```

## Лаборатория

Лаборатория — локальный веб-интерфейс для обучения машинок (эволюцией или PPO, см. ниже): несколько запусков параллельно (по отдельному процессу на запуск), пауза, скорость и остановка, «горячее» изменение части параметров. Запуск (нужен только браузер, окно pygame не требуется):

```powershell
uv run --all-groups python scripts/lab.py
```

Скрипт печатает адрес (по умолчанию `http://127.0.0.1:8765`) и открывает его в браузере; остановить лабораторию можно через `Ctrl+C`. Параметры: `--port` (порт, `0` — любой свободный), `--runs-dir` (каталог артефактов, по умолчанию `runs/lab`), `--no-browser` (не открывать браузер). Если порт занят, скрипт сообщит об этом и завершится с кодом 2 — выберите другой порт.

Состояние на этом этапе: интерфейс собирается по частям. Сейчас страница в браузере — заглушка (заголовок «Лаборатория»), а серверная часть уже готова: каталог параметров, создание и управление запусками и потоковая передача кадров и результатов поколений через HTTP API (`/api/catalog`, `/api/runs`, `/api/runs/<id>/command`, `/api/runs/<id>/stream`). Вкладки запусков, форма параметров, трасса с машинками, граф сети и кривые обучения появятся в следующих этапах.

Артефакты каждого запуска лежат в `runs/lab/<имя>-<время>/` (`config.json`, `history.jsonl`, `best_weights.json`) и обновляются после каждого поколения. Всё считается на CPU, видеокарта не нужна. Сервер слушает только `127.0.0.1` и недоступен из сети, поэтому аутентификации нет.

## Обучатели: эволюция и PPO

Запуск лаборатории выбирает «обучатель» полем `learner` конфигурации: `evolution` (по умолчанию) или `ppo`. Описание параметров каждого обучателя (подписи, границы, значения по умолчанию, подсказки, можно ли менять на ходу) отдаётся в `/api/catalog` в поле `learners`, интерфейс строит форму по нему. Ось X кривых — «итерация»: у эволюции это поколение, у PPO — один цикл «сбор опыта + обновление сети». Лучшая сеть любого обучателя сохраняется в одном формате `best_weights.json`.

**Что такое PPO.** Это обучение с подкреплением: одна сеть-водитель управляет всеми машинками сразу, немного случайно варьируя действия, и после каждой порции опыта сдвигает веса так, чтобы чаще делать то, что принесло награду больше ожидаемой. «Proximal» значит, что за одно обновление поведение меняется лишь чуть-чуть (ограничение clip), поэтому обучение идёт плавно и редко срывается.

Как это устроено: машинки собирают `rollout_steps` шагов опыта, разбившаяся или финишировавшая машинка сразу возвращается на старт, заезд длиннее `max_steps` обрывается. Награда на шаг собирается из той же смеси слагаемых, что и фитнес эволюции (`learners/rewards.py`), плюс штраф за аварию. Сеть задаётся той же формой модели (входы, слои, активация, выходы); вторая сеть такой же формы оценивает «насколько хороша ситуация».

Параметры (`ppo.*` в конфигурации; на ходу меняются отмеченные «горячие»):

| Параметр | По умолчанию | На ходу | Смысл |
|---|---|---|---|
| `learning_rate` | 3e-4 | да | размер шага обновления весов |
| `gamma` | 0.99 | да | насколько ценится будущая награда |
| `gae_lambda` | 0.95 | да | сглаживание оценки удачности действия |
| `clip_epsilon` | 0.2 | да | насколько сильно может измениться поведение за обновление |
| `entropy_coef` | 0.005 | да | награда за разнообразие действий («любопытство») |
| `value_coef` | 0.5 | да | вес обучения сети-оценщика |
| `crash_penalty` | 1.0 | да | штраф за вылет с трассы |
| `rollout_steps` | 128 | нет | шагов опыта на машинку перед обновлением |
| `epochs` | 4 | нет | проходов по собранному опыту |
| `minibatch_size` | 256 | нет | размер порции опыта для одного шага обучения |
| `initial_std` | 0.6 | нет | начальный разброс действий |
| `max_grad_norm` | 0.5 | нет | ограничение размера обновления |

Также на ходу меняется смесь наград (`fitness`), а `population` (число машинок) и `max_steps` задаются при создании. Горячие параметры присылаются командой `update` в виде `{"ppo": {"learning_rate": 0.001}}`; недопустимые или структурные отклоняются с замечанием.

PPO использует PyTorch, причём только CPU-сборку: в `pyproject.toml` для `torch` указан явный индекс `https://download.pytorch.org/whl/cpu`, поэтому на машинах без NVIDIA не скачиваются CUDA-колёса. `torch` загружается только при запуске PPO; эволюция работает без него.

Проверка качества обучения вручную (не тест): скрипт обучает PPO с умолчаниями и печатает кривые — завершённые эпизоды, средний и лучший прогресс в кругах, финиши, средняя награда эпизода, энтропия, σ, KL и время; в конце сеть без шума проезжает один заезд.

```powershell
uv run python scripts/ppo_check.py --track oval --minutes 2
uv run python scripts/ppo_check.py --track circuit --minutes 5 --every 10
```

Любой параметр PPO можно переопределить флагом (`--learning-rate 1e-3`, `--entropy-coef 0.01`, `--rollout-steps 256`, …), а также `--population`, `--max-steps`, `--seed`, `--iterations`.

## Структура проекта

### Continuous Snake training

Run an automatic 16×16 search with PPO, A2C, REINFORCE, DQN, Double DQN, and TD3:

```powershell
.venv\Scripts\python.exe -u scripts/snake_training_loop.py --seconds 1200 --steps 1000000 --target-occupancy 0.5
```

Each cycle changes parameters, trains without rendering, saves weights, and evaluates
20 ordinary games starting at length 3. Latest training states continue separately
for each algorithm; the best evaluated weights remain available for playback.
The loop continues until stopped; reaching 50% mean occupancy triggers one frozen,
independent 100-game validation and ends the search. Use `--target-occupancy 1`
to require full board completion instead. TD3 uses the experimental scores-to-action adapter.
This is a measured target, not a guarantee of a perfect learned policy.
Best evaluated weights remain available in the UI pretrained selector.

Status and completed results are recorded in `runs/lab/snake/training-loop.json`.
Only one loop can run at a time. Request a graceful stop with:

```powershell
New-Item -ItemType File -Force runs/lab/snake/training-loop.stop
```

The current cycle saves its checkpoint and evaluation before stopping. Remove the
stop file only when explicitly restarting. A completed independent validation
requires review before starting a new search, so its results are not repeatedly
used to tune parameters. `--max-cycles N` sets a finite cycle budget.

- `src/rl_fun/` — среды, алгоритмы, policy, rollout, runtime и метрики;
- `src/rl_fun/racing/` — гоночная среда, флот машинок, отрисовка, эталонная эволюция;
- `scripts/` — обучение, оценка, сравнение;
- `scripts/play.py` — ручное вождение;
- `src/rl_fun/lab/`, `scripts/lab.py` — лаборатория: сервер, запуски и веб-интерфейс;
- `configs/` — воспроизводимые конфигурации;
- `notebooks/` — исполняемый учебный материал;
- `tests/` — проверки поведения;
- `runs/` — создаваемые артефакты.

[Как добавить новую среду](docs/environment-authoring.md): runtime отвечает за запуск и ресурсы, среда — за наблюдения, действия, награды и границы эпизода. PyTorch (CPU) используется обучателем PPO лаборатории; Arena остаётся будущим этапом.

## Fourier epicycle animation

Two chains of rotating circles (x on top, y on the left) draw a contour (π, a heart, your own SVG\nor a silhouette image); the error of the current approximation is shown under the counter.
It is unrelated to the RL code and uses the optional `animation` dependency group:

```bash
uv run --group animation python scripts/fourier.py --source pi --quality low
uv run --group animation python scripts/fourier.py --source my-shape.svg --stages 3,10,50,300
```

Presets: `pi`, `heart`, `star` (see `src/rl_fun/fourier/assets/`). Raster images should
be dark shapes on a light background. Output goes to `runs/fourier/<name>.mp4`
(1080x1920, 60 fps with `--quality high`).
