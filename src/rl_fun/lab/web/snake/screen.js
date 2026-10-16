import * as api from './api.js';
import {SnakeBoard} from './board.js';
import {SnakeChart} from './chart.js';
import {keyDirection, TurnQueue} from './input.js';

const STATUS = {running:'Обучается', paused:'Пауза', finished:'Завершено', stopped:'Остановлено', error:'Ошибка'};
const END = {collision:'Столкновение', starvation:'Лимит без еды', loop:'Зациклилась · следующий кандидат', win:'Поле заполнено!'};
const ALGORITHM = {dqn:'DQN', double_dqn:'Double DQN', reinforce:'REINFORCE · Policy Gradient', a2c:'A2C', ppo:'PPO', td3:'TD3 · адаптер ходов', evolution:'Эволюция'};
const isDqn = algorithm => ['dqn','double_dqn'].includes(algorithm);
const FIELDS = {
  common: [['max_steps','Лимит ходов',10,100000,10],
    ['curriculum_steps','Ходов до полной сложности (0 — обычный старт)',0,10000000,1000]],
  rewards: [['food_reward','Награда за еду',0,1000,.5],['death_penalty','Штраф за смерть',0,1000,.5],
    ['win_reward','Бонус за полное поле',0,10000,1],['step_penalty','Штраф за каждый ход',0,10,.01],
    ['distance_reward','Награда за шаг к еде (0 — выкл.)',0,10,.01],
    ['idle_steps','Ходов без еды до штрафа (0 — выкл.)',0,10000,1],
    ['idle_penalty','Штраф за ход без роста',0,100,.01]],
  dqn: [['episodes','Эпизоды',1,100000,1],['learning_rate','Шаг обучения',.000001,.1,.0001],
    ['gamma','Важность будущего',0,1,.01],['batch_size','Размер выборки',4,512,1],
    ['epsilon_decay','Спад исследования, эп.',1,100000,1],
    ['evaluation_interval','Проверять каждые N эпизодов',1,10000,1],
    ['evaluation_episodes','Игр на проверке',1,100,1]],
  evolution: [['generations','Поколения',1,10000,1],['population','Популяция',4,128,1],
    ['elite_fraction','Доля элиты',.01,.5,.01],['mutation_scale','Сила мутации',.001,2,.01],
    ['evaluation_episodes','Игр на кандидата',1,100,1]],
  advanced: [['replay_capacity','Память опыта',8,100000,1],['warmup','Опыт до обучения',4,20000,1],
    ['train_interval','Обновлять каждые N ходов',1,100,1],['target_interval','Обновлять целевую сеть',1,10000,1],
    ['epsilon_start','Исследование в начале',0,1,.01],['epsilon_end','Исследование в конце',0,1,.01]],
};
FIELDS.double_dqn = FIELDS.dqn;
FIELDS.reinforce = [['episodes','Эпизоды',1,100000,1],
  ['learning_rate','Шаг обучения',.000001,.1,.0001],['gamma','Важность будущего',0,1,.001],
  ['entropy_coefficient','Поощрение разнообразия ходов',0,1,.001],
  ['evaluation_interval','Проверять каждые N эпизодов',1,10000,1],
  ['evaluation_episodes','Игр на проверке',1,100,1]];
FIELDS.a2c = [...FIELDS.reinforce,['rollout_steps','Ходов между обновлениями',4,10000,1],
  ['value_coefficient','Вес ошибки оценки состояния',0,10,.1]];
FIELDS.ppo = [...FIELDS.a2c,['batch_size','Размер выборки',4,512,1],
  ['gae_lambda','GAE λ',0,1,.01],['ppo_clip','Ограничение изменения политики',.01,.5,.01],
  ['ppo_epochs','Проходов по траектории',1,50,1]];
FIELDS.td3 = FIELDS.dqn.filter(([key]) => key !== 'epsilon_decay').concat([
  ['replay_capacity','Память опыта',8,100000,1],['warmup','Случайных ходов до обучения',4,20000,1],
  ['train_interval','Обновлять каждые N ходов',1,100,1],
  ['action_noise','Шум действий',0,1,.01],['target_noise','Шум целевых действий',0,1,.01],
  ['noise_clip','Ограничение целевого шума',0,1,.01],
  ['policy_delay','Обновлять политику каждые N обновлений критиков',1,100,1],
  ['tau','Скорость обновления целевых сетей',.000001,1,.001]]);

export function numericField(key, label, min, max, step, value) {
  if (step < 1) step = 'any';
  return `<label class="snake-field">${label}<input name="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${value}" required></label>`;
}

export function initSnake({root, toast}) {
  const stylesheet = document.createElement('link');
  stylesheet.rel = 'stylesheet'; stylesheet.href = '/static/snake/screen.css';
  document.head.append(stylesheet);
  root.innerHTML = `
    <div class="snake-workspace"><div class="snake-main">
      <section class="snake-toolbar" aria-label="Режим игры">
        <div class="snake-modes" role="group" aria-label="Кто играет">
          <button class="btn" data-mode="manual" aria-pressed="false">Я играю</button>
          <button class="btn" data-mode="agent" aria-pressed="false">Нейросеть</button>
          <button class="btn" data-mode="split" aria-pressed="true">Я vs нейросеть</button>
        </div>
        <div class="snake-world-controls">
          <label>Поле<input data-id="size" type="number" min="8" max="32" step="1" value="16"></label>
          <label>Seed<input data-id="seed" type="number" min="0" max="2147483647" step="1" value="42"></label>
          <label>Ходов / сек<select data-id="speed"><option>4</option><option selected>8</option><option>12</option><option>20</option><option>30</option><option>60</option><option>120</option><option>240</option></select></label>
        </div>
      </section>
      <section class="snake-arena" data-id="arena" aria-label="Игровые поля">
        <article class="snake-board-card" data-id="human-card">
          <header><span class="snake-player human"><i></i>Вы</span><span class="snake-score"><b data-id="human-score">0</b> яблок</span></header>
          <canvas data-id="human-board" tabindex="0" aria-label="Ваша змейка. Стрелки или WASD для управления"></canvas>
          <footer><span data-id="human-state">Готовы к игре</span><span data-id="human-steps">0 ходов</span></footer>
        </article>
        <article class="snake-board-card" data-id="agent-card">
          <header><span class="snake-player agent"><i></i>Нейросеть</span><span class="snake-score"><b data-id="agent-score">0</b> яблок</span></header>
          <canvas data-id="agent-board" tabindex="0" aria-label="Игра нейросети"></canvas>
          <footer><span data-id="agent-state">Необученная сеть</span><span data-id="agent-steps">0 ходов</span></footer>
        </article>
      </section>
      <div class="snake-playbar"><div class="snake-buttons">
        <button class="btn btn-primary" data-id="play">Играть</button><button class="btn" data-id="pause" disabled>Пауза</button><button class="btn" data-id="restart">Заново</button>
        </div><span class="snake-play-hint" data-id="play-hint">Стрелки / WASD · пробел — пауза</span>
        <div class="snake-pad" aria-label="Повороты"><button class="btn" data-dir="3" aria-label="Вверх">↑</button><button class="btn" data-dir="2" aria-label="Влево">←</button><button class="btn" data-dir="1" aria-label="Вниз">↓</button><button class="btn" data-dir="0" aria-label="Вправо">→</button></div>
      </div>
      <p class="snake-policy-note" data-id="policy-note"></p>
      <section class="snake-progress" aria-label="Прогресс обучения">
        <header><h2>Прогресс обучения</h2><select data-id="metric" aria-label="Показатель графика"><option value="occupancy">Заполнение поля, %</option><option value="win_rate">Победы, %</option><option value="score">Яблоки</option><option value="reward">Награда</option></select></header>
        <div class="snake-metrics"><div><span data-id="iteration-label">Эпизод</span><b data-id="iteration">—</b></div><div><span>Средний счёт</span><b data-id="mean">—</b></div><div><span>На проверке</span><b data-id="evaluation">—</b></div><div><span data-id="extra-label">Исследование</span><b data-id="extra">—</b></div></div>
        <p class="snake-note" data-id="completion">Цель: занять все клетки поля. Для 16×16 нужно 253 яблока.</p>
        <canvas data-id="chart" aria-label="График результатов обучения"></canvas>
        <div class="snake-chart-legend" data-id="legend"></div>
        <p class="snake-note" data-id="loss">Начните обучение: справа появится текущая модель.</p>
      </section>
    </div>
    <aside class="snake-training" aria-label="Настройки обучения">
      <header><h2>Обучить агента</h2><span class="snake-status" data-id="status" role="status">Не запущено</span></header>
      <form data-id="form">
        <label class="snake-field">Пресет<select data-id="preset"><option value="">Свои параметры</option></select></label>
        <p class="snake-note" data-id="preset-note">Проверенный пресет требует реальных побед на поле минимум 16×16.</p>
        <label class="snake-field">Алгоритм<select data-id="algorithm"><option value="dqn">DQN · обучение с подкреплением</option><option value="double_dqn">Double DQN · раздельный выбор и оценка</option><option value="reinforce">REINFORCE · Policy Gradient</option><option value="a2c">A2C · Actor-Critic</option><option value="ppo">PPO · ограниченные обновления политики</option><option value="td3">TD3 · непрерывные оценки → дискретный ход</option><option value="evolution">Эволюция · отбор и мутации</option></select></label>
        <label class="snake-field">Что видит сеть<select data-id="observation"><option value="local">11 локальных признаков · старые модели</option><option value="board">Карта тела и еды · полное поле</option><option value="egocentric">Карта относительно головы · тело, еда и стены</option></select></label>
        <p class="snake-note" data-id="algorithm-note"></p><div class="snake-fields" data-id="fields"></div>
        <label class="snake-resume"><input data-id="animate" type="checkbox">Анимация обучения — видеть каждый ход</label>
        <p class="snake-note">Скорость анимации задаётся в «Ходов / сек». Выключите её для быстрого обучения.</p>
        <details data-id="rewards"><summary>Награды и штрафы</summary><div class="snake-fields" data-id="reward-fields"></div></details>
        <details data-id="advanced"><summary>Дополнительные параметры DQN</summary><div class="snake-fields" data-id="advanced-fields"></div></details>
        <p class="snake-note" data-id="network-note">Сеть: 11 → 64 → 64 → 3.</p>
        <div class="snake-training-buttons"><button class="btn btn-primary" data-id="train" type="submit">Начать обучение</button><button class="btn" data-id="train-pause" type="button" disabled>Пауза</button><button class="btn btn-danger" data-id="stop" type="button" disabled>Стоп</button></div>
      </form>
      <div class="snake-saved"><h2>Сохранённые модели</h2>
        <label class="snake-field">Лучшие предобученные<select data-id="pretrained"><option value="">Выберите алгоритм и готовую модель</option></select></label>
        <p class="snake-note" data-id="pretrained-note">Лучшие оценённые веса каждого алгоритма. Рядом — заполнение поля и победы на проверке.</p>
        <label class="snake-field">Агент на поле<select data-id="model"><option value="__live__">Обучение вживую</option><option value="">Необученная сеть</option></select></label>
        <div class="snake-buttons"><button class="btn" data-id="save" disabled>Сохранить сейчас</button><button class="btn" data-id="refresh">Обновить</button></div>
        <label class="snake-resume"><input data-id="resume" type="checkbox">Продолжить выбранную модель</label>
        <p class="snake-note">Новые веса автоматически появляются на поле: каждые 20 эпизодов или каждое поколение эволюции. При продолжении бюджет добавляется к пройденному.</p>
        <div class="snake-model-list" data-id="model-list"></div>
      </div><p class="snake-error" data-id="error" role="alert" hidden></p>
    </aside></div>`;
  const el = id => root.querySelector(`[data-id="${id}"]`);
  const human = new SnakeBoard(el('human-board'), '--success');
  const agent = new SnakeBoard(el('agent-board'), '--info');
  const chart = new SnakeChart(el('chart')), turns = new TurnQueue();
  let visible = false, mode = 'split', game = null, playing = false;
  let timer = null, busy = false, building = false, buildRevision = 0;
  let defaults = null, stopSubscription = null, run = null, history = [], models = [];
  let modelSignature = '', controlsBusy = false;
  let trainingFrame = null;
  let presets = [], fieldValues = {};

  function fail(error) {
    el('error').hidden = false; el('error').textContent = error.message; toast(error.message, 'error');
  }
  async function safe(fn) { try { el('error').hidden = true; await fn(); } catch (error) { fail(error); } }
  function stopPlay() { playing = false; clearTimeout(timer); timer = null; turns.clear(); updatePlay(); }
  function allDone() {
    return game && (!game.human || game.human.done)
      && (!game.agent || (game.agent.done && game.agent.reason === 'win'));
  }
  function updatePlay() {
    el('play').disabled = playing || building || !game;
    el('play').textContent = game && !allDone() ? 'Играть' : 'Новая игра';
    el('pause').disabled = !playing;
  }
  function renderGame() {
    const live = el('model').value === '__live__' && el('animate').checked
      && trainingFrame?.run_id === run?.id ? trainingFrame : null;
    for (const [side, view] of [['human', human], ['agent', agent]]) {
      const frame = side === 'agent' && live && mode !== 'manual' ? live.frame : game?.[side];
      el(side + '-card').hidden = !frame;
      if (!frame) continue;
      view.setFrame(frame); el(side + '-score').textContent = frame.score;
      el(side + '-steps').textContent = `${frame.steps} ходов · ${((frame.occupancy ?? frame.snake.length / frame.size ** 2) * 100).toFixed(1)}% поля` + (side === 'agent' ? ` · игра ${game?.agent_episode ?? 1}` : '');
      el(side + '-state').textContent = frame.done ? END[frame.reason] ?? 'Игра завершена'
        : playing ? 'В игре' : 'Нажмите «Играть»';
      if (side === 'agent' && live) {
        const attempt = live.generation
          ? `поколение ${live.generation} · кандидат ${live.candidate}/${live.population} · игра ${live.episode}`
          : `эпизод ${live.episode}`;
        el('agent-steps').textContent = `${frame.steps} ходов · ${attempt}`;
        el('agent-state').textContent = run.status === 'paused' ? 'Обучение на паузе'
          : ['finished','stopped','error'].includes(run.status) ? STATUS[run.status]
          : live.ended ? END[frame.reason] ?? 'Лимит ходов · следующий эпизод'
          : 'Обучается — реальные ходы';
      }
    }
    el('arena').dataset.mode = mode;
    const selected = models.find(row => row.id === game?.model);
    const algorithm = selected?.algorithm ?? (game?.follow_training ? run?.config.algorithm : null);
    const policy = game?.trained ? `${ALGORITHM[algorithm] ?? 'Сеть'} · ${game.model.slice(0,8)} · итерация ${game.model_iteration}` : 'Необученная сеть';
    el('policy-note').textContent = mode === 'manual' ? 'Кликните по полю для управления. Обратный ход запрещён.'
      : live ? `Живое обучение ${ALGORITHM[run.config.algorithm]}. ${live.epsilon !== undefined ? `Исследование: ${(live.epsilon * 100).toFixed(0)}%. ` : ''}Справа — настоящая обучающая попытка. После смерти начинается следующая. Пауза обучения — в настройках.`
      : `${policy}. ${game?.follow_training ? 'Для показа обучающих ходов включите анимацию и начните обучение.' : 'Показ выбранной модели.'} После смерти агент начинает следующую игру автоматически.`;
    updatePlay();
  }
  async function newGame(start = false) {
    stopPlay(); building = true; updatePlay();
    const revision = ++buildRevision, previous = game?.id;
    try {
      const follow = el('model').value === '__live__';
      const size = Number(el('size').value);
      const model = follow ? (!run ? models.find(row => row.size === size)?.id : null) : el('model').value;
      const next = await api.request('games', {mode, size, seed:Number(el('seed').value), model:model || null, follow_training:follow});
      if (revision !== buildRevision) { await api.request('games/' + next.id, undefined, 'DELETE'); return; }
      game = next;
      if (previous) api.request('games/' + previous, undefined, 'DELETE').catch(() => {});
      turns.clear(); renderGame(); if (start && visible) play();
    } finally { if (revision === buildRevision) { building = false; updatePlay(); } }
  }
  function play() {
    if (!game || allDone()) { safe(() => newGame(true)); return; }
    playing = true; el(mode === 'agent' ? 'agent-board' : 'human-board').focus(); renderGame(); schedule();
  }
  function schedule() { clearTimeout(timer); if (playing && visible) timer = setTimeout(tick, 1000 / Number(el('speed').value)); }
  async function tick() {
    if (!playing || !visible || busy || building) return;
    busy = true; const id = game.id;
    try {
      const next = await api.request(`games/${id}/step`, {direction:turns.take(), follow_training:el('model').value === '__live__'});
      if (game?.id !== id) return;
      turns.ack(); game = next; if (allDone()) stopPlay(); renderGame();
    } catch (error) { if (game?.id === id) { stopPlay(); fail(error); } }
    finally { busy = false; schedule(); }
  }
  function queue(direction) { if (playing && game?.human && !game.human.done) turns.push(direction, game.human.direction); }
  function renderFields(values = defaults) {
    if (!defaults) return;
    if (values?.target) values = fieldValues;
    fieldValues = {...defaults, ...values};
    el('observation').value = fieldValues.observation_mode ?? 'local';
    const fieldMarkup = ([key,label,min,max,step]) => numericField(key,label,min,max,step,values?.[key] ?? defaults[key]);
    const algorithm = el('algorithm').value;
    const common = FIELDS.common.filter(([key]) => algorithm !== 'evolution' || key !== 'curriculum_steps');
    el('fields').innerHTML = [...FIELDS[algorithm], ...common].map(fieldMarkup).join('');
    el('reward-fields').innerHTML = FIELDS.rewards.map(fieldMarkup).join('');
    el('advanced-fields').innerHTML = isDqn(algorithm) ? FIELDS.advanced.map(fieldMarkup).join('') : '';
    el('advanced').hidden = !isDqn(algorithm);
    el('algorithm-note').textContent = {
      reinforce:'Policy Gradient: выбирает ходы по вероятностям и обновляет политику по наградам полного эпизода.',
      a2c:'Actor-Critic: политика выбирает ход, оценщик учит её по преимуществу действия. Обновления по свежим траекториям.',
      ppo:'Политика и оценщик учатся по свежим траекториям. Ограничение PPO сдерживает резкое изменение вероятностей ходов.',
      td3:'Экспериментальный адаптер TD3: три непрерывные оценки превращаются в ход по максимуму. Два критика, задержанные обновления политики и шум целевых действий. Обычный TD3 предназначен для непрерывных действий.',
      evolution:'Сети играют на одинаковых seed. Лучшие дают потомство с изменёнными весами.',
    }[algorithm] ?? 'Агент пробует действия, запоминает опыт и учится предсказывать будущую награду.';
    updateNetwork();
  }
  function updateNetwork() {
    const observation = el('observation').value, size = Number(el('size').value);
    el('network-note').textContent = observation === 'egocentric'
      ? `Сеть: ${5 * (2 * size - 1)**2} → 128 → 128 → 3. Голова в центре и смотрит вверх: тело, еда, длина и стены.`
      : observation === 'board' ? `Сеть: ${11 + 2 * size * size} → 128 → 128 → 3. Видит порядок сегментов тела и еду.` : 'Сеть: 11 → 64 → 64 → 3. Видит ближайшую опасность и направление еды.';
    if (['a2c','ppo'].includes(el('algorithm').value)) el('network-note').textContent += ' Отдельная сеть оценивает состояние.';
    if (el('algorithm').value === 'td3') el('network-note').textContent += ' Политика выдаёт оценки в диапазоне −1…1; два критика оценивают состояние и эти оценки.';
  }
  function updateTraining() {
    const active = run && !['finished','stopped','error'].includes(run.status);
    el('status').textContent = run ? STATUS[run.status] : 'Не запущено';
    el('status').dataset.status = run?.status ?? 'idle';
    el('train').disabled = Boolean(active) || !defaults || controlsBusy;
    el('train-pause').disabled = !active || controlsBusy;
    el('train-pause').textContent = run?.status === 'paused' ? 'Продолжить' : 'Пауза';
    el('stop').disabled = !active || controlsBusy;
    el('save').disabled = !run || (!active && !run.saved) || controlsBusy;
    const selectedModel = models.find(model => model.id === el('model').value)
      ?? (el('model').value === '__live__' && !run ? models.find(model => model.size === Number(el('size').value)) : null);
    const displayConfig = active ? run.config : selectedModel?.config ?? run?.config;
    const row = history.at(-1), evolution = (displayConfig?.algorithm ?? el('algorithm').value) === 'evolution';
    el('iteration-label').textContent = evolution ? 'Поколение' : 'Эпизод';
    el('iteration').textContent = active && trainingFrame?.run_id === run.id
      ? trainingFrame.generation ?? trainingFrame.episode : row?.iteration ?? '—';
    el('mean').textContent = row ? row.mean_score.toFixed(2) : '—';
    el('evaluation').textContent = row ? row.evaluation_score.toFixed(2) : '—';
    el('completion').textContent = row?.evaluation_occupancy !== undefined
      ? `На проверке: ${(row.evaluation_occupancy * 100).toFixed(1)}% поля · ${(row.evaluation_win_rate * 100).toFixed(1)}% побед. Цель — 100% заполнения.`
      : `Цель: занять все клетки поля. Для ${el('size').value}×${el('size').value} нужно ${Number(el('size').value) ** 2 - 3} яблок.`;
    const evidence = selectedModel?.verification;
    if (!active && evidence) {
      el('completion').textContent = `Финальная проверка ${selectedModel.size}×${selectedModel.size}: ${evidence.wins}/${evidence.episodes} побед · ${(evidence.occupancy * 100).toFixed(1)}% поля. ${evidence.status === 'verified' ? 'Пресет подтверждён.' : 'Прохождение пока не подтверждено.'}`;
    }
    const dqn = isDqn(displayConfig?.algorithm ?? el('algorithm').value);
    const td3 = (displayConfig?.algorithm ?? el('algorithm').value) === 'td3';
    el('extra-label').textContent = evolution ? 'Лучшая награда' : dqn ? 'Исследование' : td3 ? 'Шум действий' : 'Энтропия политики';
    el('extra').textContent = row ? evolution ? row.best_reward.toFixed(1)
      : dqn ? (row.epsilon * 100).toFixed(0) + '%' : td3 ? row.action_noise?.toFixed(3) ?? '—' : row.entropy?.toFixed(3) ?? '—' : '—';
    el('loss').textContent = active && evolution && trainingFrame?.run_id === run.id
      ? `Поколение ${trainingFrame.generation}: кандидат ${trainingFrame.candidate} из ${trainingFrame.population}, игра ${trainingFrame.episode} из ${run.config.evaluation_episodes}. График обновится после оценки популяции.`
      : row && !evolution && typeof row.loss === 'number'
      ? `${dqn ? 'Ошибка предсказания' : 'Функция потерь'}: ${row.loss.toFixed(4)} · поле ${displayConfig.size} × ${displayConfig.size}`
      : 'Графики показывают реальные результаты. Сила игры зависит от времени обучения.';
    chart.update(history, el('metric').value);
    el('legend').innerHTML = ['occupancy','win_rate'].includes(el('metric').value)
      ? '<span class="human">Обучение</span><span class="agent">Проверка без исследования</span>'
      : el('metric').value === 'score'
      ? '<span class="human">Средний счёт</span><span class="agent">Проверка на фиксированных seed</span>'
      : '<span class="reward">Награда / лучшая</span><span class="agent">Средняя по популяции (эволюция)</span>';
  }
  async function refreshCatalog() {
    const catalog = await api.request('catalog');
    const firstLoad = !defaults;
    defaults = catalog.defaults; presets = catalog.presets ?? [];
    const algorithm = el('algorithm').value;
    el('algorithm').replaceChildren(...catalog.algorithms.map(value => new Option(ALGORITHM[value] ?? value,value)));
    el('algorithm').value = catalog.algorithms.includes(algorithm) ? algorithm : defaults.algorithm;
    const selected = el('preset').value, trained = el('pretrained').value;
    el('preset').replaceChildren(new Option('Свои параметры',''));
    el('pretrained').replaceChildren(new Option('Выберите алгоритм и готовую модель',''));
    for (const preset of presets) {
      el('preset').add(new Option(preset.label,preset.id));
      if (preset.id.startsWith('best_')) el('pretrained').add(new Option(preset.label,preset.id));
    }
    el('preset').value = presets.some(row => row.id === selected) ? selected : '';
    el('pretrained').value = presets.some(row => row.id === trained) ? trained : '';
    el('pretrained').disabled = !presets.some(row => row.id.startsWith('best_'));
    if (firstLoad) renderFields();
  }
  async function refreshModels() {
    models = await api.request('models'); const selected = el('model').value;
    el('model').replaceChildren(new Option('Обучение вживую','__live__'),new Option('Необученная сеть',''));
    for (const model of models) el('model').add(new Option(`${ALGORITHM[model.algorithm]} · ${model.iteration} ит. · ${model.size}×${model.size} · ${model.verification?.status === 'verified' ? `проверено: ${model.verification.wins}/${model.verification.episodes} побед · ` : ''}${model.id.slice(0,8)}`,model.id));
    if (selected === '__live__' || selected === '' || models.some(row => row.id === selected)) el('model').value = selected;
    el('model-list').replaceChildren();
    for (const model of models.slice(0,5)) {
      const button = document.createElement('button'); button.className = 'snake-model-row';
      button.textContent = `${ALGORITHM[model.algorithm]} · ${model.id.slice(0,8)} · ${model.iteration} ит.`;
      button.addEventListener('click', () => safe(async () => { el('model').value = model.id; await chooseModel(); }));
      el('model-list').append(button);
    }
  }
  async function chooseModel() {
    const model = models.find(row => row.id === el('model').value);
    if (model) {
      el('size').value = model.size; el('algorithm').value = model.algorithm; renderFields(model.config);
      if (!run || ['finished','stopped','error'].includes(run.status)) {
        history = model.history; updateTraining();
      }
    }
    await newGame(false);
  }
  async function command(name) {
    controlsBusy = true; updateTraining();
    try { await api.request('control',{command:name}); }
    finally { controlsBusy = false; updateTraining(); }
  }
  function onState(state) {
    const previousRun = run?.id;
    run = state.run;
    if (run && run.id !== previousRun && !['finished','stopped','error'].includes(run.status)) {
      el('algorithm').value = run.config.algorithm;
      renderFields(run.config);
    }
    if (run && state.visualization) {
      el('animate').checked = state.visualization.enabled;
      const fps = String(state.visualization.fps);
      if (![...el('speed').options].some(option => option.value === fps)) el('speed').add(new Option(fps,fps));
      el('speed').value = fps;
    }
    onFrame(state.frame);
    const selected = models.find(model => model.id === el('model').value)
      ?? (el('model').value === '__live__' && !run ? models.find(model => model.size === Number(el('size').value)) : null);
    history = selected && (!run || ['finished','stopped','error'].includes(run.status))
      ? selected.history : run?.history ?? [];
    updateTraining();
    const signature = `${run?.id}:${run?.saved_iteration}:${run?.status}`;
    if (signature !== modelSignature) { modelSignature = signature; refreshModels().catch(fail); }
    if (run?.error) { el('error').hidden = false; el('error').textContent = run.error; }
  }
  function onFrame(frame) {
    if (frame && frame.run_id !== run?.id) return;
    if (frame && trainingFrame?.run_id === frame.run_id && frame.seq < trainingFrame.seq) return;
    trainingFrame = frame;
    if (visible) {
      renderGame();
      if (run?.config.algorithm === 'evolution' && frame) {
        el('iteration').textContent = frame.generation;
        el('loss').textContent = `Поколение ${frame.generation}: кандидат ${frame.candidate} из ${frame.population}, игра ${frame.episode} из ${run.config.evaluation_episodes}. График обновится после оценки популяции.`;
      }
    }
  }
  async function visualization() {
    await api.request('visualization', {enabled:el('animate').checked, fps:Number(el('speed').value)});
    renderGame();
  }
  el('play').addEventListener('click',play);
  el('pause').addEventListener('click',() => { stopPlay(); renderGame(); });
  el('restart').addEventListener('click',() => safe(() => newGame(true)));
  for (const button of root.querySelectorAll('[data-mode]')) button.addEventListener('click',() => safe(async () => {
    mode = button.dataset.mode;
    for (const candidate of root.querySelectorAll('[data-mode]')) candidate.setAttribute('aria-pressed',String(candidate === button));
    el('play-hint').hidden = mode === 'agent'; await newGame(false);
  }));
  for (const button of root.querySelectorAll('[data-dir]')) button.addEventListener('click',() => { queue(Number(button.dataset.dir)); el('human-board').focus(); });
  for (const id of ['size','seed']) el(id).addEventListener('change',() => safe(() => { updateNetwork(); return newGame(false); }));
  el('algorithm').addEventListener('change',renderFields); el('metric').addEventListener('change',updateTraining);
  el('observation').addEventListener('change',updateNetwork);
  async function applyPreset(id) {
    const preset = presets.find(row => row.id === id);
    if (!preset) return;
    el('size').value = preset.config.size;
    el('seed').value = preset.config.seed;
    el('algorithm').value = preset.config.algorithm;
    el('animate').checked = preset.visualization.enabled;
    el('speed').value = String(preset.visualization.fps);
    el('resume').checked = Boolean(preset.model);
    if (preset.model) el('model').value = preset.model;
    el('preset').value = preset.id;
    el('pretrained').value = preset.id.startsWith('best_') ? preset.id : '';
    el('pretrained-note').textContent = preset.model ? preset.description : 'Выберите готовую модель по алгоритму.';
    el('preset-note').textContent = preset.description;
    renderFields(preset.config); await visualization();
    if (preset.model) await chooseModel(); else await newGame(false);
  }
  el('preset').addEventListener('change',() => safe(() => applyPreset(el('preset').value)));
  el('pretrained').addEventListener('change',() => safe(() => applyPreset(el('pretrained').value)));
  el('model').addEventListener('change',() => safe(chooseModel));
  el('refresh').addEventListener('click',() => safe(async () => { await refreshCatalog(); await refreshModels(); }));
  el('animate').addEventListener('change',() => safe(visualization));
  el('speed').addEventListener('change',() => safe(visualization));
  el('train-pause').addEventListener('click',() => safe(() => command(run.status === 'paused' ? 'resume' : 'pause')));
  el('stop').addEventListener('click',() => safe(() => command('stop')));
  el('save').addEventListener('click',() => safe(() => command('save')));
  el('form').addEventListener('submit',event => {
    event.preventDefault(); safe(async () => {
      const config = {...fieldValues, algorithm:el('algorithm').value,size:Number(el('size').value),seed:Number(el('seed').value),observation_mode:el('observation').value};
      const common = FIELDS.common.filter(([key]) => config.algorithm !== 'evolution' || key !== 'curriculum_steps');
      const groups = [FIELDS[config.algorithm],common,FIELDS.rewards,...(isDqn(config.algorithm) ? [FIELDS.advanced] : [])];
      for (const [key] of groups.flat()) config[key] = Number(el('form').elements.namedItem(key).value);
      if (config.algorithm === 'evolution') config.curriculum_steps = 0;
      const resume = el('resume').checked ? el('model').value : null;
      if (el('resume').checked && (!resume || resume === '__live__')) throw new Error('Выберите сохранённую модель для продолжения');
      controlsBusy = true; updateTraining();
      try {
        await api.request('train',{config,resume, visualization:{enabled:el('animate').checked, fps:Number(el('speed').value)}});
        onState(await api.request('state'));
        el('model').value = '__live__';
        el('resume').checked = false;
        if (!game) await newGame(false);
        renderGame(); el('agent-board').focus();
      }
      finally { controlsBusy = false; updateTraining(); }
    });
  });
  root.addEventListener('keydown',event => {
    const direction = keyDirection(event);
    if (direction !== null && mode !== 'agent') { event.preventDefault(); queue(direction); }
    if (event.code === 'Space' && event.target.tagName === 'CANVAS') {
      event.preventDefault(); if (playing) { stopPlay(); renderGame(); } else play();
    }
  });
  el('human-board').addEventListener('blur',event => {
    if (playing && !event.relatedTarget?.closest('[data-dir]')) { stopPlay(); renderGame(); }
  });
  window.addEventListener('blur',() => { if (playing) { stopPlay(); renderGame(); } });
  document.addEventListener('visibilitychange',() => { if (document.hidden) { stopPlay(); renderGame(); } });
  updateTraining();
  updatePlay();
  return {
    async show() {
      visible = true;
      await safe(async () => {
        await refreshCatalog();
        await refreshModels(); onState(await api.request('state'));
        if (!game) await newGame(false); else renderGame();
        if (visible && !stopSubscription) stopSubscription = api.subscribe(onState,() => { el('status').textContent = 'Связь восстанавливается…'; },onFrame);
      });
    },
    hide() { visible = false; stopPlay(); stopSubscription?.(); stopSubscription = null; },
  };
}
