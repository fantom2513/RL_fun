import * as api from './api.js';
import {SnakeBoard} from './board.js';
import {SnakeChart} from './chart.js';
import {keyDirection, TurnQueue} from './input.js';

const STATUS = {running:'Обучается', paused:'Пауза', finished:'Завершено', stopped:'Остановлено', error:'Ошибка'};
const END = {collision:'Столкновение', starvation:'Лимит без еды', win:'Поле заполнено!'};
const FIELDS = {
  common: [['max_steps','Лимит ходов',10,10000,10]],
  dqn: [['episodes','Эпизоды',1,100000,1],['learning_rate','Шаг обучения',.000001,.1,.0001],
    ['gamma','Важность будущего',0,1,.01],['batch_size','Размер выборки',4,512,1],
    ['epsilon_decay','Спад исследования, эп.',1,100000,1]],
  evolution: [['generations','Поколения',1,10000,1],['population','Популяция',4,128,1],
    ['elite_fraction','Доля элиты',.01,.5,.01],['mutation_scale','Сила мутации',.001,2,.01],
    ['evaluation_episodes','Игр на кандидата',1,10,1]],
  advanced: [['replay_capacity','Память опыта',8,100000,1],['warmup','Опыт до обучения',4,20000,1],
    ['train_interval','Обновлять каждые N ходов',1,100,1],['target_interval','Обновлять целевую сеть',1,10000,1],
    ['epsilon_start','Исследование в начале',0,1,.01],['epsilon_end','Исследование в конце',0,1,.01]],
};

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
          <label>Ходов / сек<select data-id="speed"><option>4</option><option selected>8</option><option>12</option><option>20</option></select></label>
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
        <header><h2>Прогресс обучения</h2><select data-id="metric" aria-label="Показатель графика"><option value="score">Яблоки</option><option value="reward">Награда</option></select></header>
        <div class="snake-metrics"><div><span data-id="iteration-label">Эпизод</span><b data-id="iteration">—</b></div><div><span>Средний счёт</span><b data-id="mean">—</b></div><div><span>На проверке</span><b data-id="evaluation">—</b></div><div><span data-id="extra-label">Исследование</span><b data-id="extra">—</b></div></div>
        <canvas data-id="chart" aria-label="График результатов обучения"></canvas>
        <div class="snake-chart-legend" data-id="legend"></div>
        <p class="snake-note" data-id="loss">Обучение идёт отдельно от игры.</p>
      </section>
    </div>
    <aside class="snake-training" aria-label="Настройки обучения">
      <header><h2>Обучить агента</h2><span class="snake-status" data-id="status" role="status">Не запущено</span></header>
      <form data-id="form">
        <label class="snake-field">Алгоритм<select data-id="algorithm"><option value="dqn">DQN · обучение с подкреплением</option><option value="evolution">Эволюция · отбор и мутации</option></select></label>
        <p class="snake-note" data-id="algorithm-note"></p><div class="snake-fields" data-id="fields"></div>
        <details data-id="advanced"><summary>Дополнительные параметры DQN</summary><div class="snake-fields" data-id="advanced-fields"></div></details>
        <p class="snake-note">Сеть: 11 → 64 → 64 → 3. Награда: +10 за еду, −10 за столкновение.</p>
        <div class="snake-training-buttons"><button class="btn btn-primary" data-id="train" type="submit">Начать обучение</button><button class="btn" data-id="train-pause" type="button" disabled>Пауза</button><button class="btn btn-danger" data-id="stop" type="button" disabled>Стоп</button></div>
      </form>
      <div class="snake-saved"><h2>Сохранённые модели</h2>
        <label class="snake-field">Агент на поле<select data-id="model"><option value="">Необученная сеть</option></select></label>
        <div class="snake-buttons"><button class="btn" data-id="save" disabled>Сохранить сейчас</button><button class="btn" data-id="refresh">Обновить</button></div>
        <label class="snake-resume"><input data-id="resume" type="checkbox">Продолжить выбранную модель</label>
        <p class="snake-note">При продолжении бюджет добавляется к пройденному. Автосохранение: первая итерация, каждые 20 и при завершении.</p>
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

  function fail(error) {
    el('error').hidden = false; el('error').textContent = error.message; toast(error.message, 'error');
  }
  async function safe(fn) { try { el('error').hidden = true; await fn(); } catch (error) { fail(error); } }
  function stopPlay() { playing = false; clearTimeout(timer); timer = null; turns.clear(); updatePlay(); }
  function allDone() { return game && [game.human, game.agent].filter(Boolean).every(frame => frame.done); }
  function updatePlay() {
    el('play').disabled = playing || building || !game;
    el('play').textContent = game && !allDone() ? 'Играть' : 'Новая игра';
    el('pause').disabled = !playing;
  }
  function renderGame() {
    for (const [side, view] of [['human', human], ['agent', agent]]) {
      const frame = game?.[side]; el(side + '-card').hidden = !frame;
      if (!frame) continue;
      view.setFrame(frame); el(side + '-score').textContent = frame.score;
      el(side + '-steps').textContent = frame.steps + ' ходов';
      el(side + '-state').textContent = frame.done ? END[frame.reason] ?? 'Игра завершена'
        : playing ? 'В игре' : 'Нажмите «Играть»';
    }
    el('arena').dataset.mode = mode;
    const selected = models.find(row => row.id === game?.model);
    const policy = game?.trained ? `${selected?.algorithm === 'evolution' ? 'Эволюция' : 'DQN'} · ${game.model.slice(0,8)}` : 'Необученная сеть';
    el('policy-note').textContent = mode === 'manual' ? 'Кликните по полю для управления. Обратный ход запрещён.'
      : `${policy}. ${mode === 'split' ? 'Общие seed и скорость. После проигрыша одного второй продолжает.' : 'Агент выбирает действия без исследования.'}`;
    updatePlay();
  }
  async function newGame(start = false) {
    stopPlay(); building = true; updatePlay();
    const revision = ++buildRevision, previous = game?.id;
    try {
      const next = await api.request('games', {mode, size:Number(el('size').value), seed:Number(el('seed').value), model:el('model').value || null});
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
      const next = await api.request(`games/${id}/step`, {direction:turns.take()});
      if (game?.id !== id) return;
      turns.ack(); game = next; if (allDone()) stopPlay(); renderGame();
    } catch (error) { if (game?.id === id) { stopPlay(); fail(error); } }
    finally { busy = false; schedule(); }
  }
  function queue(direction) { if (playing && game?.human && !game.human.done) turns.push(direction, game.human.direction); }
  function renderFields(values = defaults) {
    if (!defaults) return;
    const fieldMarkup = ([key,label,min,max,step]) => numericField(key,label,min,max,step,values?.[key] ?? defaults[key]);
    const algorithm = el('algorithm').value;
    el('fields').innerHTML = [...FIELDS[algorithm], ...FIELDS.common].map(fieldMarkup).join('');
    el('advanced-fields').innerHTML = FIELDS.advanced.map(fieldMarkup).join('');
    el('advanced').hidden = algorithm !== 'dqn';
    el('algorithm-note').textContent = algorithm === 'dqn'
      ? 'Агент пробует действия, запоминает опыт и учится предсказывать будущую награду.'
      : 'Сети играют на одинаковых seed. Лучшие дают потомство с изменёнными весами.';
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
    const displayConfig = active ? run.config : models.find(model => model.id === el('model').value)?.config ?? run?.config;
    const row = history.at(-1), evolution = (displayConfig?.algorithm ?? el('algorithm').value) === 'evolution';
    el('iteration-label').textContent = evolution ? 'Поколение' : 'Эпизод';
    el('iteration').textContent = row?.iteration ?? '—';
    el('mean').textContent = row ? row.mean_score.toFixed(2) : '—';
    el('evaluation').textContent = row ? row.evaluation_score.toFixed(2) : '—';
    el('extra-label').textContent = evolution ? 'Лучшая награда' : 'Исследование';
    el('extra').textContent = row ? evolution ? row.best_reward.toFixed(1) : (row.epsilon * 100).toFixed(0) + '%' : '—';
    el('loss').textContent = row && !evolution && row.loss !== null
      ? `Ошибка предсказания: ${row.loss.toFixed(4)} · поле ${displayConfig.size} × ${displayConfig.size}`
      : 'Графики показывают реальные результаты. Сила игры зависит от времени обучения.';
    chart.update(history, el('metric').value);
    el('legend').innerHTML = el('metric').value === 'score'
      ? '<span class="human">Средний счёт</span><span class="agent">Проверка на фиксированных seed</span>'
      : '<span class="reward">Награда / лучшая</span><span class="agent">Средняя по популяции (эволюция)</span>';
  }
  async function refreshModels() {
    models = await api.request('models'); const selected = el('model').value;
    el('model').replaceChildren(new Option('Необученная сеть',''));
    for (const model of models) el('model').add(new Option(`${model.algorithm === 'dqn' ? 'DQN' : 'Эволюция'} · ${model.iteration} ит. · ${model.size}×${model.size} · ${model.id.slice(0,8)}`,model.id));
    if (models.some(row => row.id === selected)) el('model').value = selected;
    el('model-list').replaceChildren();
    for (const model of models.slice(0,5)) {
      const button = document.createElement('button'); button.className = 'snake-model-row';
      button.textContent = `${model.algorithm === 'dqn' ? 'DQN' : 'Эволюция'} · ${model.id.slice(0,8)} · ${model.iteration} ит.`;
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
    run = state.run;
    const selected = models.find(model => model.id === el('model').value);
    history = selected && (!run || ['finished','stopped','error'].includes(run.status))
      ? selected.history : run?.history ?? [];
    updateTraining();
    const signature = `${run?.id}:${run?.saved_iteration}:${run?.status}`;
    if (signature !== modelSignature) { modelSignature = signature; refreshModels().catch(fail); }
    if (run?.error) { el('error').hidden = false; el('error').textContent = run.error; }
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
  for (const id of ['size','seed']) el(id).addEventListener('change',() => safe(() => newGame(false)));
  el('algorithm').addEventListener('change',renderFields); el('metric').addEventListener('change',updateTraining);
  el('model').addEventListener('change',() => safe(chooseModel));
  el('refresh').addEventListener('click',() => safe(refreshModels));
  el('train-pause').addEventListener('click',() => safe(() => command(run.status === 'paused' ? 'resume' : 'pause')));
  el('stop').addEventListener('click',() => safe(() => command('stop')));
  el('save').addEventListener('click',() => safe(() => command('save')));
  el('form').addEventListener('submit',event => {
    event.preventDefault(); safe(async () => {
      const config = {algorithm:el('algorithm').value,size:Number(el('size').value),seed:Number(el('seed').value)};
      const groups = [FIELDS[config.algorithm],FIELDS.common,...(config.algorithm === 'dqn' ? [FIELDS.advanced] : [])];
      for (const [key] of groups.flat()) config[key] = Number(el('form').elements.namedItem(key).value);
      const resume = el('resume').checked ? el('model').value : null;
      if (el('resume').checked && !resume) throw new Error('Выберите модель для продолжения');
      controlsBusy = true; updateTraining();
      try { await api.request('train',{config,resume}); onState(await api.request('state')); }
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
        if (!defaults) { defaults = (await api.request('catalog')).defaults; renderFields(); }
        await refreshModels(); onState(await api.request('state'));
        if (!game) await newGame(false); else renderGame();
        if (visible && !stopSubscription) stopSubscription = api.subscribe(onState,() => { el('status').textContent = 'Связь восстанавливается…'; });
      });
    },
    hide() { visible = false; stopPlay(); stopSubscription?.(); stopSubscription = null; },
  };
}
