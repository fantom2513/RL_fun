// Garage: the best network of every run, a demo drive of the chosen one on any track, and a table
// of how it does on all tracks (does it generalise, or did it only learn its own track?).

import { h, icon } from '../core/dom.js';
import { number } from '../core/format.js';
import { TrackView } from '../track_view.js';
import '../ui/seg.js';

const STEP_SECONDS = 1 / 30;
const TERMINAL = new Set(['finished', 'stopped', 'error']);
const LEARNERS = { evolution: 'Эволюция', ppo: 'PPO', cem: 'CEM', a2c: 'A2C' };

const options = (...items) => JSON.stringify(items.map(([value, label]) => ({ value, label })));

function describe(result, laps) {
  if (!result) return 'Нет результата';
  if (result.finished) return `Проехал ${laps} кр. за ${number(result.extra.steps * STEP_SECONDS, 1)} с`;
  const reached = `${number(result.best, 2)} из ${laps} кр.`;
  return result.extra.crashed ? `Вылетел с трассы: ${reached}` : `Не успел за лимит: ${reached}`;
}

export function initGarage({ root, api, getCatalog, toast }) {
  let cars = [];
  let selected = null;
  let trackName = null;
  let laps = 1;
  let speed = 1;
  let current = null; // the demo being watched: {id, subscription}
  let view = null;
  let loaded = false;
  const tracks = new Map();
  const results = new Map(); // car id -> Map(track -> {result, status})
  let token = 0; // bumps whenever the watched drive changes, so a late answer is ignored

  const list = h('div', { class: 'garage-cars' });
  const note = h('p', { class: 'muted', hidden: true });
  const title = h('h2', { class: 'garage-title' });
  const sub = h('p', { class: 'muted garage-sub' });
  const trackSeg = h('lab-seg', { label: 'Трасса', value: '', options: '[]' });
  const lapsSeg = h('lab-seg', { label: 'Кругов', value: '1', options: options(['1', '1 круг'], ['2', '2 круга'], ['3', '3 круга']) });
  const speedSeg = h('lab-seg', { label: 'Скорость', value: '1', options: options(['1', '×1'], ['4', '×4'], ['max', 'макс']) });
  const restart = h('button', { type: 'button', class: 'btn btn-sm', onclick: () => watch() }, icon('rotate-ccw'), 'Заново');
  const checkAll = h('button', { type: 'button', class: 'btn btn-sm btn-primary', onclick: () => checkEverywhere() }, 'Проверить на всех трассах');
  const exportLink = h('a', { class: 'btn btn-sm btn-ghost', download: '' }, icon('download'), 'Скачать JSON');
  const canvas = h('canvas', { class: 'garage-canvas', 'aria-label': 'Проверочный заезд выбранной сети' });
  const stage = h('div', { class: 'garage-stage' }, canvas);
  const verdict = h('p', { class: 'garage-verdict', role: 'status' });
  const table = h('table', { class: 'table garage-results' });
  const panel = h('section', { class: 'garage-test', hidden: true },
    h('div', { class: 'garage-test-head' }, h('div', null, title, sub), exportLink),
    h('div', { class: 'garage-controls' }, trackSeg, lapsSeg, speedSeg, restart, checkAll),
    stage, verdict, table);
  root.replaceChildren(h('div', { class: 'garage' }, h('section', { class: 'garage-left' }, note, list), panel));

  const label = (car) => `${car.model.inputs} → ${[...car.model.hidden, car.model.outputs].join(' → ')}`;
  const record = (car) => {
    if (!results.has(car.id)) results.set(car.id, new Map());
    return results.get(car.id);
  };

  // ---- the list of cars -------------------------------------------------------------------

  function renderList() {
    if (!cars.length) {
      list.replaceChildren(h('div', { class: 'empty' },
        h('b', null, 'В гараже пока пусто'),
        h('span', null, 'Сеть сохраняется после первого поколения любого запуска. Запустите обучение в лаборатории, и машина появится здесь.')));
      panel.hidden = true;
      return;
    }
    list.replaceChildren(...cars.map((car) => {
      const lap = car.best_lap_steps ? `${number(car.best_lap_steps * STEP_SECONDS, 1)} с` : `${number(car.best ?? 0, 2)} кр.`;
      return h('button', {
        type: 'button', class: 'card garage-car', 'aria-pressed': String(car.id === selected?.id),
        onclick: () => select(car.id),
      },
      h('div', { class: 'garage-car-head' },
        h('b', null, car.name),
        h('span', { class: 'tag' }, LEARNERS[car.learner] ?? car.learner),
        car.archived ? h('span', { class: 'tag' }, 'архив') : null),
      h('div', { class: 'muted garage-car-meta' },
        `обучалась: ${car.track}, ${car.laps} кр. · лучший результат: ${lap} · сеть ${label(car)}`));
    }));
  }

  // ---- the demo drive ---------------------------------------------------------------------

  async function loadTrack(name) {
    if (!tracks.has(name)) tracks.set(name, await api.getTrack(name));
    return tracks.get(name);
  }

  // Runs one demo and resolves with its result; frames go to `onFrame` while it drives.
  function runDemo(car, track, { lapCount, pace, onFrame, onStart }) {
    return new Promise((resolve) => {
      api.createDemo({ run: car.id, track, laps: lapCount, speed: pace, max_steps: Math.min(5000, 1500 * lapCount) })
        .then(({ id }) => {
          onStart?.(id);
          let result = null;
          let subscription = null;
          const finish = (status, message) => {
            subscription?.close();
            api.deleteRun(id).catch(() => {});
            resolve({ result, status, message });
          };
          subscription = api.subscribe(id, {
            frame: (frame) => onFrame?.(frame),
            gen: (gen) => { result = gen; },
            status: (message) => { if (TERMINAL.has(message.status)) finish(message.status, message.message); },
            disconnected: () => finish('error', 'нет связи с сервером'),
          }, { maxRetries: 2 });
          if (current && current.id === id) current.subscription = subscription;
        })
        .catch((error) => resolve({ result: null, status: 'error', message: error.message }));
    });
  }

  function stopWatching() {
    token += 1;
    if (!current) return;
    current.subscription?.close();
    api.deleteRun(current.id).catch(() => {});
    current = null;
  }

  async function watch() {
    if (!selected || !trackName) return;
    stopWatching();
    const mine = token;
    const car = selected;
    const catalog = getCatalog();
    verdict.textContent = 'Едем…';
    let track;
    try {
      track = await loadTrack(trackName);
    } catch {
      verdict.textContent = 'Не удалось загрузить трассу.';
      return;
    }
    if (mine !== token) return;
    view.setStyle(catalog.style);
    view.setTrack(track, catalog.style);
    view.setLaps(laps);
    view.setLearner('demo');
    const outcome = await runDemo(car, trackName, {
      lapCount: laps,
      pace: speed,
      onFrame: (frame) => { if (mine === token) view.setFrame(frame); },
      onStart: (id) => { if (mine === token) current = { id, subscription: null }; },
    });
    if (mine !== token) return;
    current = null;
    if (outcome.status === 'error') {
      verdict.textContent = `Проверка не удалась: ${outcome.message ?? 'ошибка'}`;
      return;
    }
    if (outcome.status !== 'finished') return;
    record(car).set(trackName, { ...outcome, laps });
    verdict.textContent = describe(outcome.result, laps);
    renderTable();
  }

  async function checkEverywhere() {
    if (!selected) return;
    const car = selected;
    const names = getCatalog().tracks;
    checkAll.disabled = true;
    try {
      for (const name of names) {
        verdict.textContent = `Проверяем: ${name}…`;
        const outcome = await runDemo(car, name, { lapCount: laps, pace: 'max' });
        if (selected !== car) return;
        record(car).set(name, { ...outcome, laps });
        renderTable();
      }
      verdict.textContent = 'Проверка на всех трассах завершена.';
    } finally {
      checkAll.disabled = false;
    }
  }

  function renderTable() {
    const done = selected ? record(selected) : new Map();
    const names = getCatalog()?.tracks ?? [];
    table.replaceChildren(
      h('thead', null, h('tr', null,
        h('th', null, 'Трасса'), h('th', null, 'Результат'), h('th', null, 'Проехал'), h('th', null, 'Время, с'))),
      h('tbody', null, ...names.map((name) => {
        const entry = done.get(name);
        const result = entry?.result;
        const better = result?.finished ? h('span', { class: 'cell-best' }, number(result.extra.steps * STEP_SECONDS, 1)) : '—';
        return h('tr', {
          tabindex: '0', 'aria-selected': String(name === trackName),
          onclick: () => chooseTrack(name),
          onkeydown: (event) => { if (event.key === 'Enter') chooseTrack(name); },
        },
        h('td', null, name === selected?.track ? `${name} (учили здесь)` : name),
        h('td', null, entry ? (entry.status === 'finished' ? describe(result, entry.laps) : 'ошибка') : 'не проверяли'),
        h('td', null, result ? `${number(result.best, 2)} из ${entry.laps} кр.` : '—'),
        h('td', null, better));
      })));
  }

  function chooseTrack(name) {
    trackName = name;
    trackSeg.value = name;
    renderTable();
    watch();
  }

  // ---- selection --------------------------------------------------------------------------

  function select(id) {
    selected = cars.find((car) => car.id === id) ?? null;
    renderList();
    if (!selected) return;
    const catalog = getCatalog();
    const names = catalog?.tracks ?? [];
    panel.hidden = false;
    title.textContent = selected.name;
    sub.textContent = `${LEARNERS[selected.learner] ?? selected.learner} · сеть ${label(selected)} · ${selected.model.activation}`;
    exportLink.href = `/api/garage/${encodeURIComponent(selected.id)}/export`;
    exportLink.download = `${selected.name}.json`;
    trackSeg.setAttribute('options', options(...names.map((name) => [name, name])));
    laps = selected.laps;
    lapsSeg.value = String(Math.min(3, laps));
    trackName = names.includes(selected.track) ? selected.track : names[0];
    trackSeg.value = trackName;
    if (!view) view = new TrackView(canvas);
    renderTable();
    watch();
  }

  trackSeg.addEventListener('seg-change', (event) => chooseTrack(event.detail.value));
  lapsSeg.addEventListener('seg-change', (event) => {
    laps = Number(event.detail.value);
    watch();
  });
  speedSeg.addEventListener('seg-change', (event) => {
    speed = event.detail.value === 'max' ? 'max' : Number(event.detail.value);
    watch();
  });

  async function show() {
    note.hidden = false;
    note.textContent = 'Загружаем гараж…';
    try {
      cars = await api.getGarage();
    } catch {
      note.textContent = 'Не удалось получить список машин. Проверьте, что сервер запущен.';
      return;
    }
    note.hidden = true;
    loaded = true;
    const keep = selected && cars.find((car) => car.id === selected.id);
    renderList();
    if (cars.length) select(keep ? keep.id : cars[cars.length - 1].id);
  }

  return {
    show,
    // Leaving the screen ends a drive that nobody watches.
    hide() {
      stopWatching();
    },
    get loaded() {
      return loaded;
    },
  };
}
