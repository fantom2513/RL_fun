// Lab shell: run tabs, run controls, the track view, the parameters form and the learning curves.
// The network panel is a placeholder for now.

import * as api from './api.js';
import { Chart, METRICS } from './charts.js';
import { ParamsForm } from './params_form.js';
import { TrackView } from './track_view.js';

const RUN_COLORS = ['#3b82c4', '#e08a3c', '#2fa59a', '#9a6fd0', '#d05d8f', '#8fa83a', '#5ab4e0', '#a8795a'];
const STATUS_LABELS = {
  running: 'идёт',
  paused: 'пауза',
  finished: 'завершён',
  stopped: 'остановлен',
  error: 'ошибка',
};
const TERMINAL = new Set(['finished', 'stopped', 'error']);
const ZOOM_STEP = 1.25;

const $ = (id) => document.getElementById(id);
const el = {
  tabs: $('tabs'),
  newRun: $('new-run'),
  emptyNew: $('empty-new'),
  pause: $('pause'),
  speed: $('speed'),
  stop: $('stop'),
  remove: $('delete'),
  canvas: $('track'),
  stage: document.querySelector('.stage'),
  empty: $('stage-empty'),
  loading: $('stage-loading'),
  banner: $('stage-banner'),
  bannerText: $('banner-text'),
  bannerRetry: $('banner-retry'),
  cameraMode: $('camera-mode'),
  zoomIn: $('zoom-in'),
  zoomOut: $('zoom-out'),
  hint: $('stage-hint'),
  toasts: $('toasts'),
  paramsPanel: document.querySelector('.params'),
  paramsBody: $('params-body'),
  chart: $('chart'),
  metrics: $('metrics'),
  legend: $('legend'),
};

const state = {
  catalog: null,
  runs: new Map(),
  activeId: null,
  tracks: new Map(),
  colorIndex: 0,
  creating: false,
};

let view = null;
let form = null;
let chart = null;

// ---- notifications ------------------------------------------------------------------------

function toast(text, kind = 'info') {
  const node = document.createElement('div');
  node.className = `toast${kind === 'error' ? ' toast-error' : ''}`;
  node.textContent = text;
  el.toasts.append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 8000 : 5000);
  while (el.toasts.children.length > 4) el.toasts.firstChild.remove();
}

api.setErrorHandler((message) => toast(message, 'error'));

// ---- runs ---------------------------------------------------------------------------------

const activeRun = () => state.runs.get(state.activeId) ?? null;

function addRun(row) {
  const run = {
    id: row.id,
    name: row.name,
    status: row.status,
    gens: [],
    best: row.best ?? null,
    color: RUN_COLORS[state.colorIndex++ % RUN_COLORS.length],
    config: null,
    frame: null,
    speed: '1',
    connection: 'open',
    error: null,
    subscription: null,
  };
  run.subscription = api.subscribe(run.id, {
    frame: (frame) => {
      run.frame = frame;
      if (run.id === state.activeId) view.setFrame(frame);
    },
    gen: (gen) => {
      run.gens.push(gen);
      run.best = Math.max(run.best ?? 0, gen.best);
      renderTabs();
      chart.update();
      form.onGen(run);
    },
    status: (message) => {
      run.status = message.status;
      run.error = message.message ?? null;
      renderTabs();
      renderControls();
      renderBanner();
      if (run.id === state.activeId) form.refreshLock();
    },
    notice: (message) => toast(message.text),
    open: () => setConnection(run, 'open'),
    reconnecting: () => setConnection(run, 'reconnecting'),
    disconnected: () => setConnection(run, 'lost'),
  });
  state.runs.set(run.id, run);
  return run;
}

function setConnection(run, connection) {
  run.connection = connection;
  if (run.id === state.activeId) renderBanner();
}

async function ensureConfig(run) {
  if (run.config) return run.config;
  const data = await api.getRun(run.id);
  run.config = data.config;
  return run.config;
}

async function loadTrack(name) {
  if (!state.tracks.has(name)) state.tracks.set(name, await api.getTrack(name));
  return state.tracks.get(name);
}

async function select(id) {
  state.activeId = id;
  try {
    sessionStorage.setItem('lab.active', id ?? '');
  } catch {
    // storage may be unavailable; the selection is a convenience only
  }
  renderTabs();
  renderControls();
  renderBanner();
  chart.setHighlight(id);
  const run = activeRun();
  el.empty.hidden = Boolean(run);
  if (!run) {
    view.clearTrack();
    el.loading.hidden = true;
    form.showNew({ name: nextRunName() });
    return;
  }
  el.speed.value = run.speed;
  try {
    const config = await ensureConfig(run);
    if (state.activeId !== id) return;
    form.showRun(run, config);
    if (view.trackName !== config.track) {
      el.loading.hidden = false;
      view.clearTrack();
      const track = await loadTrack(config.track);
      if (state.activeId !== id) return;
      view.setTrack(track, state.catalog.style);
    } else {
      view.reset();
    }
    el.loading.hidden = true;
    if (run.frame) view.setFrame(run.frame);
  } catch {
    if (state.activeId === id) {
      el.loading.hidden = true;
      el.bannerText.textContent = 'Не удалось загрузить трассу запуска.';
      el.bannerRetry.hidden = false;
      el.banner.hidden = false;
      el.bannerRetry.onclick = () => select(id);
    }
  }
}

function nextRunName() {
  const names = new Set([...state.runs.values()].map((run) => run.name));
  let number = state.runs.size + 1;
  while (names.has(`Запуск ${number}`)) number += 1;
  return `Запуск ${number}`;
}

// "+ new run" opens the form in new-run mode (it keeps the unsent draft) instead of creating at once.
function openNewForm() {
  if (!state.catalog) return;
  form.showNew({ name: nextRunName() });
  el.paramsPanel.scrollIntoView({ block: 'nearest' });
  form.focusName();
}

function copySettings(run) {
  if (!run?.config) return;
  const config = structuredClone(run.config);
  const names = new Set([...state.runs.values()].map((item) => item.name));
  let name = `${run.name} копия`.slice(0, 40);
  for (let n = 2; names.has(name); n += 1) name = `${run.name} копия ${n}`.slice(0, 40);
  form.showNew({ config: { ...config, name } });
  el.paramsPanel.scrollIntoView({ block: 'nearest' });
  form.focusName();
}

// Called by the form on submit; resolves to true when the run was created.
async function createFromForm(config) {
  if (state.creating) return false;
  state.creating = true;
  renderControls();
  try {
    const { id } = await api.createRun(config);
    const run = addRun({ id, name: config.name, status: 'running', best: null });
    await select(run.id);
    return true;
  } catch (error) {
    form.showServerError(error.message);
    return false;
  } finally {
    state.creating = false;
    renderControls();
  }
}

// Live-editable parameters of a running run; the server echoes them back with the next generation.
async function liveUpdate(run, params) {
  const ok = await sendCommand(run, { cmd: 'update', params });
  if (ok && run.config) Object.assign(run.config, params);
  return ok;
}

async function sendCommand(run, cmd) {
  try {
    await api.command(run.id, cmd);
    return true;
  } catch {
    return false;
  }
}

async function removeRun() {
  const run = activeRun();
  if (!run || !window.confirm(`Удалить запуск «${run.name}»? Его кривые и кадры пропадут.`)) return;
  try {
    await api.deleteRun(run.id);
  } catch (error) {
    if (error.status !== 404) return; // already gone on the server: just drop it from the page
  }
  run.subscription.close();
  state.runs.delete(run.id);
  const remaining = [...state.runs.keys()];
  await select(remaining.length ? remaining[remaining.length - 1] : null);
}

// ---- rendering of the chrome --------------------------------------------------------------

function renderTabs() {
  const fragment = document.createDocumentFragment();
  for (const run of state.runs.values()) {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'tab';
    tab.setAttribute('role', 'tab');
    tab.id = `tab-${run.id}`;
    tab.dataset.status = run.status;
    tab.dataset.id = run.id;
    tab.style.setProperty('--run', run.color);
    tab.setAttribute('aria-selected', String(run.id === state.activeId));
    const label = STATUS_LABELS[run.status] ?? run.status;
    const best = run.best == null ? '' : ` · ${Math.round(run.best * 100)}%`;
    tab.setAttribute('aria-label', `${run.name}, ${label}`);
    const body = document.createElement('span');
    body.className = 'tab-body';
    const name = document.createElement('span');
    name.className = 'tab-name';
    name.textContent = run.name;
    name.title = run.name;
    const meta = document.createElement('span');
    meta.className = 'tab-meta';
    const dot = document.createElement('span');
    dot.className = 'tab-dot';
    meta.append(dot, `${label} · пок. ${run.gens.length}${best}`);
    body.append(name, meta);
    tab.append(body);
    fragment.append(tab);
  }
  el.tabs.replaceChildren(fragment);
  renderLegend();
}

function renderLegend() {
  const items = [...state.runs.values()].map((run) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'legend-item';
    button.dataset.id = run.id;
    button.style.setProperty('--run', run.color);
    button.setAttribute('aria-pressed', String(run.id === state.activeId));
    button.title = `Открыть запуск «${run.name}»`;
    button.textContent = run.name;
    const item = document.createElement('li');
    item.append(button);
    return item;
  });
  el.legend.replaceChildren(...items);
}

function renderMetrics() {
  el.metrics.replaceChildren(
    ...METRICS.map((metric) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'segment';
      button.setAttribute('role', 'radio');
      button.setAttribute('aria-checked', String(metric.id === chart.metric));
      button.dataset.metric = metric.id;
      button.textContent = metric.label;
      return button;
    }),
  );
}

function renderControls() {
  const run = activeRun();
  const live = Boolean(run) && !TERMINAL.has(run.status);
  el.pause.disabled = !live;
  el.pause.textContent = run?.status === 'paused' ? 'Продолжить' : 'Пауза';
  el.speed.disabled = !live;
  el.stop.disabled = !live;
  el.remove.disabled = !run;
  el.newRun.disabled = state.creating || !state.catalog;
}

function renderBanner() {
  const run = activeRun();
  el.bannerRetry.hidden = true;
  if (!run) {
    el.banner.hidden = true;
    return;
  }
  if (run.connection === 'reconnecting') {
    el.bannerText.textContent = 'Соединение потеряно, переподключаемся…';
    el.banner.hidden = false;
  } else if (run.connection === 'lost') {
    el.bannerText.textContent = 'Поток данных отключён.';
    el.bannerRetry.hidden = false;
    el.bannerRetry.onclick = () => {
      run.connection = 'reconnecting';
      renderBanner();
      run.subscription.reconnect();
    };
    el.banner.hidden = false;
  } else if (run.status === 'error') {
    el.bannerText.textContent = `Запуск завершился с ошибкой: ${run.error ?? 'причина неизвестна'}`;
    el.banner.hidden = false;
  } else {
    el.banner.hidden = true;
  }
}

function renderHint() {
  const follow = view.mode === 'follow';
  el.cameraMode.textContent = follow ? 'Следовать' : 'Вписать';
  el.hint.textContent = `Колесо или + / −: масштаб ${view.zoomLabel} · C: камера${follow ? '' : ' · перетаскивание: сдвиг'}`;
}

// ---- wiring -------------------------------------------------------------------------------

function zoom(factor) {
  view.zoomBy(factor);
  renderHint();
}

function wire() {
  el.newRun.addEventListener('click', openNewForm);
  el.emptyNew.addEventListener('click', openNewForm);
  el.tabs.addEventListener('click', (event) => {
    const tab = event.target.closest('.tab');
    if (!tab) return;
    if (tab.dataset.id !== state.activeId) select(tab.dataset.id);
    else if (form.mode === 'new') form.showRun(activeRun(), activeRun().config);
  });
  el.legend.addEventListener('click', (event) => {
    const item = event.target.closest('.legend-item');
    if (item && item.dataset.id !== state.activeId) select(item.dataset.id);
  });
  el.metrics.addEventListener('click', (event) => {
    const segment = event.target.closest('.segment');
    if (!segment) return;
    chart.setMetric(segment.dataset.metric);
    renderMetrics();
  });
  el.pause.addEventListener('click', async () => {
    const run = activeRun();
    if (!run) return;
    const wasPaused = run.status === 'paused';
    if (await sendCommand(run, wasPaused ? 'resume' : 'pause')) {
      run.status = wasPaused ? 'running' : 'paused'; // confirmed later by the status event
      renderTabs();
      renderControls();
    }
  });
  el.speed.addEventListener('change', async () => {
    const run = activeRun();
    if (!run) return;
    const value = el.speed.value === 'max' ? 'max' : Number(el.speed.value);
    if (await sendCommand(run, { cmd: 'speed', value })) run.speed = el.speed.value;
    else el.speed.value = run.speed;
  });
  el.stop.addEventListener('click', () => {
    const run = activeRun();
    if (run) sendCommand(run, 'stop');
  });
  el.remove.addEventListener('click', removeRun);
  el.cameraMode.addEventListener('click', () => view.toggleMode());
  el.zoomIn.addEventListener('click', () => zoom(ZOOM_STEP));
  el.zoomOut.addEventListener('click', () => zoom(1 / ZOOM_STEP));
  el.canvas.addEventListener('wheel', () => requestAnimationFrame(renderHint));
  el.canvas.addEventListener('dblclick', () => requestAnimationFrame(renderHint));

  window.addEventListener('keydown', (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest?.('input, select, textarea, [contenteditable]')) return;
    switch (event.code) {
      case 'KeyC':
        view.toggleMode();
        break;
      case 'Equal':
      case 'NumpadAdd':
        zoom(ZOOM_STEP);
        break;
      case 'Minus':
      case 'NumpadSubtract':
        zoom(1 / ZOOM_STEP);
        break;
      case 'Digit0':
      case 'Numpad0':
        view.resetView();
        renderHint();
        break;
      default:
    }
  });
}

async function init() {
  view = new TrackView(el.canvas, { onModeChange: renderHint });
  chart = new Chart(el.chart, () => [...state.runs.values()]);
  window.__lab = { state, view, chart }; // handle for debugging in the browser console
  renderMetrics();
  wire();
  renderHint();
  renderControls();
  try {
    state.catalog = await api.getCatalog();
  } catch {
    el.empty.hidden = false;
    el.empty.querySelector('p').textContent = 'Не удалось загрузить каталог. Обновите страницу, когда сервер будет доступен.';
    return;
  }
  form = new ParamsForm(el.paramsBody, state.catalog, {
    onCreate: createFromForm,
    onCopy: copySettings,
    onLiveUpdate: liveUpdate,
  });
  window.__lab.form = form;
  form.showNew({ name: nextRunName() });
  document.documentElement.style.setProperty('--grass', state.catalog.style.grass);
  view.setStyle(state.catalog.style);
  renderControls();
  let rows = [];
  try {
    rows = await api.listRuns();
  } catch {
    // reported already; start with an empty list
  }
  rows.forEach(addRun);
  let initial = null;
  try {
    initial = sessionStorage.getItem('lab.active');
  } catch {
    initial = null;
  }
  const ids = rows.map((row) => row.id);
  await select(ids.includes(initial) ? initial : (ids[ids.length - 1] ?? null));
}

init();
