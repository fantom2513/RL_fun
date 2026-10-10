// Lab shell: run tabs, run controls, the track view, the parameters form, the learning curves and
// the leader's network graph.

import * as api from './api.js';
import { Chart, METRICS } from './charts.js';
import { startKeys } from './core/keys.js';
import { hashFor, startRouter } from './core/router.js';
import { getTheme, initTheme, setTheme, toggleTheme } from './core/theme.js';
import { modelLabels, NetworkView } from './network_view.js';
import { ParamsForm } from './params_form.js';
import { TrackView } from './track_view.js';
import { LEVELS, bestStars, bestValue, isUnlocked, loadStars, saveStars } from './core/levels.js';
import { renderChallenges, renderTaskPanel, showLevelComplete, starSummary } from './screens/challenges.js';
import { initGarage } from './screens/garage.js';
import { initTracks } from './screens/tracks.js';
import { compareRows, renderCompare, sortRows } from './screens/compare.js';
import { initTooltips } from './ui/tooltip.js';
import { summarizeRun } from './core/run_summary.js';
import './ui/seg.js';

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
  layout: $('run-layout'),
  paramsToggle: $('params-toggle'),
  fleetToggle: $('fleet-toggle'),
  raysToggle: $('rays-toggle'),
  networkToggle: $('network-toggle'),
  networkContent: $('network-content'),
  toasts: $('toasts'),
  announcer: $('announcer'),
  paramsPanel: document.querySelector('.params'),
  paramsBody: $('params-body'),
  paramsStatus: $('params-status'),
  paramsRetry: $('params-retry'),
  networkBody: $('network-body'),
  networkLegend: $('network-legend'),
  chart: $('chart'),
  metrics: $('metrics'),
  legend: $('legend'),
};

const state = {
  challenge: null,
  stars: loadStars(),
  compareSort: { key: 'bestProgress', direction: 'desc' },
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
let network = null;
let pauseBusy = false;

// ---- notifications ------------------------------------------------------------------------

function toast(text, kind = 'info') {
  const node = document.createElement('div');
  node.className = `toast${kind === 'error' ? ' toast-error' : ''}`;
  if (kind === 'error') node.setAttribute('role', 'alert');
  node.textContent = text;
  el.toasts.append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 8000 : 5000);
  while (el.toasts.children.length > 4) el.toasts.firstChild.remove();
}

api.setErrorHandler((message) => toast(message, 'error'));

// Screen-reader announcement of state changes that are otherwise only visible (pause, finish…).
function announce(text) {
  el.announcer.textContent = '';
  setTimeout(() => {
    el.announcer.textContent = text;
  }, 30);
}

function setRunStatus(run, status) {
  if (run.status === status) return;
  run.status = status;
  if (run.id === state.activeId) announce(`${run.name}: ${STATUS_LABELS[status] ?? status}`);
}

// What the run tabs and the new-run form say about the selection: the open run's tab looks
// selected only while its own parameters are shown, not while the form is writing a new run.
const formMode = () => (form?.mode === 'edit' ? 'run' : (form?.mode ?? 'new'));

// ---- runs ---------------------------------------------------------------------------------

const activeRun = () => state.runs.get(state.activeId) ?? null;

function addRun(row) {
  const run = {
    id: row.id,
    name: row.name,
    status: row.status,
    gens: [],
    best: row.best ?? null,
    learner: row.learner ?? 'evolution',
    archived: Boolean(row.archived),
    color: RUN_COLORS[state.colorIndex++ % RUN_COLORS.length],
    config: null,
    frame: null,
    net: null,
    speed: '1',
    connection: 'open',
    error: null,
    subscription: null,
  };
  run.subscription = api.subscribe(run.id, {
    frame: (frame) => {
      run.frame = frame;
      run.best = Math.max(run.best ?? 0, frame.hud.progress);
      const first = !run.net && Boolean(frame.net);
      if (frame.net) run.net = frame.net;
      if (run.id !== state.activeId) return;
      view.setFrame(frame);
      renderSummary();
      if (run.config) network.setNet(frame.net);
      if (first) renderNetworkState();
    },
    gen: (gen) => {
      run.gens.push(gen);
      run.best = Math.max(run.best ?? 0, gen.best);
      if (run.id === state.activeId) renderSummary();
      renderTabs();
      chart.update();
      form.onGen(run);
      scheduleGame();
    },
    status: (message) => {
      setRunStatus(run, message.status);
      run.error = message.message ?? null;
      renderTabs();
      renderControls();
      renderBanner();
      scheduleGame();
      if (run.id === state.activeId) renderNetworkState();
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
  if (run.id !== state.activeId) return;
  renderBanner();
  renderNetworkState();
}

// Empty, waiting and error texts of the network panel; the graph itself shows once a net arrived.
function renderNetworkState() {
  const run = activeRun();
  let text = null;
  if (!run) {
    text = 'Сеть лидера появится, когда вы создадите запуск.';
  } else if (!run.net) {
    if (run.status === 'error') text = 'Запуск завершился с ошибкой, сеть не получена.';
    else if (run.connection === 'lost') text = 'Поток данных отключён, сети пока нет.';
    else if (run.archived) text = 'Запуск из прошлого сеанса: кадров нет, но кривые и настройки на месте. «Изменить и перезапустить» запустит его снова.';
    else if (TERMINAL.has(run.status)) text = 'Запуск закончился до первого кадра, сети нет.';
    else text = 'Ждём первый кадр запуска…';
  }
  network.setMessage(text);
}

async function ensureConfig(run) {
  if (run.config) return run.config;
  const data = await api.getRun(run.id);
  run.config = data.config;
  run.learner = data.config.learner ?? 'evolution';
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
  renderSummary();
  const run = activeRun();
  if (run) view.setLearner(run.learner);
  el.empty.hidden = Boolean(run);
  network.clear();
  network.setLabels(null);
  renderNetworkState();
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
    renderSummary();
    form.showRun(run, config);
    view.setLaps(config.laps ?? 1);
    network.setLabels(modelLabels(config.model, state.catalog));
    network.setNet(run.net);
    renderNetworkState();
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
  setParamsOpen(true);
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
  setParamsOpen(true);
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
    const run = addRun({ id, name: config.name, status: 'running', best: null, learner: config.learner });
    state.challenge = null;
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

// Edit mode: the run's settings open for any change; submitting restarts it in the same tab.
async function editRun(run) {
  if (!run) return;
  const config = await ensureConfig(run);
  form.showEdit(run, config);
  el.paramsPanel.scrollIntoView({ block: 'nearest' });
}

// Starts a replacement with the edited config and puts it where the old run was: same name,
// colour and tab position, so the user sees one run that was changed, not a second one.
async function restartRun(old, config) {
  if (state.creating) return false;
  state.creating = true;
  renderControls();
  try {
    const { id } = await api.createRun(config);
    try {
      await api.deleteRun(old.id);
    } catch (error) {
      if (error.status !== 404) throw error;
    }
    old.subscription.close();
    const run = addRun({ id, name: config.name, status: 'running', best: null, learner: config.learner });
    state.colorIndex -= 1;
    run.color = old.color;
    state.runs = new Map([...state.runs].flatMap(([key, item]) => {
      if (key === old.id) return [[run.id, run]];
      return key === run.id ? [] : [[key, item]];
    }));
    await select(run.id);
    toast(`«${run.name}» перезапущен с новыми настройками`);
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

// ---- tracks: the list, the editor and the use of a track in a new run -------------------------

let tracksScreen = null;
let garageScreen = null;

// A track was saved or deleted: the catalog (the form's list of tracks) follows the server.
async function reloadTracks() {
  try {
    state.catalog = await api.getCatalog();
  } catch {
    return;
  }
  form?.setTracks(state.catalog.tracks);
}

function useTrack(name) {
  if (!state.catalog) return;
  location.hash = '#/lab';
  openNewForm();
  form.setTrack(name);
}

// ---- game layer: levels, stars and the comparison table --------------------------------------

let gameTimer = null;

// Gen messages arrive many times a second; the screens that depend on them redraw at most twice.
function scheduleGame() {
  if (gameTimer) return;
  gameTimer = setTimeout(() => {
    gameTimer = null;
    refreshGame();
  }, 500);
}

function refreshGame() {
  const runs = [...state.runs.values()];
  const next = bestStars(runs, state.stars);
  let changed = false;
  for (const level of LEVELS) {
    const count = next[level.id] ?? 0;
    if (count <= (state.stars[level.id] ?? 0)) continue;
    changed = true;
    // The first pass after the page loaded only restores what the history already earned.
    if (!state.gameReady) continue;
    const unlocked = LEVELS.find((other) => other.requires?.level === level.id
      && isUnlocked(other, next) && !isUnlocked(other, state.stars));
    showLevelComplete({
      level, stars: count, value: bestValue(level, runs), unlocked,
      onNext: (target) => startLevel(target),
    });
  }
  state.stars = next;
  if (changed) saveStars(next);
  renderChallenges($('challenges-list'), { stars: next, runs, onStart: startLevel });
  $('stars-total').textContent = starSummary(next);
  renderCompareView();
}

function renderCompareView() {
  const rows = compareRows([...state.runs.values()]);
  const { key, direction } = state.compareSort;
  renderCompare($('compare-view'), sortRows(rows, key, direction), {
    sortKey: key,
    sortDirection: direction,
    onSort: (column) => {
      const same = state.compareSort.key === column;
      state.compareSort = { key: column, direction: same && state.compareSort.direction === 'desc' ? 'asc' : 'desc' };
      renderCompareView();
    },
    onSelect: (id) => {
      location.hash = '#/lab';
      select(id);
    },
  });
}

// "Start attempt" of a level: the stage shows its track and the form opens with the stock settings
// except the track; the task panel lists what the level asks and, on request, how to set it.
async function startLevel(level) {
  if (!state.catalog) return;
  location.hash = '#/lab';
  state.challenge = { level, opened: false };
  await select(null);
  const config = structuredClone(state.catalog.defaults);
  form.showNew({ config: { ...config, track: level.track, name: level.title } });
  setParamsOpen(true);
  renderTask();
  el.paramsPanel.scrollIntoView({ block: 'nearest' });
  form.focusName();
}

// The task panel follows the form while a challenge is being set up.
function renderTask() {
  const panel = $('task-panel');
  const challenge = state.challenge;
  if (!challenge || !form || form.mode !== 'new') {
    panel.hidden = true;
    return;
  }
  renderTaskPanel(panel, {
    level: challenge.level,
    config: form.read(),
    opened: challenge.opened,
    onToggle: () => {
      challenge.opened = !challenge.opened;
      renderTask();
    },
    onCancel: () => {
      state.challenge = null;
      renderTask();
    },
  });
}

// With no run open the stage previews the track chosen in the form instead of staying empty.
async function previewTrack(name) {
  if (!form || !state.catalog || form.mode !== 'new' || activeRun() || view.trackName === name) return;
  try {
    const track = await loadTrack(name);
    if (activeRun() || form.fields.track.value !== name) return;
    view.setTrack(track, state.catalog.style);
    el.empty.hidden = true;
    el.loading.hidden = true;
    el.hint.textContent = `Предпросмотр трассы «${name}»: настройте запуск слева и нажмите «Создать запуск».`;
  } catch {
    // the preview is a convenience; the form still works without it
  }
}

// The lab shows either the stage or the comparison table; the route `#/lab/compare` picks the table.
function showLabView(compare) {
  const labScreen = document.querySelector('.lab-screen');
  labScreen.dataset.view = compare ? 'compare' : 'run';
  $('compare-view').hidden = !compare;
  document.querySelector('.layout').style.display = compare ? 'none' : '';
  const toggle = $('view-toggle');
  toggle.textContent = compare ? 'Заезд' : 'Сравнение';
  toggle.setAttribute('href', compare ? '#/lab' : '#/lab/compare');
  if (compare) renderCompareView();
  else window.dispatchEvent(new Event('resize'));
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
    // While the form writes a new run no tab is the selected one; the stage keeps showing the
    // last opened run, and its tab is marked as such only through `aria-current`.
    const selected = run.id === state.activeId && formMode() === 'run';
    tab.setAttribute('aria-selected', String(selected));
    if (run.id === state.activeId && !selected) {
      tab.setAttribute('aria-current', 'true');
      tab.title = 'Этот запуск показан на трассе; справа открыта форма нового запуска';
    }
    const label = STATUS_LABELS[run.status] ?? run.status;
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
    meta.append(dot, `${label}${run.archived ? ' · архив' : ''}`);
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
  el.hint.textContent = `Колесо или + / −: масштаб ${view.zoomLabel} · C: камера · Пробел: пауза${follow ? '' : ' · перетаскивание: сдвиг'}`;
}

function renderSummary() {
  const summary = summarizeRun(activeRun());
  for (const key of ['iteration', 'best', 'alive', 'finished']) {
    const node = $(`stat-${key}`);
    if (node.textContent !== summary[key]) node.textContent = summary[key];
  }
  $('stat-iteration-label').textContent = summary.iterationLabel;
  document.querySelectorAll('.stat-race-hint').forEach((node) => {
    node.textContent = summary.raceHint;
  });
}

function setParamsOpen(open) {
  el.layout.dataset.paramsOpen = String(open);
  el.paramsToggle.setAttribute('aria-expanded', String(open));
  el.paramsToggle.textContent = open ? 'Скрыть параметры' : 'Параметры';
}

// ---- wiring -------------------------------------------------------------------------------

// Pause or resume the open run (button and Space).
async function togglePause() {
  const run = activeRun();
  if (!run || TERMINAL.has(run.status) || pauseBusy) return;
  pauseBusy = true;
  try {
    const wasPaused = run.status === 'paused';
    if (await sendCommand(run, wasPaused ? 'resume' : 'pause')) {
      setRunStatus(run, wasPaused ? 'running' : 'paused'); // confirmed later by the status event
      renderTabs();
      renderControls();
    }
  } finally {
    pauseBusy = false;
  }
}

function zoom(factor) {
  view.zoomBy(factor);
  renderHint();
}

function wire() {
  setParamsOpen(window.matchMedia('(min-width: 981px)').matches);
  el.paramsToggle.addEventListener('click', () => {
    setParamsOpen(el.paramsToggle.getAttribute('aria-expanded') !== 'true');
  });
  el.networkToggle.addEventListener('click', () => {
    const open = el.networkContent.hidden;
    el.networkContent.hidden = !open;
    el.networkToggle.setAttribute('aria-expanded', String(open));
    el.networkToggle.textContent = open ? 'Скрыть сеть' : 'Показать сеть';
  });
  el.fleetToggle.addEventListener('click', () => {
    const leaderOnly = el.fleetToggle.getAttribute('aria-pressed') !== 'true';
    el.fleetToggle.setAttribute('aria-pressed', String(leaderOnly));
    view.setDisplay({ showFleet: !leaderOnly });
  });
  el.raysToggle.addEventListener('click', () => {
    const showRays = el.raysToggle.getAttribute('aria-pressed') !== 'true';
    el.raysToggle.setAttribute('aria-pressed', String(showRays));
    view.setDisplay({ showRays });
  });
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
  el.pause.addEventListener('click', togglePause);
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
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;
    if (document.body.dataset.screen !== 'lab') return;
    // Never act while the user types or operates a control: shortcuts are for the stage only.
    const target = event.target.closest ? event.target : document.body;
    if (target.closest('input, select, textarea, [contenteditable]')) return;
    if (event.code === 'Space') {
      // On a focused button Space already means "press it".
      if (target.closest('button, a[href], summary, [role="tab"], [role="radio"]')) return;
      event.preventDefault();
      if (!event.repeat) togglePause();
      return;
    }
    switch (event.code) {
      case 'KeyC':
        view.toggleMode();
        break;
      case 'KeyG':
        location.hash = document.querySelector('.lab-screen').dataset.view === 'compare' ? '#/lab' : '#/lab/compare';
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
  network = new NetworkView(el.networkBody, el.networkLegend);
  window.__lab = { state, view, chart, network }; // handle for debugging in the browser console
  renderMetrics();
  wire();
  renderHint();
  renderControls();
  renderNetworkState();
  el.paramsRetry.addEventListener('click', start);
  await start();
}

const EMPTY_TEXT = el.empty.querySelector('p').textContent;

// Loads the catalog and everything that depends on it; the retry button runs it again.
async function start() {
  el.paramsStatus.textContent = 'Загрузка параметров…';
  el.paramsRetry.hidden = true;
  try {
    state.catalog = await api.getCatalog();
  } catch (error) {
    el.paramsStatus.textContent = 'Параметры недоступны: каталог не загрузился.';
    el.paramsRetry.hidden = false;
    el.empty.hidden = false;
    el.emptyNew.hidden = true;
    el.empty.querySelector('h2').textContent = 'Лаборатория недоступна';
    el.empty.querySelector('p').textContent = `${error.message} Нажмите «Повторить» в панели параметров.`;
    return;
  }
  el.empty.querySelector('h2').textContent = 'Создайте запуск';
  el.empty.querySelector('p').textContent = EMPTY_TEXT;
  el.emptyNew.hidden = false;
  form = new ParamsForm(el.paramsBody, state.catalog, {
    onCreate: createFromForm,
    onCopy: copySettings,
    onEdit: editRun,
    onCancelEdit: (run) => select(run.id),
    onRestart: restartRun,
    onLiveUpdate: liveUpdate,
    onModeChange: () => {
      renderTabs();
      renderTask();
    },
    onChange: renderTask,
    onTrack: previewTrack,
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
  await Promise.all(rows.map((row) => ensureConfig(state.runs.get(row.id)).catch(() => null)));
  refreshGame();
  state.gameReady = true;
  const ids = rows.map((row) => row.id);
  await select(ids.includes(initial) ? initial : (ids[ids.length - 1] ?? null));
}

// The shell: hash routes, section keys, the theme preference and the rail tooltips.
function initShell() {
  initTheme();
  initTooltips();
  tracksScreen = initTracks({
    root: $('tracks-root'), api, onUse: useTrack, onChanged: reloadTracks, toast,
  });
  garageScreen = initGarage({ root: $('garage-root'), api, getCatalog: () => state.catalog, toast });
  startRouter((route) => {
    showLabView(route.name === 'lab' && route.rest[0] === 'compare');
    if (route.name === 'tracks') tracksScreen.show(route);
    if (route.name === 'garage') garageScreen.show();
    else garageScreen.hide();
    $('track-new').hidden = route.name === 'tracks' && route.rest.length > 0;
  });
  startKeys({ navigate: (name) => { location.hash = hashFor(name); }, toggleTheme });
  const themePref = $('theme-pref');
  themePref.value = getTheme();
  themePref.addEventListener('seg-change', (event) => setTheme(event.detail.value));
  window.addEventListener('themechange', (event) => { themePref.value = event.detail.preference; });
}

initShell();
init();
