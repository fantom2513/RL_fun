// Comparison table of runs: row derivation, best-per-track flags, sorting and rendering.

import { h } from "../core/dom.js";
import { number, percent } from "../core/format.js";
import { STEP_SECONDS } from "../core/levels.js";
import { completedIds } from "../core/run_history.js";

export const LEARNER_LABELS = { evolution: "Эволюция", ppo: "PPO", cem: "CEM", a2c: "A2C" };

const STATUS_LABELS = {
  running: "идёт",
  paused: "пауза",
  finished: "готово",
  stopped: "остановлен",
  error: "ошибка",
};

const COLUMNS = [
  { key: "name", label: "Запуск" },
  { key: "status", label: "Статус" },
  { key: "iterations", label: "Итераций" },
  { key: "bestProgress", label: "Лучший прогресс" },
  { key: "meanProgress", label: "Средний прогресс" },
  { key: "bestLap", label: "Лучшее время, с" },
  { key: "firstLap", label: "Первый круг, итер." },
  { key: "finishedLast", label: "Финишировали" },
];

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const isNil = (v) => v === null || v === undefined;

function deriveRow(run) {
  const gens = run.gens ?? [];
  const last = gens.length ? gens[gens.length - 1] : null;
  const bests = gens.map((g) => g.best).filter(isNum);
  const laps = gens
    .map((g) => g.best_lap_steps)
    .filter(isNum)
    .map((steps) => steps * STEP_SECONDS);
  const firstIndex = gens.findIndex((g) => g.finished > 0);

  return {
    id: run.id,
    name: run.name,
    color: run.color,
    learner: run.learner,
    status: run.status,
    track: run.config?.track ?? null,
    iterations: gens.length,
    bestProgress: bests.length ? Math.max(...bests) : null,
    meanProgress: last && isNum(last.mean) ? last.mean : null,
    bestLap: laps.length ? Math.min(...laps) : null,
    firstLap: firstIndex === -1 ? null : firstIndex + 1,
    finishedLast: last && isNum(last.finished) ? last.finished : 0,
    population: run.config?.population ?? null,
    best: { progress: false, lap: false, firstLap: false },
  };
}

// Marks, within one track group, the rows that hold the winning value of a field.
function markWinners(group, field, flag, pick) {
  const values = group.map((row) => row[field]).filter(isNum);
  if (!values.length) return;
  const winner = pick(...values);
  for (const row of group) {
    if (row[field] === winner) row.best[flag] = true;
  }
}

export function compareRows(runs) {
  const rows = runs.map(deriveRow);

  const byTrack = new Map();
  for (const row of rows) {
    if (row.track === null) continue; // rows without a track never compete
    if (!byTrack.has(row.track)) byTrack.set(row.track, []);
    byTrack.get(row.track).push(row);
  }
  for (const group of byTrack.values()) {
    markWinners(group, "bestProgress", "progress", Math.max);
    markWinners(group, "bestLap", "lap", Math.min);
    markWinners(group, "firstLap", "firstLap", Math.min);
  }
  return rows;
}

// Nulls always go last; the comparison is stable and does not mutate the input.
export function sortRows(rows, key, direction) {
  const sign = direction === "desc" ? -1 : 1;
  return [...rows].sort((x, y) => {
    const a = x[key];
    const b = y[key];
    const aNil = isNil(a);
    const bNil = isNil(b);
    if (aNil || bNil) return aNil === bNil ? 0 : aNil ? 1 : -1;
    if (typeof a === "string" && typeof b === "string") return sign * a.localeCompare(b, "ru");
    return sign * (a > b ? 1 : a < b ? -1 : 0);
  });
}

const dash = (v, format) => (isNil(v) ? "—" : format(v));

function bestWrap(flagged, text) {
  return flagged ? h("span", { class: "cell-best" }, text) : text;
}

function nameCell(row) {
  const learner = LEARNER_LABELS[row.learner] ?? row.learner;
  const track = row.track ?? "—";
  return h(
    "td",
    null,
    h("div", null, h("i", { class: "dot", style: `background:${row.color}` }), row.name),
    h("small", { class: "muted" }, `${learner} · ${track}`),
  );
}

function statusCell(row) {
  return h(
    "td",
    null,
    h("i", { class: `dot dot-${row.status}` }),
    STATUS_LABELS[row.status] ?? row.status,
  );
}

function bodyRow(row, { onSelect, onToggle, onDelete, selectedIds, busy }) {
  const removable = completedIds([row]).length > 0;
  const finished = isNil(row.population)
    ? number(row.finishedLast, 0)
    : `${number(row.finishedLast, 0)} из ${number(row.population, 0)}`;
  return h(
    "tr",
    {
      tabindex: 0,
      dataset: { historyFocus: `row:${row.id}` },
      onclick: () => onSelect?.(row.id),
      onkeydown: (event) => {
        if (event.target === event.currentTarget && event.key === "Enter") onSelect?.(row.id);
      },
      "aria-selected": String(selectedIds?.has(row.id) ?? false),
    },
    h("td", { class: "history-select", onclick: (event) => event.stopPropagation() },
      h("input", {
        type: "checkbox",
        checked: selectedIds?.has(row.id),
        disabled: !removable || busy,
        "aria-label": `Выбрать заезд «${row.name}»`,
        dataset: { selectRun: row.id, historyFocus: `select:${row.id}` },
        onchange: (event) => onToggle?.(row.id, event.target.checked),
      })),
    nameCell(row),
    statusCell(row),
    h("td", null, number(row.iterations, 0)),
    h("td", null, bestWrap(row.best.progress, dash(row.bestProgress, percent))),
    h("td", null, dash(row.meanProgress, percent)),
    h("td", null, bestWrap(row.best.lap, dash(row.bestLap, (v) => number(v, 2)))),
    h("td", null, bestWrap(row.best.firstLap, dash(row.firstLap, (v) => number(v, 0)))),
    h("td", null, finished),
    h("td", { class: "history-actions", onclick: (event) => event.stopPropagation() },
      h("button", {
        type: "button", class: "btn btn-small btn-danger",
        dataset: { historyFocus: `delete:${row.id}` },
        disabled: !removable || busy,
        "aria-label": `Удалить заезд «${row.name}»`,
        title: removable ? "Удалить из истории и гаража" : "Сначала остановите заезд",
        onclick: () => onDelete?.(row.id),
      }, "Удалить")),
  );
}

function headerCell(column, sortKey, sortDirection, onSort) {
  let ariaSort = "none";
  if (column.key === sortKey) ariaSort = sortDirection === "desc" ? "descending" : "ascending";
  return h(
    "th",
    { class: "none", "aria-sort": ariaSort },
    h(
      "button",
      {
        type: "button",
        dataset: { sort: column.key, historyFocus: `sort:${column.key}` },
        onclick: () => onSort?.(column.key),
      },
      column.label,
    ),
  );
}

export function renderCompare(container, rows, options = {}) {
  const { sortKey, sortDirection, onSort, onSelectAll, onDeleteSelected, selectedIds = new Set(), busy = false } = options;
  const focusKey = container.contains(document.activeElement) ? document.activeElement?.dataset?.historyFocus : null;
  const wrap = container.querySelector('.history-table-wrap');
  const scrollLeft = wrap?.scrollLeft ?? 0;
  const scrollTop = wrap?.scrollTop ?? 0;
  if (!rows.length) {
    container.replaceChildren(
      h(
        "div",
        { class: "empty" },
        h("b", null, "Нет запусков"),
        h("span", null, "Создайте запуск в лаборатории, и он появится в таблице."),
      ),
    );
    return;
  }

  const ordered = sortKey ? sortRows(rows, sortKey, sortDirection) : rows;
  const removable = completedIds(rows);
  const selected = removable.filter((id) => selectedIds.has(id));
  const allSelected = removable.length > 0 && selected.length === removable.length;
  const toolbar = h("div", { class: "history-toolbar" },
    h("div", { class: "history-heading" },
      h("h2", null, "История заездов"),
      h("p", { class: "muted" }, `Всего: ${rows.length} · можно удалить: ${removable.length}`)),
    h("div", { class: "history-controls" },
      h("button", {
        type: "button", class: "btn", disabled: !removable.length || busy,
        dataset: { historyFocus: 'select-all' },
        onclick: () => onSelectAll?.(!allSelected),
      }, allSelected ? "Снять выделение" : "Выбрать старые"),
      h("button", {
        type: "button", class: "btn btn-danger", disabled: !selected.length || busy,
        dataset: { historyFocus: 'delete-selected' },
        onclick: onDeleteSelected,
      }, busy ? "Удаление…" : `Удалить выбранные${selected.length ? ` (${selected.length})` : ''}`)));
  const table = h(
    "table",
    { class: "table" },
    h(
      "thead",
      null,
      h("tr", null, h("th", { "aria-label": "Выбор заездов" }),
        COLUMNS.map((column) => headerCell(column, sortKey, sortDirection, onSort)),
        h("th", null, "Действия")),
    ),
    h("tbody", null, ordered.map((row) => bodyRow(row, { ...options, selectedIds, busy }))),
  );
  const nextWrap = h("div", { class: "history-table-wrap" }, table);
  container.replaceChildren(toolbar, nextWrap);
  nextWrap.scrollLeft = scrollLeft;
  nextWrap.scrollTop = scrollTop;
  if (focusKey) {
    [...container.querySelectorAll('[data-history-focus]')]
      .find((node) => node.dataset.historyFocus === focusKey)?.focus({ preventScroll: true });
  }
}
