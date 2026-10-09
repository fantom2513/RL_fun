// Tracks screen: the list of tracks to pick from and the editor to draw your own.
// Routes: `#/tracks` (list), `#/tracks/new` (editor on a template), `#/tracks/<name>` (edit a saved
// track, or make a copy of a built-in one).

import { h, icon } from '../core/dom.js';
import { number } from '../core/format.js';
import { TEMPLATES, chaikin, polylineLength, selfIntersections } from '../core/geometry.js';
import { TrackEditorCanvas } from '../views/track_editor_canvas.js';
import '../ui/param.js';

const NAME_PATTERN = /^[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё _-]{0,39}$/;
const MAX_SPEED = 20; // m/s, the car's top speed: sets the best possible lap
const TEMPLATE_LABELS = { oval: 'Овал', trefoil: 'Клевер', bean: 'Почка' };
const MIN_POINTS = 4;

// Draws a track as a small picture: the road at its real width over grass.
function drawThumbnail(canvas, row) {
  const style = getComputedStyle(canvas);
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * ratio));
  canvas.height = Math.max(1, Math.round(box.height * ratio));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = style.getPropertyValue('--scene-grass').trim() || '#7a8e9c';
  ctx.fillRect(0, 0, box.width, box.height);
  const xs = row.centerline.map((p) => p[0]);
  const ys = row.centerline.map((p) => p[1]);
  const pad = row.width + 6;
  const spanX = Math.max(...xs) - Math.min(...xs) + pad * 2;
  const spanY = Math.max(...ys) - Math.min(...ys) + pad * 2;
  const scale = Math.min(box.width / spanX, box.height / spanY);
  const centreX = (Math.max(...xs) + Math.min(...xs)) / 2;
  const centreY = (Math.max(...ys) + Math.min(...ys)) / 2;
  const point = ([x, y]) => [box.width / 2 + (x - centreX) * scale, box.height / 2 - (y - centreY) * scale];
  ctx.beginPath();
  row.centerline.forEach((p, index) => ctx[index ? 'lineTo' : 'moveTo'](...point(p)));
  ctx.closePath();
  ctx.lineJoin = 'round';
  ctx.strokeStyle = style.getPropertyValue('--scene-line').trim() || '#fffffc';
  ctx.lineWidth = (row.width + 1.4) * scale;
  ctx.stroke();
  ctx.strokeStyle = style.getPropertyValue('--scene-road').trim() || '#4f616e';
  ctx.lineWidth = row.width * scale;
  ctx.stroke();
}

const limitSeconds = (length) => length / MAX_SPEED;

export function initTracks({ root, api, onUse, onChanged, toast }) {
  const list = h('div', { class: 'track-list' });
  const listNote = h('p', { class: 'muted', hidden: true });
  const listView = h('div', { class: 'track-list-view' }, listNote, list);
  const editorView = h('div', { class: 'track-editor', hidden: true });
  root.replaceChildren(listView, editorView);

  // ---- list -------------------------------------------------------------------------------

  function card(row) {
    const thumb = h('canvas', { class: 'track-thumb', role: 'img', 'aria-label': `Трасса «${row.name}»` });
    const actions = [h('button', { type: 'button', class: 'btn btn-sm', onclick: () => onUse(row.name) }, 'Выбрать для запуска')];
    if (row.builtin) {
      actions.push(h('a', { class: 'btn btn-sm btn-ghost', href: `#/tracks/${encodeURIComponent(row.name)}` }, 'Сделать копию'));
    } else {
      actions.push(h('a', { class: 'btn btn-sm btn-ghost', href: `#/tracks/${encodeURIComponent(row.name)}` }, 'Изменить'));
      actions.push(h('button', { type: 'button', class: 'btn btn-sm btn-ghost btn-danger', onclick: () => remove(row) }, 'Удалить'));
    }
    const element = h('article', { class: 'card track-item', 'data-track': row.name },
      thumb,
      h('div', { class: 'card-body' },
        h('div', { class: 'track-item-head' },
          h('b', null, row.name),
          h('span', { class: `tag ${row.builtin ? '' : 'tag-info'}` }, row.builtin ? 'встроенная' : 'моя')),
        h('p', { class: 'muted' },
          `${number(row.length, 0)} м · ширина ${number(row.width, 0)} м · круг не быстрее ${number(limitSeconds(row.length), 1)} с`)),
      h('div', { class: 'card-foot' }, ...actions));
    requestAnimationFrame(() => drawThumbnail(thumb, row));
    return element;
  }

  async function remove(row) {
    if (!window.confirm(`Удалить трассу «${row.name}»? Запуски на ней останутся, но новых на этой трассе не будет.`)) return;
    try {
      await api.deleteTrack(row.name);
    } catch {
      return; // the client reports the error itself
    }
    toast(`Трасса «${row.name}» удалена`);
    await onChanged();
    await showList();
  }

  async function showList() {
    listView.hidden = false;
    editorView.hidden = true;
    listNote.hidden = false;
    listNote.textContent = 'Загружаем трассы…';
    list.replaceChildren();
    try {
      const rows = await api.listTracks();
      listNote.hidden = true;
      list.replaceChildren(...rows.map(card));
    } catch {
      listNote.textContent = 'Не удалось получить список трасс. Проверьте, что сервер запущен, и обновите страницу.';
    }
  }

  // ---- editor -----------------------------------------------------------------------------

  let editor = null;
  let original = null; // name of the saved track being edited, null for a new one

  function buildEditor() {
    const canvas = h('canvas', {
      class: 'track-canvas', tabindex: '0',
      'aria-label': 'Поле редактора: щёлкните, чтобы поставить точку трассы; перетащите точку, чтобы сдвинуть; правая кнопка или Delete удаляют точку',
    });
    const name = h('input', { class: 'input', id: 'track-name', maxlength: '40', autocomplete: 'off' });
    const width = h('lab-param', { label: 'Ширина трассы', hint: 'Шире — проще проехать, но трассе нужно больше места.', min: 6, max: 24, step: 1, value: 10, default: 10, unit: 'м', digits: 0 });
    const stats = h('p', { class: 'track-stats' });
    const status = h('div', { class: 'callout', role: 'status' });
    const serverError = h('div', { class: 'callout callout-err', role: 'alert', hidden: true });
    const undo = h('button', { type: 'button', class: 'btn btn-sm', onclick: () => editor.undo() }, icon('undo-2'), 'Отменить');
    const smooth = h('button', { type: 'button', class: 'btn btn-sm', onclick: () => editor.setPoints(chaikin(editor.points, 1)) }, 'Сгладить');
    const clear = h('button', { type: 'button', class: 'btn btn-sm', onclick: () => editor.setPoints([]) }, 'Очистить');
    const templates = Object.keys(TEMPLATES).map((key) =>
      h('button', { type: 'button', class: 'chip', onclick: () => editor.setPoints(TEMPLATES[key]()) }, TEMPLATE_LABELS[key]));
    const save = h('button', { type: 'button', class: 'btn btn-primary', onclick: () => submit() }, 'Сохранить трассу');
    const cancel = h('a', { class: 'btn', href: '#/tracks' }, 'Отмена');
    const remove = h('button', { type: 'button', class: 'btn btn-danger btn-ghost', onclick: () => removeCurrent() }, 'Удалить');

    editor = new TrackEditorCanvas(canvas, { onChange: refresh });
    width.addEventListener('param-change', (event) => editor.setWidth(event.detail.value));
    name.addEventListener('input', refresh);

    const panel = h('aside', { class: 'card track-props' },
      h('div', { class: 'card-body track-props-body' },
        h('div', { class: 'field' }, h('label', { class: 'field-label', for: 'track-name' }, 'Название'), name),
        width,
        h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'Начать с формы'), h('div', { class: 'track-tools' }, ...templates)),
        h('div', { class: 'track-tools' }, undo, smooth, clear),
        stats, status, serverError,
        h('p', { class: 'muted' }, 'Щёлчок — новая точка, на дороге — вставка. Перетаскивайте точки, Delete или правая кнопка удаляют, стрелки сдвигают на метр, Ctrl+Z отменяет. Первая точка и стрелка — старт.')),
      h('div', { class: 'card-foot' }, save, cancel, remove));
    editorView.replaceChildren(h('div', { class: 'track-canvas-wrap' }, canvas), panel);
    Object.assign(editorView, { parts: { name, width, stats, status, serverError, save, remove, undo, smooth } });
  }

  // The reason the track cannot be saved yet, or null; the server repeats and extends these rules.
  function problem() {
    const { parts } = editorView;
    if (!NAME_PATTERN.test(parts.name.value)) return 'Введите название: буквы, цифры, пробел, дефис, до 40 символов.';
    if (editor.points.length < MIN_POINTS) return `Поставьте хотя бы ${MIN_POINTS} точки: щёлкните по полю или выберите форму.`;
    const points = editor.points;
    for (let index = 0; index < points.length; index += 1) {
      const next = points[(index + 1) % points.length];
      if (Math.hypot(next[0] - points[index][0], next[1] - points[index][1]) < 1) return 'Две соседние точки слишком близко: расстояние не меньше 1 м.';
    }
    if (selfIntersections(points).length) return 'Трасса пересекает сама себя: красные участки нужно развести.';
    return null;
  }

  function refresh() {
    const { parts } = editorView;
    const length = polylineLength(editor.points);
    parts.stats.textContent = `Точек: ${editor.points.length} · длина ${number(length, 0)} м` +
      (length ? ` · круг не быстрее ${number(limitSeconds(length), 1)} с` : '');
    const reason = problem();
    parts.status.className = `callout ${reason ? 'callout-warn' : 'callout-ok'}`;
    parts.status.textContent = reason ?? 'Трасса готова: можно сохранять.';
    parts.save.disabled = Boolean(reason);
    parts.undo.disabled = !editor.canUndo;
    parts.serverError.hidden = true;
  }

  async function submit(overwrite = original !== null && editorView.parts.name.value === original) {
    const { parts } = editorView;
    parts.save.disabled = true;
    const definition = { name: parts.name.value, width: Number(parts.width.value), centerline: editor.points, overwrite };
    try {
      await api.saveTrack(definition);
    } catch (error) {
      if (error.status === 409 && window.confirm(`${error.message}. Заменить её?`)) {
        await submit(true);
        return;
      }
      parts.serverError.textContent = error.message;
      parts.serverError.hidden = false;
      parts.save.disabled = Boolean(problem());
      return;
    }
    toast(`Трасса «${definition.name}» сохранена`);
    await onChanged();
    location.hash = '#/tracks';
  }

  async function removeCurrent() {
    if (original === null) return;
    await remove({ name: original });
    location.hash = '#/tracks';
  }

  async function showEditor(target) {
    listView.hidden = true;
    editorView.hidden = false;
    if (!editor) buildEditor();
    const { parts } = editorView;
    original = null;
    let points = TEMPLATES.oval();
    let width = 10;
    let name = '';
    if (target !== 'new') {
      try {
        const row = await api.getTrack(target);
        points = row.centerline;
        width = row.width;
        const saved = (await api.listTracks()).find((item) => item.name === target);
        if (saved && !saved.builtin) {
          original = target;
          name = target;
        } else {
          name = `${target} копия`.slice(0, 40);
        }
      } catch {
        location.hash = '#/tracks';
        return;
      }
    }
    parts.name.value = name;
    parts.width.value = width;
    editor.setTrack(points, width);
    parts.remove.hidden = original === null;
    editor.resize();
    parts.name.focus();
  }

  return {
    show(route) {
      const [target] = route.rest;
      if (target === undefined) return showList();
      return showEditor(target === 'new' ? 'new' : decodeURIComponent(target));
    },
    refresh: showList,
  };
}
