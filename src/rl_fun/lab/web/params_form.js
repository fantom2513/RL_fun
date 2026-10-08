// Parameters form. It builds the whole form once from the catalog and keeps the DOM as the source
// of truth: `read()` turns the controls into a run config, `write()` fills them from one. Client
// validation repeats the main RunConfig/ModelSpec rules for instant feedback; the server stays the
// authority and its errors are shown too. For a running run only the live-editable fields stay
// enabled and their changes are sent as `update` commands (debounced).

const RAY_PREFIX = 'ray:';
const RAY_PRESETS = [
  { label: '3 луча', angles: [-60, 0, 60] },
  { label: '5 лучей', angles: [-90, -30, 0, 30, 90] },
  { label: '7 лучей', angles: [-90, -60, -30, 0, 30, 60, 90] },
];
const MAX_LAYERS = 6;
const MAX_NEURONS = 64;
const MAX_NAME = 40;
const LIVE_DELAY = 300;
const TERMINAL = new Set(['finished', 'stopped', 'error']);

const OUTPUT_TEXT = {
  steer: ['Руль', 'обязателен'],
  throttle: ['Газ и тормоз', 'один выход: плюс — газ, минус — тормоз'],
  accelerate: ['Газ', 'отдельный выход, к нему можно добавить тормоз'],
  brake: ['Тормоз', 'только вместе с отдельным газом'],
  boost: ['Буст', 'необязательный рывок'],
};
const OUTPUT_RULES =
  'Руль нужен всегда. Газ задаётся одним выходом «Газ и тормоз» либо отдельным «Газ»; ' +
  'тормоз добавляется только к отдельному газу.';

let uid = 0;

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key in node && key !== 'list') node[key] = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.flat().filter((child) => child != null && child !== false));
  return node;
}

function plural(n, one, few, many) {
  const tail = Math.abs(n) % 100;
  const last = tail % 10;
  if (tail > 10 && tail < 20) return many;
  if (last === 1) return one;
  if (last >= 2 && last <= 4) return few;
  return many;
}

const fmt = (value, digits = 2) =>
  Number(value).toLocaleString('ru-RU', { maximumFractionDigits: digits });

const isInt = (value) => Number.isInteger(value);
const rayInput = (angle) => `${RAY_PREFIX}${angle}`;

export function summarizeModel(inputs, hidden, outputs) {
  const sizes = [inputs, ...hidden, outputs];
  let weights = 0;
  for (let i = 1; i < sizes.length; i += 1) weights += (sizes[i - 1] + 1) * sizes[i];
  const text =
    `${inputs} ${plural(inputs, 'вход', 'входа', 'входов')} → ` +
    [...hidden.map(String), `${outputs} ${plural(outputs, 'выход', 'выхода', 'выходов')}`].join(' → ');
  return { sizes, weights, text };
}

export class ParamsForm {
  constructor(root, catalog, hooks = {}) {
    this.root = root;
    this.catalog = catalog;
    this.hooks = hooks;
    this.mode = 'new';
    this.run = null;
    this.draft = null;
    this.suggestedName = '';
    this.fields = {};
    this.rays = [];
    this.hidden = [];
    this.scalars = new Set();
    this.outputs = new Set(['steer', 'throttle']);
    this.passthrough = { init_scale: catalog.defaults.init_scale };
    this.liveTimer = null;
    this.liveDirty = new Set();
    this.pending = null;
    this.serverError = null;
    this.dirty = false;
    this.build();
    this.write(catalog.defaults);
  }

  // ---- building ---------------------------------------------------------------------------

  build() {
    const c = this.catalog;
    this.summary = h('div', { class: 'model-summary' });
    this.modeLine = h('p', { class: 'form-mode' });
    this.liveNote = h('p', { class: 'live-note', role: 'status' });
    this.lockNote = h('p', { class: 'lock-note', text: 'Структура модели, размер популяции и трасса задаются при создании запуска.' });

    const run = h('details', { class: 'group', open: true },
      h('summary', { text: 'Запуск' }),
      this.textField('name', 'Название', { maxLength: MAX_NAME }),
      this.selectField('track', 'Трасса', c.tracks.map((id) => [id, id])),
      this.sliderField('population', 'Популяция', { min: 2, max: 200, step: 1, int: true, hint: 'машинок в поколении' }),
      h('div', { class: 'field-grid' },
        this.numberField('max_steps', 'Лимит шагов', { min: 50, max: 5000, step: 50, hint: 'Шагов симуляции на заезд, 50–5000' }),
        this.numberField('ray_range', 'Дальность лучей, м', { min: 1, step: 5, hint: 'Дальше этого расстояния луч ничего не видит' }),
        this.numberField('seed', 'Seed', { min: 0, step: 1, hint: 'Одинаковый seed — одинаковый ход обучения' }),
        this.numberField('generations', 'Поколений', { min: 1, step: 1, placeholder: 'без лимита', hint: 'Пусто — обучение идёт до остановки' }),
      ),
    );

    const evolution = h('details', { class: 'group', open: true },
      h('summary', { text: 'Эволюция' }),
      this.liveNote,
      this.sliderField('elite', 'Элита', { min: 1, max: 199, step: 1, int: true, live: true, hint: 'лучших переходят без изменений' }),
      this.sliderField('mutation_rate', 'Мутация', { min: 0.01, max: 1, step: 0.01, live: true, hint: 'доля весов, меняющихся у потомка' }),
      this.sliderField('mutation_scale', 'Сила мутации', { min: 0.01, max: 2, step: 0.01, live: true, hint: 'размах изменения веса' }),
    );

    const presetOptions = [...Object.keys(c.fitness.presets).map((id) => [id, id]), ['custom', 'свой']];
    this.fields.preset = this.makeSelect('preset', presetOptions, true);
    this.fields.preset.addEventListener('change', () => this.applyPreset());
    this.termsBox = h('div', { class: 'terms' });
    for (const term of c.fitness.terms) this.termsBox.append(this.termField(term));
    const reward = h('details', { class: 'group', open: true },
      h('summary', { text: 'Награда' }),
      h('div', { class: 'field' },
        h('label', { for: this.fields.preset.id, text: 'Профиль' }),
        this.fields.preset,
        h('p', { class: 'hint', text: 'Что считать хорошей машинкой. Вес 0 — слагаемое не учитывается, минус — штраф.' }),
      ),
      this.termsBox,
      this.errorLine('fitness'),
    );

    this.inputsBox = h('div', { class: 'checks' });
    for (const item of c.inputs) {
      const box = h('input', { type: 'checkbox', id: `f${++uid}`, value: item.id });
      box.addEventListener('change', () => {
        if (box.checked) this.scalars.add(item.id);
        else this.scalars.delete(item.id);
        this.changed('model');
      });
      this.fields[`in:${item.id}`] = box;
      this.inputsBox.append(
        h('label', { class: 'check', for: box.id },
          box, h('span', { class: 'check-text' }, item.hint || item.label, h('code', { text: item.label })),
        ),
      );
    }
    this.raysBox = h('div', { class: 'rays' });
    this.presetsBox = h('div', { class: 'chips' });
    for (const preset of RAY_PRESETS) {
      this.presetsBox.append(
        h('button', {
          type: 'button', class: 'chip-btn', 'data-lockable': '', text: preset.label,
          onclick: () => {
            this.rays = preset.angles.map(String);
            this.renderRays();
            this.changed('model');
          },
        }),
      );
    }
    this.addRayBtn = h('button', {
      type: 'button', class: 'chip-btn', 'data-lockable': '', text: '+ луч',
      onclick: () => {
        this.rays.push(String(this.nextAngle()));
        this.renderRays();
        this.changed('model');
        this.raysBox.querySelector('.ray:last-child input')?.focus();
      },
    });
    this.hiddenBox = h('div', { class: 'layers' });
    this.addLayerBtn = h('button', {
      type: 'button', class: 'chip-btn', 'data-lockable': '', text: '+ слой',
      onclick: () => {
        this.hidden.push(String(this.hidden.length ? this.hidden[this.hidden.length - 1] : 8));
        this.renderHidden();
        this.changed('model');
        this.hiddenBox.querySelector('.layer:last-child input')?.focus();
      },
    });
    this.outputsBox = h('div', { class: 'checks' });
    for (const item of c.outputs) {
      const [title, note] = OUTPUT_TEXT[item.id] ?? [item.label, item.hint];
      const box = h('input', { type: 'checkbox', id: `f${++uid}`, value: item.id });
      box.addEventListener('change', () => this.toggleOutput(item.id, box.checked));
      this.fields[`out:${item.id}`] = box;
      this.outputsBox.append(
        h('label', { class: 'check', for: box.id },
          box, h('span', { class: 'check-text' }, title, h('small', { text: note })),
        ),
      );
    }
    this.fields.activation = this.makeSelect('activation', c.activations.map((a) => [a.id, a.label]));
    this.fields.activation.addEventListener('change', () => this.changed('model'));

    const model = h('details', { class: 'group', open: true },
      h('summary', { text: 'Модель' }),
      h('div', { class: 'sub' },
        h('h3', { text: 'Входы' }),
        this.inputsBox,
        h('div', { class: 'sub-rays' },
          h('h4', { text: 'Лучи дальномера' }),
          h('p', { class: 'hint', text: `Углы от −180° до 180° относительно курса, без повторов. ${c.ray?.hint ?? ''}` }),
          this.raysBox,
          h('div', { class: 'chips' }, this.addRayBtn, this.presetsBox),
        ),
        this.errorLine('inputs'),
      ),
      h('div', { class: 'sub' },
        h('h3', { text: 'Скрытые слои' }),
        h('p', { class: 'hint', text: `Число нейронов в слое: 1–${MAX_NEURONS}, слоёв до ${MAX_LAYERS}. Можно без слоёв.` }),
        this.hiddenBox,
        h('div', { class: 'chips' }, this.addLayerBtn),
      ),
      h('div', { class: 'sub' },
        h('h3', { text: 'Выходы' }),
        this.outputsBox,
        h('p', { class: 'hint', text: OUTPUT_RULES }),
        this.errorLine('outputs'),
      ),
      h('div', { class: 'field' },
        h('label', { for: this.fields.activation.id, text: 'Активация' }),
        this.fields.activation,
      ),
    );

    this.formError = h('p', { class: 'form-error', role: 'alert', hidden: true });
    this.submitBtn = h('button', { type: 'submit', class: 'btn btn-primary btn-wide', text: 'Создать запуск' });
    this.resetBtn = h('button', { type: 'button', class: 'btn btn-quiet', text: 'По умолчанию', onclick: () => this.reset() });
    this.copyBtn = h('button', {
      type: 'button', class: 'btn btn-wide', text: 'Копировать настройки текущего запуска',
      onclick: () => this.hooks.onCopy?.(this.run),
    });
    this.actions = h('div', { class: 'form-actions' }, this.formError, this.submitBtn, this.resetBtn, this.copyBtn);

    this.form = h('form', { class: 'params-form', noValidate: true, autocomplete: 'off' },
      this.modeLine, this.summary, this.lockNote, run, evolution, reward, model, this.actions);
    this.form.addEventListener('submit', (event) => {
      event.preventDefault();
      this.submit();
    });
    this.root.replaceChildren(this.form);
  }

  errorLine(key) {
    const node = h('p', { class: 'err', role: 'alert', 'data-error': key, hidden: true });
    this.fields[`err:${key}`] = node;
    return node;
  }

  field(key, label, control, hint) {
    const error = this.errorLine(key);
    return h('div', { class: 'field', 'data-field': key },
      h('label', { for: control.id || control.querySelector?.('input[type=number]')?.id, text: label }),
      control,
      hint ? h('p', { class: 'hint', text: hint }) : null,
      error,
    );
  }

  textField(key, label, { maxLength }) {
    const input = h('input', { type: 'text', id: `f${++uid}`, maxLength: maxLength + 20 });
    input.addEventListener('input', () => this.changed(key));
    this.fields[key] = input;
    return this.field(key, label, input);
  }

  makeSelect(key, options, live = false) {
    const select = h('select', { id: `f${++uid}`, 'data-live': live ? '' : null });
    for (const [value, text] of options) select.append(h('option', { value, text }));
    return select;
  }

  selectField(key, label, options) {
    const select = this.makeSelect(key, options);
    select.addEventListener('change', () => this.changed(key));
    this.fields[key] = select;
    return this.field(key, label, select);
  }

  numberField(key, label, { min, max, step, placeholder, hint, live = false }) {
    const input = h('input', {
      type: 'number', id: `f${++uid}`, min, max, step, placeholder, inputMode: 'decimal',
      'data-live': live ? '' : null,
    });
    input.addEventListener('input', () => this.changed(key));
    this.fields[key] = input;
    if (hint) input.title = hint;
    return this.field(key, label, input);
  }

  sliderField(key, label, { min, max, step, int, hint, live = false }) {
    const range = h('input', { type: 'range', min, max, step, 'aria-label': label, 'data-live': live ? '' : null });
    const input = h('input', {
      type: 'number', id: `f${++uid}`, class: 'num', min, step, inputMode: 'decimal',
      'data-live': live ? '' : null,
    });
    if (int) input.max = max;
    range.addEventListener('input', () => {
      input.value = range.value;
      this.changed(key);
    });
    input.addEventListener('input', () => {
      if (input.value !== '' && Number.isFinite(Number(input.value))) range.value = input.value;
      this.changed(key);
    });
    this.fields[key] = input;
    this.fields[`range:${key}`] = range;
    return this.field(key, label, h('div', { class: 'slide' }, range, input), hint);
  }

  termField(term) {
    const [title, explanation] = term.label.split(/:\s*/, 2);
    const range = h('input', { type: 'range', min: -1, max: 2, step: 0.05, 'aria-label': title, 'data-live': '' });
    const input = h('input', {
      type: 'number', id: `f${++uid}`, class: 'num', min: -5, max: 5, step: 0.05, inputMode: 'decimal', 'data-live': '',
    });
    range.addEventListener('input', () => {
      input.value = range.value;
      this.termChanged();
    });
    input.addEventListener('input', () => {
      if (input.value !== '' && Number.isFinite(Number(input.value))) range.value = input.value;
      this.termChanged();
    });
    this.fields[`term:${term.id}`] = input;
    this.fields[`range:term:${term.id}`] = range;
    return h('div', { class: 'field term', 'data-field': `term:${term.id}` },
      h('label', { for: input.id, text: title }),
      h('div', { class: 'slide' }, range, input),
      explanation ? h('p', { class: 'hint', text: explanation }) : null,
    );
  }

  // ---- lists ------------------------------------------------------------------------------

  nextAngle() {
    const used = new Set(this.rays.map(Number));
    for (const angle of [0, 15, -15, 45, -45, 75, -75, 105, -105, 135, -135, 165, -165, 180]) {
      if (!used.has(angle)) return angle;
    }
    return 0;
  }

  renderRays() {
    this.raysBox.replaceChildren(
      ...this.rays.map((value, index) => {
        const input = h('input', {
          type: 'number', class: 'num', value, min: -180, max: 180, step: 5, inputMode: 'decimal',
          'aria-label': `Угол луча ${index + 1}, градусы`,
        });
        input.addEventListener('input', () => {
          this.rays[index] = input.value;
          this.changed('model');
        });
        return h('div', { class: 'ray', 'data-index': index },
          input, h('span', { class: 'unit', text: '°' }),
          h('button', {
            type: 'button', class: 'icon-btn', 'data-lockable': '', 'aria-label': `Убрать луч ${index + 1}`, text: '×',
            onclick: () => {
              this.rays.splice(index, 1);
              this.renderRays();
              this.changed('model');
            },
          }),
          h('p', { class: 'err', role: 'alert', hidden: true }),
        );
      }),
    );
    if (!this.rays.length) this.raysBox.append(h('p', { class: 'hint', text: 'Лучей нет: машинка не увидит края трассы.' }));
    this.applyLock();
  }

  renderHidden() {
    this.hiddenBox.replaceChildren(
      ...this.hidden.map((value, index) => {
        const input = h('input', {
          type: 'number', class: 'num', value, min: 1, max: MAX_NEURONS, step: 1, inputMode: 'numeric',
          'aria-label': `Нейронов в слое ${index + 1}`,
        });
        input.addEventListener('input', () => {
          this.hidden[index] = input.value;
          this.changed('model');
        });
        return h('div', { class: 'layer', 'data-index': index },
          h('span', { class: 'layer-name', text: `Слой ${index + 1}` }),
          input, h('span', { class: 'unit', text: 'нейр.' }),
          h('button', {
            type: 'button', class: 'icon-btn', 'data-lockable': '', 'aria-label': `Убрать слой ${index + 1}`, text: '−',
            onclick: () => {
              this.hidden.splice(index, 1);
              this.renderHidden();
              this.changed('model');
            },
          }),
          h('p', { class: 'err', role: 'alert', hidden: true }),
        );
      }),
    );
    if (!this.hidden.length) this.hiddenBox.append(h('p', { class: 'hint', text: 'Без скрытых слоёв: входы соединены с выходами напрямую.' }));
    this.addLayerBtn.disabled = this.hidden.length >= MAX_LAYERS || this.lockedNow();
    this.applyLock();
  }

  toggleOutput(id, checked) {
    const out = this.outputs;
    if (checked) out.add(id);
    else out.delete(id);
    if (checked && id === 'throttle') {
      out.delete('accelerate');
      out.delete('brake');
    }
    if (checked && id === 'accelerate') out.delete('throttle');
    if (!checked && id === 'accelerate') out.delete('brake');
    this.syncOutputs();
    this.changed('model');
  }

  syncOutputs() {
    for (const item of this.catalog.outputs) {
      const box = this.fields[`out:${item.id}`];
      box.checked = this.outputs.has(item.id);
      box.dataset.rule = '';
      if (item.id === 'steer') {
        box.checked = true;
        box.dataset.rule = 'always';
      } else if (item.id === 'brake') {
        box.dataset.rule = this.outputs.has('accelerate') ? '' : 'needs';
      }
    }
    this.outputs.add('steer');
    this.applyLock();
  }

  // ---- reading and writing ----------------------------------------------------------------

  num(key) {
    const text = this.fields[key].value.trim();
    return text === '' ? null : Number(text);
  }

  weights() {
    const weights = {};
    for (const term of this.catalog.fitness.terms) {
      const value = this.num(`term:${term.id}`);
      if (value != null && Number.isFinite(value) && value !== 0) weights[term.id] = value;
    }
    return weights;
  }

  modelInputs() {
    const rays = this.rays.map((value) => rayInput(Number(value)));
    const scalars = this.catalog.inputs.map((i) => i.id).filter((id) => this.scalars.has(id));
    return [...rays, ...scalars];
  }

  read() {
    return {
      name: this.fields.name.value.trim(),
      track: this.fields.track.value,
      model: {
        inputs: this.modelInputs(),
        hidden: this.hidden.map(Number),
        outputs: this.catalog.outputs.map((o) => o.id).filter((id) => this.outputs.has(id)),
        activation: this.fields.activation.value,
      },
      fitness: { weights: this.weights() },
      population: this.num('population'),
      elite: this.num('elite'),
      mutation_rate: this.num('mutation_rate'),
      mutation_scale: this.num('mutation_scale'),
      init_scale: this.passthrough.init_scale,
      max_steps: this.num('max_steps'),
      ray_range: this.num('ray_range'),
      generations: this.num('generations'),
      seed: this.num('seed'),
    };
  }

  setValue(key, value) {
    this.fields[key].value = value ?? '';
    const range = this.fields[`range:${key}`];
    if (range && value != null) range.value = value;
  }

  write(config) {
    const c = { ...this.catalog.defaults, ...config };
    this.passthrough.init_scale = c.init_scale;
    this.setValue('name', c.name);
    if (!this.catalog.tracks.includes(c.track)) {
      this.fields.track.append(h('option', { value: c.track, text: c.track }));
    }
    this.fields.track.value = c.track;
    for (const key of ['population', 'max_steps', 'ray_range', 'seed', 'generations', 'mutation_rate', 'mutation_scale']) {
      this.setValue(key, c[key]);
    }
    this.fields.elite.max = Math.max(1, c.population - 1);
    this.fields['range:elite'].max = Math.max(1, c.population - 1);
    this.setValue('elite', c.elite);
    const inputs = c.model.inputs;
    this.rays = inputs.filter((n) => n.startsWith(RAY_PREFIX)).map((n) => String(parseFloat(n.slice(RAY_PREFIX.length))));
    this.scalars = new Set(inputs.filter((n) => !n.startsWith(RAY_PREFIX)));
    for (const item of this.catalog.inputs) this.fields[`in:${item.id}`].checked = this.scalars.has(item.id);
    this.hidden = c.model.hidden.map(String);
    this.outputs = new Set(c.model.outputs);
    this.fields.activation.value = c.model.activation;
    this.writeWeights(c.fitness.weights ?? c.fitness);
    this.renderRays();
    this.renderHidden();
    this.syncOutputs();
    this.refresh();
  }

  writeWeights(weights) {
    for (const term of this.catalog.fitness.terms) this.setValue(`term:${term.id}`, weights[term.id] ?? 0);
    this.syncPreset();
  }

  syncPreset() {
    const current = this.weights();
    const same = (a, b) => {
      const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
      return [...keys].every((k) => Math.abs((a[k] ?? 0) - (b[k] ?? 0)) < 1e-9);
    };
    const match = Object.entries(this.catalog.fitness.presets).find(([, spec]) => same(spec.weights, current));
    this.fields.preset.value = match ? match[0] : 'custom';
  }

  applyPreset() {
    const preset = this.catalog.fitness.presets[this.fields.preset.value];
    if (preset) this.writeWeights(preset.weights);
    this.liveDirty.add('fitness');
    this.refresh();
    this.scheduleLive();
  }

  termChanged() {
    this.syncPreset();
    this.changed('fitness');
  }

  reset() {
    const name = this.suggestedName || this.catalog.defaults.name;
    this.write({ ...this.catalog.defaults, name });
  }

  // ---- validation -------------------------------------------------------------------------

  validate(config = this.read()) {
    const errors = {};
    const intIn = (key, low, high, text) => {
      const v = config[key];
      if (v == null || !Number.isFinite(v) || !isInt(v) || v < low || (high != null && v > high)) errors[key] = text;
    };
    if (!config.name) errors.name = 'Введите название запуска.';
    else if (config.name.length > MAX_NAME) errors.name = `Не длиннее ${MAX_NAME} символов (сейчас ${config.name.length}).`;
    intIn('population', 2, 200, 'Целое число от 2 до 200.');
    const eliteMax = Number.isFinite(config.population) ? config.population - 1 : 199;
    intIn('elite', 1, eliteMax, config.elite >= config.population && Number.isFinite(config.elite)
      ? `Элита должна быть меньше популяции: от 1 до ${eliteMax}.`
      : `Целое число от 1 до ${eliteMax}.`);
    const rate = config.mutation_rate;
    if (rate == null || !Number.isFinite(rate) || rate <= 0 || rate > 1) errors.mutation_rate = 'Число больше 0 и не больше 1.';
    for (const key of ['mutation_scale', 'ray_range']) {
      const v = config[key];
      if (v == null || !Number.isFinite(v) || v <= 0) errors[key] = 'Число больше 0.';
    }
    intIn('max_steps', 50, 5000, 'Целое число от 50 до 5000.');
    intIn('seed', 0, null, 'Целое число, не меньше 0.');
    if (config.generations != null) intIn('generations', 1, null, 'Целое число от 1 или пустое поле.');
    if (!config.track) errors.track = 'Выберите трассу.';

    // model
    const rayErrors = [];
    const seen = new Set();
    this.rays.forEach((raw, index) => {
      const value = raw.trim() === '' ? NaN : Number(raw);
      if (!Number.isFinite(value)) rayErrors[index] = 'Введите угол числом.';
      else if (value < -180 || value > 180) rayErrors[index] = 'Угол от −180° до 180°.';
      else if (seen.has(value)) rayErrors[index] = 'Такой угол уже есть.';
      seen.add(value);
    });
    errors.rays = rayErrors;
    if (!this.rays.length && !this.scalars.size) errors.inputs = 'Выберите хотя бы один вход: луч или параметр.';
    const layerErrors = [];
    this.hidden.forEach((raw, index) => {
      const value = raw.trim() === '' ? NaN : Number(raw);
      if (!isInt(value) || value < 1 || value > MAX_NEURONS) layerErrors[index] = `Целое число от 1 до ${MAX_NEURONS}.`;
    });
    errors.layers = layerErrors;
    if (this.hidden.length > MAX_LAYERS) errors.inputs = errors.inputs ?? `Скрытых слоёв не больше ${MAX_LAYERS}.`;
    const out = this.outputs;
    if (!out.has('steer')) errors.outputs = 'Нужен выход «Руль».';
    else if (out.has('throttle') && (out.has('accelerate') || out.has('brake'))) {
      errors.outputs = '«Газ и тормоз» нельзя сочетать с отдельным газом или тормозом.';
    } else if (!out.has('throttle') && !out.has('accelerate')) errors.outputs = 'Нужен газ: «Газ и тормоз» либо отдельный «Газ».';
    else if (out.has('brake') && !out.has('accelerate')) errors.outputs = 'Тормоз работает только вместе с отдельным газом.';
    if (!Object.keys(config.fitness.weights).length) errors.fitness = 'Задайте вес хотя бы одного слагаемого награды.';
    const bad = this.catalog.fitness.terms.find((t) => !Number.isFinite(this.num(`term:${t.id}`) ?? 0));
    if (bad) errors.fitness = 'Веса награды должны быть числами.';
    return errors;
  }

  showErrors(errors) {
    for (const [key, node] of Object.entries(this.fields)) {
      if (!key.startsWith('err:')) continue;
      const text = errors[key.slice(4)];
      node.textContent = typeof text === 'string' ? text : '';
      node.hidden = typeof text !== 'string';
    }
    for (const [key, control] of Object.entries(this.fields)) {
      if (key.includes(':') || !(control instanceof HTMLElement)) continue;
      control.toggleAttribute('aria-invalid', Boolean(errors[key]) && typeof errors[key] === 'string');
    }
    this.raysBox.querySelectorAll('.ray').forEach((row, index) => this.rowError(row, errors.rays?.[index]));
    this.hiddenBox.querySelectorAll('.layer').forEach((row, index) => this.rowError(row, errors.layers?.[index]));
    const outputsBad = typeof errors.outputs === 'string';
    for (const item of this.catalog.outputs) this.fields[`out:${item.id}`].toggleAttribute('aria-invalid', outputsBad);
  }

  rowError(row, text) {
    const node = row.querySelector('.err');
    node.textContent = text ?? '';
    node.hidden = !text;
    row.querySelector('input').toggleAttribute('aria-invalid', Boolean(text));
  }

  hasErrors(errors) {
    return Object.values(errors).some((v) => (Array.isArray(v) ? v.some(Boolean) : Boolean(v)));
  }

  // ---- change handling --------------------------------------------------------------------

  changed(key) {
    if (key === 'population') {
      const max = Math.max(1, (this.num('population') ?? 2) - 1);
      this.fields.elite.max = max;
      this.fields['range:elite'].max = max;
    }
    this.clearServerError();
    this.dirty = true;
    this.refresh();
    if (this.mode === 'run') {
      this.liveDirty.add(key);
      this.scheduleLive();
    }
  }

  refresh() {
    const errors = this.validate();
    this.showErrors(errors);
    this.renderSummary();
    return errors;
  }


  renderSummary() {
    const inputs = this.rays.length + this.scalars.size;
    const hidden = this.hidden.map(Number).filter((n) => isInt(n) && n >= 1);
    const outputs = this.outputs.size;
    const { sizes, weights, text } = summarizeModel(inputs, hidden, outputs);
    const nodes = [];
    sizes.forEach((size, index) => {
      if (index) nodes.push(h('span', { class: 'net-arrow', 'aria-hidden': 'true', text: '→' }));
      const caption = index === 0 ? 'входы' : index === sizes.length - 1 ? 'выходы' : 'скрытый';
      nodes.push(h('span', { class: `net-node${index === 0 || index === sizes.length - 1 ? ' is-edge' : ''}` },
        h('b', { text: String(size) }), h('small', { text: caption })));
    });
    this.summary.replaceChildren(
      h('div', { class: 'net-line', role: 'img', 'aria-label': `Модель: ${text}` }, ...nodes),
      h('p', { class: 'net-text', text: `${text} · ${weights} ${plural(weights, 'вес', 'веса', 'весов')}` }),
    );
  }

  // ---- modes and locking ------------------------------------------------------------------

  lockedNow() {
    return this.mode === 'run';
  }

  liveEditable() {
    return this.mode === 'new' || (this.run && !TERMINAL.has(this.run.status));
  }

  applyLock() {
    const run = this.mode === 'run';
    const live = this.liveEditable();
    this.form.dataset.mode = this.mode;
    for (const node of this.form.querySelectorAll('input, select, button[data-lockable]')) {
      let disabled = run && !node.hasAttribute('data-live');
      if (run && node.hasAttribute('data-live') && !live) disabled = true;
      if (node.dataset.rule === 'always') disabled = true;
      if (node.dataset.rule === 'needs' && !disabled) disabled = true;
      node.disabled = disabled;
    }
    this.addLayerBtn.disabled = this.addLayerBtn.disabled || this.hidden.length >= MAX_LAYERS;
    this.addRayBtn.disabled = this.addRayBtn.disabled || false;
    this.lockNote.hidden = !run;
    this.submitBtn.hidden = run;
    this.resetBtn.hidden = run;
    this.copyBtn.hidden = !run;
    this.liveNote.hidden = !run;
  }

  showNew({ config = null, name = '' } = {}) {
    this.stashDraft();
    this.mode = 'new';
    this.run = null;
    this.clearServerError();
    this.suggestedName = name;
    this.modeLine.replaceChildren('Новый запуск');
    const base = config ?? this.draft ?? this.catalog.defaults;
    const next = { ...base };
    if (config || !this.draft || /^Запуск \d+$/.test(next.name)) next.name = name || next.name;
    this.write(next);
    this.dirty = false;
    this.root.scrollTop = 0;
  }

  showRun(run, config) {
    this.stashDraft();
    this.mode = 'run';
    this.run = run;
    this.clearServerError();
    clearTimeout(this.liveTimer);
    this.liveDirty.clear();
    this.pending = null;
    this.modeLine.replaceChildren(
      h('span', { class: 'run-swatch', style: `background:${run.color}` }), `Запуск «${run.name}»`);
    const last = run.gens[run.gens.length - 1]?.params;
    this.write({ ...config, ...(last ?? {}) });
    this.setLiveNote();
  }

  stashDraft() {
    if (this.mode === 'new' && this.dirty) this.draft = this.read();
  }

  focusName() {
    this.root.scrollTop = 0;
    this.fields.name.focus();
    this.fields.name.select();
  }

  refreshLock() {
    this.applyLock();
    this.setLiveNote();
  }

  // ---- live updates of a running run ------------------------------------------------------

  scheduleLive() {
    if (this.mode !== 'run' || !this.run) return;
    clearTimeout(this.liveTimer);
    this.liveTimer = setTimeout(() => this.flushLive(), LIVE_DELAY);
  }

  async flushLive() {
    const run = this.run;
    if (!run || !this.liveEditable() || !this.liveDirty.size) return;
    const errors = this.validate();
    const config = this.read();
    const params = {};
    const dirty = [...this.liveDirty];
    this.liveDirty.clear();
    for (const key of dirty) {
      if (key === 'fitness') {
        if (!errors.fitness) params.fitness = config.fitness;
      } else if (['elite', 'mutation_rate', 'mutation_scale'].includes(key) && !errors[key]) {
        params[key] = config[key];
      }
    }
    if (!Object.keys(params).length) return;
    const ok = await this.hooks.onLiveUpdate?.(run, params);
    if (ok && this.run === run) {
      this.pending = { gens: run.gens.length, params };
      this.setLiveNote();
    }
  }

  onGen(run) {
    if (run !== this.run) return;
    if (this.pending && run.gens.length > this.pending.gens) this.pending = null;
    this.setLiveNote();
  }

  setLiveNote() {
    const run = this.run;
    if (this.mode !== 'run' || !run) {
      this.liveNote.textContent = '';
      return;
    }
    if (TERMINAL.has(run.status)) {
      this.liveNote.textContent = 'Запуск завершён: параметры только для просмотра.';
      return;
    }
    const echo = run.gens[run.gens.length - 1]?.params;
    const applied = echo
      ? `Сервер применил в поколении ${run.gens.length}: мутация ${fmt(echo.mutation_rate)}, ` +
        `сила ${fmt(echo.mutation_scale)}, элита ${echo.elite}.`
      : 'Параметры можно менять на ходу — изменения вступят в силу со следующего поколения.';
    this.liveNote.textContent = this.pending
      ? `Отправлено, применится со следующего поколения. ${echo ? applied : ''}`.trim()
      : applied;
  }

  // ---- submit -----------------------------------------------------------------------------

  async submit() {
    if (this.mode !== 'new') return;
    const errors = this.refresh();
    if (this.hasErrors(errors)) {
      this.setFormError('Исправьте отмеченные поля.');
      this.form.querySelector('[aria-invalid]')?.focus();
      return;
    }
    this.setFormError(null);
    this.submitBtn.disabled = true;
    try {
      const result = await this.hooks.onCreate?.(this.read());
      if (result === true) this.draft = null;
    } finally {
      this.submitBtn.disabled = false;
    }
  }

  // Server error text: shown under the form, and next to the field when it names one.
  showServerError(text) {
    this.serverError = text;
    this.setFormError(text);
    const match = /поле (\w+)/.exec(text);
    if (!match) return;
    const key = match[1] === 'model' ? 'inputs' : match[1];
    const node = this.fields[`err:${key}`];
    if (node) {
      node.textContent = text.replace(/^поле \w+:\s*/, '');
      node.hidden = false;
    }
  }

  clearServerError() {
    if (!this.serverError) return;
    this.serverError = null;
    this.setFormError(null);
  }

  setFormError(text) {
    this.formError.textContent = text ?? '';
    this.formError.hidden = !text;
  }
}
