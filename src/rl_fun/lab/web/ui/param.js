// <lab-param label="Мутация" hint="…" min="0" max="1" step="0.01" value="0.2" default="0.1"
//            kind="float|int|log" digits="2" unit="шагов" live compact disabled
//            error="текст ошибки" applying></lab-param>
// Light-DOM slider row (DESIGN.md §8 "ползунок с полем"): label with help, number field with unit,
// "вернуть" when the value differs from the default, range (log scale with ticks), hint,
// error, "на ходу" tag and a spinner while a live change is applied.
// Fires "input" while dragging/typing and "change" when settled; detail.value is the number.
import { h, icon, uniqueId } from '../core/dom.js';
import { logScale, number, parse, powerLabel } from '../core/format.js';

const LOG_STEPS = 1000;

export class LabParam extends HTMLElement {
  static observedAttributes = ['value', 'error', 'disabled', 'applying', 'default'];

  connectedCallback() {
    if (!this.range) this.render();
    this.sync();
  }

  attributeChangedCallback() {
    if (this.range) this.sync();
  }

  get kind() {
    return this.getAttribute('kind') ?? 'float';
  }

  get min() {
    return Number(this.getAttribute('min') ?? 0);
  }

  get max() {
    return Number(this.getAttribute('max') ?? 1);
  }

  get value() {
    return Number(this.getAttribute('value') ?? this.min);
  }

  set value(value) {
    this.setAttribute('value', String(value));
  }

  get digits() {
    if (this.hasAttribute('digits')) return Number(this.getAttribute('digits'));
    return this.kind === 'int' ? 0 : 'auto';
  }

  render() {
    const id = uniqueId('param');
    const label = this.getAttribute('label') ?? '';
    const hint = this.getAttribute('hint') ?? '';
    this.classList.add('param');
    if (this.hasAttribute('compact')) this.classList.add('param-compact');
    this.scale = this.kind === 'log' ? logScale(this.min, this.max) : null;

    this.field = h('input', { id: `${id}-field`, inputmode: 'decimal', autocomplete: 'off', 'aria-label': label, 'aria-describedby': `${id}-hint` });
    this.reset = h('button', { class: 'btn btn-sm btn-ghost btn-icon param-reset', type: 'button', 'aria-label': `Вернуть значение шаблона: ${label}`, title: 'Вернуть значение шаблона' }, icon('rotate-ccw'));
    this.spinner = h('span', { class: 'spinner', role: 'status', 'aria-label': 'Применяется' });
    this.range = h('input', { class: 'range', type: 'range', 'aria-labelledby': `${id}-label`, 'aria-describedby': `${id}-hint` });
    if (this.scale) {
      Object.assign(this.range, { min: 0, max: LOG_STEPS, step: 1 });
    } else {
      Object.assign(this.range, { min: this.min, max: this.max, step: this.getAttribute('step') ?? (this.kind === 'int' ? 1 : 'any') });
    }
    const unit = this.getAttribute('unit');
    const valueField = h('span', { class: 'value-field' }, this.field, unit && h('span', { class: 'unit' }, unit));
    this.valueField = valueField;
    this.errorText = h('span', { class: 'field-error param-error', id: `${id}-error` });
    this.hintText = h('span', { class: 'param-hint', id: `${id}-hint` },
      this.hasAttribute('live') && h('span', { class: 'tag tag-live' }, icon('zap'), 'на ходу'),
      hint);

    const labelChildren = [h('span', { id: `${id}-label` }, label)];
    if (hint) labelChildren.push(h('span', { class: 'help', title: hint }, icon('circle-help')));
    const ticks = this.scale
      ? h('div', { class: 'range-scale', 'aria-hidden': 'true' }, this.scale.ticks().map((t) => h('span', {}, powerLabel(t))))
      : null;

    this.replaceChildren(
      h('label', { class: 'param-label', for: `${id}-field` }, labelChildren),
      h('span', { class: 'param-value' }, this.spinner, this.reset, valueField),
      h('div', { class: 'param-range' }, this.range, ticks),
      this.hintText,
      this.errorText,
    );

    this.range.addEventListener('input', () => this.commit(this.fromRange(), 'input'));
    this.range.addEventListener('change', () => this.commit(this.fromRange(), 'change'));
    this.field.addEventListener('change', () => {
      const typed = parse(this.field.value);
      if (Number.isFinite(typed)) this.commit(this.clamp(typed), 'change');
      else this.sync();
    });
    this.field.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') this.field.dispatchEvent(new Event('change'));
    });
    this.reset.addEventListener('click', () => this.commit(Number(this.getAttribute('default')), 'change'));
  }

  clamp(value) {
    const v = Math.min(this.max, Math.max(this.min, value));
    return this.kind === 'int' ? Math.round(v) : v;
  }

  fromRange() {
    const raw = Number(this.range.value);
    return this.clamp(this.scale ? Number(this.scale.toValue(raw / LOG_STEPS).toPrecision(2)) : raw);
  }

  commit(value, type) {
    if (this.hasAttribute('disabled')) return;
    this.value = value;
    this.dispatchEvent(new CustomEvent(type, { bubbles: true, detail: { value } }));
  }

  sync() {
    const value = this.value;
    const position = this.scale ? this.scale.toPosition(value) : (value - this.min) / (this.max - this.min || 1);
    if (document.activeElement !== this.range) this.range.value = String(this.scale ? Math.round(position * LOG_STEPS) : value);
    this.range.style.setProperty('--p', `${Math.round(position * 1000) / 10}%`);
    if (document.activeElement !== this.field || !this.field.matches(':focus')) {
      this.field.value = this.kind === 'log' ? number(value, 'auto') : number(value, this.digits);
    }
    const hasDefault = this.hasAttribute('default');
    const changed = hasDefault && Math.abs(Number(this.getAttribute('default')) - value) > 1e-12;
    this.classList.toggle('is-changed', changed);
    this.reset.hidden = !changed;

    const error = this.getAttribute('error');
    this.errorText.hidden = !error;
    this.hintText.hidden = Boolean(error);
    this.errorText.replaceChildren(...(error ? [icon('triangle-alert'), document.createTextNode(error)] : []));
    this.field.setAttribute('aria-invalid', String(Boolean(error)));
    this.field.setAttribute('aria-describedby', this.errorText.id && error ? this.errorText.id : this.hintText.id);

    const disabled = this.hasAttribute('disabled');
    this.classList.toggle('is-disabled', disabled);
    this.range.disabled = disabled;
    this.field.disabled = disabled;
    this.reset.disabled = disabled;
    this.spinner.hidden = !this.hasAttribute('applying');
  }
}

if (!customElements.get('lab-param')) customElements.define('lab-param', LabParam);
