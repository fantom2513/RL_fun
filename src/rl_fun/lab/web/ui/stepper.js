// <lab-stepper value="6" min="1" max="64" label="Нейронов в слое 1"></lab-stepper>
// Light-DOM "− N +" counter: buttons are disabled at the limits, arrow keys change the value,
// fires "change" with event.detail.value.
import { icon } from '../core/dom.js';

export class LabStepper extends HTMLElement {
  static observedAttributes = ['value', 'min', 'max', 'disabled'];

  connectedCallback() {
    if (!this.minus) this.render();
    this.sync();
  }

  attributeChangedCallback() {
    if (this.minus) this.sync();
  }

  render() {
    const label = this.getAttribute('label') ?? 'Значение';
    this.classList.add('stepper');
    this.setAttribute('role', 'group');
    this.setAttribute('aria-label', label);
    this.minus = document.createElement('button');
    this.plus = document.createElement('button');
    this.output = document.createElement('output');
    this.output.setAttribute('aria-live', 'polite');
    for (const [button, name, text, delta] of [[this.minus, 'minus', 'Меньше', -1], [this.plus, 'plus', 'Больше', 1]]) {
      button.type = 'button';
      button.setAttribute('aria-label', `${text}: ${label}`);
      button.append(icon(name));
      button.addEventListener('click', () => this.stepBy(delta));
    }
    this.replaceChildren(this.minus, this.output, this.plus);
    this.addEventListener('keydown', (event) => {
      const delta = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[event.key];
      if (delta) {
        event.preventDefault();
        this.stepBy(delta);
      }
    });
  }

  get min() {
    return Number(this.getAttribute('min') ?? -Infinity);
  }

  get max() {
    return Number(this.getAttribute('max') ?? Infinity);
  }

  get value() {
    return Number(this.getAttribute('value') ?? 0);
  }

  set value(value) {
    this.setAttribute('value', String(value));
  }

  stepBy(delta) {
    const step = Number(this.getAttribute('step') ?? 1);
    const next = Math.min(this.max, Math.max(this.min, this.value + delta * step));
    if (next === this.value || this.hasAttribute('disabled')) return;
    this.value = next;
    this.dispatchEvent(new CustomEvent('change', { bubbles: true, detail: { value: next } }));
  }

  sync() {
    const disabled = this.hasAttribute('disabled');
    this.output.textContent = String(this.value);
    this.minus.disabled = disabled || this.value <= this.min;
    this.plus.disabled = disabled || this.value >= this.max;
  }
}

if (!customElements.get('lab-stepper')) customElements.define('lab-stepper', LabStepper);
