// <lab-seg value="a" label="Показатель"> <button value="a">…</button> … </lab-seg>
// Light-DOM segmented control: one pressed button (aria-pressed), arrow keys move the choice,
// fires "change" with event.detail.value. Variants by class: seg-sm, seg-mono, seg-glass.

export class LabSeg extends HTMLElement {
  static observedAttributes = ['value', 'disabled'];

  connectedCallback() {
    this.classList.add('seg');
    this.setAttribute('role', 'group');
    if (this.hasAttribute('label')) this.setAttribute('aria-label', this.getAttribute('label'));
    for (const button of this.buttons) button.type = 'button';
    if (!this.hasAttribute('value') && this.buttons[0]) this.setAttribute('value', this.buttons[0].value);
    this.sync();
    if (this.wired) return;
    this.wired = true;
    this.addEventListener('click', (event) => {
      const button = event.target.closest('button');
      if (button && this.contains(button) && !button.disabled) this.choose(button.value);
    });
    this.addEventListener('keydown', (event) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
      if (!step) return;
      const enabled = this.buttons.filter((b) => !b.disabled);
      const index = enabled.findIndex((b) => b.value === this.value);
      const next = enabled[(index + step + enabled.length) % enabled.length];
      if (!next) return;
      event.preventDefault();
      this.choose(next.value);
      next.focus();
    });
  }

  attributeChangedCallback() {
    this.sync();
  }

  get buttons() {
    return [...this.querySelectorAll(':scope > button')];
  }

  get value() {
    return this.getAttribute('value') ?? '';
  }

  set value(value) {
    this.setAttribute('value', String(value));
  }

  choose(value) {
    if (value === this.value) return;
    this.value = value;
    this.dispatchEvent(new CustomEvent('change', { bubbles: true, detail: { value } }));
  }

  sync() {
    const disabled = this.hasAttribute('disabled');
    for (const button of this.buttons) {
      button.setAttribute('aria-pressed', String(button.value === this.value));
      if (disabled) button.disabled = true;
    }
  }
}

if (!customElements.get('lab-seg')) customElements.define('lab-seg', LabSeg);
