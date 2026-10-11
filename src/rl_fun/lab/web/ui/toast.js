// <lab-toast-host></lab-toast-host> and toast({ text, kind, action }).
// Success and info toasts leave after 5 s (the timer pauses while hovered or focused);
// error toasts stay until closed. kind: 'success' | 'info' | 'error'.
import { h, icon } from '../core/dom.js';

const ICONS = { success: 'circle-check', info: 'info', error: 'triangle-alert' };
export const TOAST_MS = 5000;

export class LabToastHost extends HTMLElement {
  connectedCallback() {
    this.classList.add('toasts');
    this.setAttribute('role', 'region');
    this.setAttribute('aria-label', 'Уведомления');
    this.setAttribute('aria-live', 'polite');
  }

  show({ text, kind = 'success', action = null, timeout = kind === 'error' ? 0 : TOAST_MS }) {
    const close = () => this.dismiss(item);
    const item = h(
      'div',
      { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
      icon(ICONS[kind] ?? ICONS.info),
      h('span', { class: 'toast-text' }, text),
      action &&
        h('button', {
          class: 'btn btn-sm btn-ghost',
          type: 'button',
          onclick: () => {
            action.run?.();
            close();
          },
        }, action.label),
      h('button', { class: 'btn btn-sm btn-ghost btn-icon', type: 'button', 'aria-label': 'Закрыть', onclick: close }, icon('x')),
    );
    this.append(item);
    if (timeout > 0) this.schedule(item, timeout);
    return item;
  }

  schedule(item, timeout) {
    let left = timeout;
    let started = performance.now();
    let timer = setTimeout(() => this.dismiss(item), left);
    const pause = () => {
      clearTimeout(timer);
      left -= performance.now() - started;
    };
    const resume = () => {
      started = performance.now();
      timer = setTimeout(() => this.dismiss(item), Math.max(800, left));
    };
    item.addEventListener('mouseenter', pause);
    item.addEventListener('mouseleave', resume);
    item.addEventListener('focusin', pause);
    item.addEventListener('focusout', resume);
  }

  dismiss(item) {
    if (!item.isConnected || item.classList.contains('is-leaving')) return;
    item.classList.add('is-leaving');
    const done = () => item.remove();
    item.addEventListener('transitionend', done, { once: true });
    setTimeout(done, 400);
  }
}

if (!customElements.get('lab-toast-host')) customElements.define('lab-toast-host', LabToastHost);

export function toast(options) {
  let host = document.querySelector('lab-toast-host');
  if (!host) {
    host = document.createElement('lab-toast-host');
    document.body.append(host);
  }
  return host.show(typeof options === 'string' ? { text: options } : options);
}
