// Delegated tooltips for elements with data-tip (text), data-tip-key (kbd) and data-tip-side.

import { h } from '../core/dom.js';

const SHOW_DELAY_MS = 500;
const GAP = 8;
const MARGIN = 8;
const TIP_ID = 'lab-tooltip';

let initialized = false;
let tipEl = null;
let current = null; // target whose tooltip is shown
let pending = null; // target waiting for the show delay
let timer = null;

function tipTarget(node) {
  return node?.closest?.('[data-tip]') ?? null;
}

function ensureTip() {
  if (!tipEl) {
    tipEl = h('div', { class: 'tooltip', role: 'tooltip', id: TIP_ID });
    document.body.append(tipEl);
  }
  return tipEl;
}

function hideNow() {
  clearTimeout(timer);
  timer = null;
  pending = null;
  if (current) current.removeAttribute('aria-describedby');
  current = null;
  if (tipEl) tipEl.style.display = 'none';
}

function place(target, el) {
  const side = ['right', 'top'].includes(target.dataset.tipSide) ? target.dataset.tipSide : 'bottom';
  const rect = target.getBoundingClientRect();
  const width = el.offsetWidth;
  const height = el.offsetHeight;

  let left;
  let top;
  if (side === 'right') {
    left = rect.right + GAP;
    top = rect.top + rect.height / 2 - height / 2;
  } else if (side === 'top') {
    left = rect.left + rect.width / 2 - width / 2;
    top = rect.top - GAP - height;
  } else {
    left = rect.left + rect.width / 2 - width / 2;
    top = rect.bottom + GAP;
  }

  // Keep the tooltip inside the viewport (client coordinates) before adding the scroll offset.
  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const viewportHeight = window.innerHeight;
  left = Math.max(MARGIN, Math.min(left, viewportWidth - width - MARGIN));
  top = Math.max(MARGIN, Math.min(top, viewportHeight - height - MARGIN));

  el.style.left = `${left + window.scrollX}px`;
  el.style.top = `${top + window.scrollY}px`;
}

function show(target) {
  hideNow();
  const el = ensureTip();
  const kbdKey = target.dataset.tipKey;
  const parts = [document.createTextNode(target.dataset.tip ?? '')];
  if (kbdKey) parts.push(h('kbd', { class: 'kbd' }, kbdKey));
  el.replaceChildren(...parts);

  current = target;
  target.setAttribute('aria-describedby', TIP_ID);
  el.style.display = '';
  place(target, el);
}

function scheduleShow(target) {
  if (target === pending || target === current) return;
  hideNow();
  pending = target;
  timer = setTimeout(() => {
    timer = null;
    if (target.isConnected) show(target);
  }, SHOW_DELAY_MS);
}

export function initTooltips() {
  if (initialized) return;
  initialized = true;

  document.addEventListener('mouseover', (event) => {
    const target = tipTarget(event.target);
    if (target) scheduleShow(target);
  });

  document.addEventListener('mouseout', (event) => {
    // Moving between children of the same target keeps the tooltip.
    const from = tipTarget(event.target);
    if (from && from === tipTarget(event.relatedTarget)) return;
    hideNow();
  });

  document.addEventListener('focusin', (event) => {
    const target = tipTarget(event.target);
    if (!target) return;
    let keyboard = false;
    try {
      keyboard = event.target.matches(':focus-visible');
    } catch {
      keyboard = false;
    }
    if (keyboard) show(target);
  });

  document.addEventListener('focusout', hideNow);
  document.addEventListener('pointerdown', hideNow);
  document.addEventListener('scroll', hideNow, true);
  document.addEventListener('keydown', (event) => {
    // Escape only hides the tooltip; it does not prevent other handlers.
    if (event.key === 'Escape') hideNow();
  });
}
