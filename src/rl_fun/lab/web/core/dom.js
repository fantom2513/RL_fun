// Small DOM helpers (DESIGN.md §11.1): h() builds elements, icon() references the sprite,
// on() delegates events.

export const SVG_NS = 'http://www.w3.org/2000/svg';
export const ICON_SPRITE = '/static/icons.svg';

// h('button', { class: 'btn', onclick: fn, 'aria-label': 'Пауза' }, icon('pause'), 'Пауза')
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs ?? {})) {
    if (value === false || value === null || value === undefined) continue;
    if (name.startsWith('on') && typeof value === 'function') {
      el.addEventListener(name.slice(2), value);
    } else if (name === 'style' && typeof value === 'object') {
      for (const [prop, v] of Object.entries(value)) el.style.setProperty(prop, v);
    } else if (name === 'dataset' && typeof value === 'object') {
      Object.assign(el.dataset, value);
    } else if (name in el && !name.includes('-') && name !== 'list' && typeof value !== 'string') {
      el[name] = value;
    } else {
      el.setAttribute(name, value === true ? '' : String(value));
    }
  }
  append(el, children);
  return el;
}

export function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return parent;
}

// <svg class="icon"><use href="/static/icons.svg#i-NAME"/></svg>; decorative unless a label is given.
export function icon(name, { label = '', className = '' } = {}) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', `icon${className ? ` ${className}` : ''}`);
  svg.setAttribute('viewBox', '0 0 24 24');
  if (label) {
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', label);
  } else {
    svg.setAttribute('aria-hidden', 'true');
  }
  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', `${ICON_SPRITE}#i-${name}`);
  svg.append(use);
  return svg;
}

// Replace placeholders <i data-icon="NAME"></i> in markup with sprite icons.
export function hydrateIcons(root = document) {
  for (const slot of root.querySelectorAll('i[data-icon]')) {
    const svg = icon(slot.dataset.icon, { label: slot.getAttribute('aria-label') ?? '' });
    for (const cls of slot.classList) svg.classList.add(cls);
    slot.replaceWith(svg);
  }
}

// Delegated listener: on(list, 'click', 'button[data-id]', (event, button) => …)
export function on(root, type, selector, handler, options) {
  const listener = (event) => {
    const target = event.target instanceof Element ? event.target.closest(selector) : null;
    if (target && root.contains(target)) handler(event, target);
  };
  root.addEventListener(type, listener, options);
  return () => root.removeEventListener(type, listener, options);
}

let uid = 0;
export function uniqueId(prefix = 'lab') {
  uid += 1;
  return `${prefix}-${uid}`;
}
