// Tiny DOM helpers: element construction, icons from the sprite, event delegation.

const SVG_NS = "http://www.w3.org/2000/svg";

function appendChildren(parent, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) appendChildren(parent, child);
    else if (child instanceof Node) parent.append(child);
    else parent.append(document.createTextNode(String(child)));
  }
}

function applyAttr(el, key, value) {
  if (value === null || value === undefined || value === false) return;
  if (key === "class" || key === "className") el.className = value;
  else if (key === "dataset") Object.assign(el.dataset, value);
  else if (key.startsWith("on") && typeof value === "function") {
    el.addEventListener(key.slice(2).toLowerCase(), value);
  } else if (value === true) el.setAttribute(key, "");
  else el.setAttribute(key, value);
}

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs === null || attrs === undefined) {
    // no attributes
  } else if (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs)) {
    children.unshift(attrs);
  } else {
    for (const [key, value] of Object.entries(attrs)) applyAttr(el, key, value);
  }
  appendChildren(el, children);
  return el;
}

export function icon(name, { size } = {}) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  if (size !== undefined && size !== null) {
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
  }
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `/static/icons.svg#i-${name}`);
  svg.append(use);
  return svg;
}

// Returns an unsubscribe function.
export function on(root, type, selector, handler) {
  const listener = (event) => {
    if (!(event.target instanceof Element)) return;
    const match = event.target.closest(selector);
    if (match && root.contains(match)) handler(event, match);
  };
  root.addEventListener(type, listener);
  return () => root.removeEventListener(type, listener);
}
