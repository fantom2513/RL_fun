// <lab-stepper>: -/+ control with optional bounds.
import { h, icon, on } from "../core/dom.js";
import { number } from "../core/format.js";

function parseNumber(text, fallback) {
  if (text === null || text.trim() === "") return fallback;
  const n = Number(text);
  return Number.isFinite(n) ? n : fallback;
}

class LabStepper extends HTMLElement {
  static get observedAttributes() {
    return ["min", "max", "step", "value", "label", "disabled"];
  }

  constructor() {
    super();
    this._refs = null;
    this._quiet = false;
    this._bound = false;
  }

  connectedCallback() {
    if (!this._bound) {
      this._bound = true;
      on(this, "click", "button[data-act]", (e, btn) => {
        this._step(btn.dataset.act === "inc" ? 1 : -1);
      });
    }
    this.render();
  }

  attributeChangedCallback(name) {
    if (this._quiet || !this.isConnected) return;
    if (name === "value" && this._refs) this._sync();
    else this.render();
  }

  get value() {
    return this._current();
  }

  set value(v) {
    this._setAttr("value", String(this._normalize(Number(v))));
    this._sync();
  }

  _setAttr(name, value) {
    this._quiet = true;
    this.setAttribute(name, value);
    this._quiet = false;
  }

  _bounds() {
    return {
      min: parseNumber(this.getAttribute("min"), null),
      max: parseNumber(this.getAttribute("max"), null),
    };
  }

  _normalize(raw) {
    const { min, max } = this._bounds();
    let v = Number.isFinite(raw) ? raw : (min ?? 0);
    if (min !== null) v = Math.max(min, v);
    if (max !== null) v = Math.min(max, v);
    return parseFloat(v.toPrecision(12));
  }

  _current() {
    const { min } = this._bounds();
    return this._normalize(parseNumber(this.getAttribute("value"), min ?? 0));
  }

  _step(dir) {
    if (this.hasAttribute("disabled")) return;
    const step = parseNumber(this.getAttribute("step"), 1);
    const cur = this._current();
    const next = this._normalize(cur + dir * step);
    if (next === cur) return;
    this._setAttr("value", String(next));
    this._sync();
    this.dispatchEvent(new CustomEvent("stepper-change", { bubbles: true, detail: { value: next } }));
  }

  _sync() {
    const r = this._refs;
    if (!r) return;
    const v = this._current();
    const { min, max } = this._bounds();
    const off = this.hasAttribute("disabled");
    r.out.textContent = number(v, "auto");
    r.dec.disabled = off || (min !== null && v <= min);
    r.inc.disabled = off || (max !== null && v >= max);
  }

  render() {
    const dec = h("button", { type: "button", "aria-label": "Меньше", "data-act": "dec" }, icon("minus"));
    const inc = h("button", { type: "button", "aria-label": "Больше", "data-act": "inc" }, icon("plus"));
    const out = h("output", { "aria-live": "polite" });
    this.replaceChildren(h("div", {
      class: "stepper",
      role: "group",
      "aria-label": this.getAttribute("label"),
    }, dec, out, inc));
    this._refs = { dec, out, inc };
    this._sync();
  }
}

customElements.define("lab-stepper", LabStepper);
