// <lab-param>: slider row with a synced numeric field, reset button, hint and error.
import { h, icon } from "../core/dom.js";
import { number, logScale } from "../core/format.js";

let uid = 0;

function parseNumber(text, fallback) {
  if (text === null || text.trim() === "") return fallback;
  const n = Number(text);
  return Number.isFinite(n) ? n : fallback;
}

// Russian input: comma decimal, U+2212 minus, any whitespace (incl. nbsp) ignored.
function parseRu(text) {
  const cleaned = text.replace(/[\s  ]/g, "").replace(/−/g, "-").replace(",", ".");
  return cleaned === "" ? NaN : Number(cleaned);
}

class LabParam extends HTMLElement {
  static get observedAttributes() {
    return ["label", "hint", "min", "max", "step", "value", "default", "unit",
      "digits", "scale", "live", "disabled", "error", "compact"];
  }

  constructor() {
    super();
    this._id = `lab-param-${++uid}`;
    this._refs = null;
    this._quiet = false;
  }

  connectedCallback() {
    this.render();
  }

  attributeChangedCallback(name) {
    if (this._quiet || !this.isConnected) return;
    // Plain value changes patch the DOM so a focused field keeps focus.
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
    const min = parseNumber(this.getAttribute("min"), 0);
    const max = parseNumber(this.getAttribute("max"), 1);
    return { min, max: max > min ? max : min + 1 };
  }

  _isLog() {
    return this.getAttribute("scale") === "log" && this._bounds().min > 0;
  }

  _normalize(raw) {
    const { min, max } = this._bounds();
    let v = Number.isFinite(raw) ? raw : min;
    const step = parseNumber(this.getAttribute("step"), 0);
    if (!this._isLog() && step > 0) v = min + Math.round((v - min) / step) * step;
    v = Math.min(max, Math.max(min, v));
    return parseFloat(v.toPrecision(12));
  }

  _fromPos(pos) {
    const { min, max } = this._bounds();
    const v = this._isLog()
      ? logScale(min, max).toValue(pos)
      : min + pos * (max - min);
    return this._normalize(v);
  }

  _toPos(v) {
    const { min, max } = this._bounds();
    const p = this._isLog()
      ? logScale(min, max).toPosition(v)
      : (v - min) / (max - min);
    return Math.min(1, Math.max(0, p));
  }

  _current() {
    const { min } = this._bounds();
    const fallback = parseNumber(this.getAttribute("default"), min);
    return this._normalize(parseNumber(this.getAttribute("value"), fallback));
  }

  _default() {
    const d = parseNumber(this.getAttribute("default"), null);
    return d === null ? null : this._normalize(d);
  }

  _digits() {
    const d = this.getAttribute("digits");
    if (d === "auto") return "auto";
    return Math.min(20, Math.max(0, Math.trunc(parseNumber(d, 2))));
  }

  _sync() {
    const r = this._refs;
    if (!r) return;
    const v = this._current();
    const def = this._default();
    const changed = def !== null && v !== def;
    const pos = this._toPos(v);
    r.text.value = number(v, this._digits());
    r.range.value = String(pos);
    r.range.style.setProperty("--p", `${pos * 100}%`);
    r.root.classList.toggle("is-changed", changed);
    r.reset.hidden = !changed;
  }

  _userSet(raw) {
    const v = this._normalize(raw);
    this._setAttr("value", String(v));
    this._sync();
    const def = this._default();
    this.dispatchEvent(new CustomEvent("param-change", {
      bubbles: true,
      detail: { value: v, changed: def !== null && v !== def },
    }));
  }

  render() {
    const label = this.getAttribute("label") ?? "";
    const hint = this.getAttribute("hint") ?? "";
    const unit = this.getAttribute("unit");
    const error = this.getAttribute("error");
    const disabled = this.hasAttribute("disabled");
    const live = this.hasAttribute("live");
    const id = this._id;

    const text = h("input", {
      id,
      inputmode: "decimal",
      "aria-label": `${label}, значение`,
      "aria-invalid": error ? "true" : null,
      disabled,
    });
    const range = h("input", {
      class: "range param-range",
      type: "range",
      min: 0,
      max: 1,
      step: "any",
      "aria-label": label,
      disabled,
    });
    const reset = h("button", {
      class: "btn btn-icon btn-sm param-reset",
      type: "button",
      "aria-label": "Вернуть",
      disabled,
      onclick: () => {
        const d = this._default();
        if (d !== null) this._userSet(d);
      },
    }, icon("rotate-ccw"));

    const classes = ["param", this.hasAttribute("compact") && "param-compact", error && "is-error"]
      .filter(Boolean)
      .join(" ");

    const root = h("div", { class: classes },
      h("label", { class: "param-label", for: id },
        label,
        hint ? [" ", h("span", { class: "help", title: hint }, icon("info"))] : null),
      h("span", { class: "param-value" },
        h("span", { class: "value-field" },
          text,
          unit ? h("span", { class: "unit" }, unit) : null),
        reset),
      range,
      hint || live
        ? h("div", { class: "param-hint" },
          hint,
          live ? [" ", h("span", { class: "tag tag-live" }, "на ходу")] : null)
        : null,
      error ? h("div", { class: "param-error", role: "alert" }, error) : null,
    );

    this.replaceChildren(root);
    this._refs = { root, text, range, reset };

    text.addEventListener("change", () => {
      const n = parseRu(text.value);
      if (Number.isFinite(n)) this._userSet(n);
      else this._sync();
    });
    range.addEventListener("input", () => this._userSet(this._fromPos(Number(range.value))));

    this._sync();
  }
}

customElements.define("lab-param", LabParam);
