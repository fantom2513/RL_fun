// <lab-seg>: segmented radio group with roving tabindex and arrow-key selection.
import { h, icon, on } from "../core/dom.js";

const KEY_STEP = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 };

function parseOptions(text) {
  try {
    const list = JSON.parse(text ?? "[]");
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

class LabSeg extends HTMLElement {
  static get observedAttributes() {
    return ["options", "value", "size", "mono", "glass", "disabled", "label"];
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
      on(this, "click", 'button[role="radio"]', (e, btn) => {
        if (!this.hasAttribute("disabled")) this._select(btn.dataset.value, true);
      });
      on(this, "keydown", 'button[role="radio"]', (e, btn) => this._onKey(e, btn));
    }
    this.render();
  }

  attributeChangedCallback(name) {
    if (this._quiet || !this.isConnected) return;
    if (name === "value" && this._refs) this._sync();
    else this.render();
  }

  get value() {
    const idx = this._selectedIndex();
    return idx < 0 ? null : this._options()[idx].value;
  }

  set value(v) {
    this._setAttr("value", v === null || v === undefined ? null : String(v));
    this._sync();
  }

  _setAttr(name, value) {
    this._quiet = true;
    if (value === null) this.removeAttribute(name);
    else this.setAttribute(name, value);
    this._quiet = false;
  }

  _options() {
    return parseOptions(this.getAttribute("options"));
  }

  _selectedIndex() {
    const v = this.getAttribute("value");
    if (v === null) return -1;
    return this._options().findIndex((o) => String(o.value) === v);
  }

  _select(raw, emit) {
    const opt = this._options().find((o) => String(o.value) === raw);
    if (!opt) return;
    const next = String(opt.value);
    const changed = next !== this.getAttribute("value");
    this._setAttr("value", next);
    this._sync();
    if (emit && changed) {
      this.dispatchEvent(new CustomEvent("seg-change", { bubbles: true, detail: { value: opt.value } }));
    }
  }

  _onKey(e, btn) {
    const step = KEY_STEP[e.key];
    if (!step || this.hasAttribute("disabled")) return;
    e.preventDefault();
    const buttons = [...this._refs.querySelectorAll('[role="radio"]')];
    const next = buttons[(buttons.indexOf(btn) + step + buttons.length) % buttons.length];
    next.focus();
    this._select(next.dataset.value, true);
  }

  _sync() {
    const root = this._refs;
    if (!root) return;
    const idx = this._selectedIndex();
    root.querySelectorAll('[role="radio"]').forEach((btn, i) => {
      const isSel = i === idx;
      btn.setAttribute("aria-checked", String(isSel));
      btn.tabIndex = isSel || (idx < 0 && i === 0) ? 0 : -1;
    });
  }

  render() {
    const size = this.getAttribute("size");
    const disabled = this.hasAttribute("disabled");
    const classes = [
      "seg",
      size === "sm" && "seg-sm",
      this.hasAttribute("mono") && "seg-mono",
      this.hasAttribute("glass") && "seg-glass",
    ].filter(Boolean).join(" ");

    const root = h("div", {
      class: classes,
      role: "radiogroup",
      "aria-label": this.getAttribute("label"),
    }, this._options().map((o) => h("button", {
      type: "button",
      role: "radio",
      "data-value": String(o.value),
      title: o.title,
      disabled,
    }, o.icon ? icon(o.icon) : null, o.label ?? "")));

    this.replaceChildren(root);
    this._refs = root;
    this._sync();
  }
}

customElements.define("lab-seg", LabSeg);
