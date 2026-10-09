// <lab-toast-host>: stack of transient messages. Error toasts stay until closed.
import { h, icon } from "../core/dom.js";

const ICONS = { success: "circle-check", error: "triangle-alert", info: "info" };

class LabToastHost extends HTMLElement {
  _stack() {
    let stack = this.querySelector(":scope > .toasts");
    if (!stack) {
      stack = h("div", { class: "toasts", "aria-live": "polite" });
      this.append(stack);
    }
    return stack;
  }

  connectedCallback() {
    this._stack();
  }

  show({ text, kind = "success", action, timeout = 5000 } = {}) {
    const isError = kind === "error";
    const timed = !isError && Number.isFinite(timeout) && timeout > 0;

    let remaining = timeout;
    let timer = null;
    let started = 0;
    let hovered = false;
    let focused = false;

    const dismiss = () => {
      clearTimeout(timer);
      timer = null;
      toast.remove();
    };
    const pause = () => {
      if (timer === null) return;
      clearTimeout(timer);
      timer = null;
      remaining -= Date.now() - started;
    };
    const resume = () => {
      if (!timed || timer !== null || hovered || focused) return;
      started = Date.now();
      timer = setTimeout(dismiss, Math.max(0, remaining));
    };

    const toast = h("div", {
      class: isError ? "toast toast-error" : "toast",
      role: isError ? "alert" : "status",
    },
      icon(ICONS[kind] ?? ICONS.info),
      h("span", { class: "toast-text" }, text),
      action
        ? h("button", {
          class: "btn btn-ghost btn-sm",
          type: "button",
          onclick: () => {
            action.onClick?.();
            dismiss();
          },
        }, action.label)
        : null,
      isError
        ? h("button", {
          class: "btn btn-icon btn-sm",
          type: "button",
          "aria-label": "Закрыть",
          onclick: () => dismiss(),
        }, icon("x"))
        : null,
    );

    toast.addEventListener("mouseenter", () => {
      hovered = true;
      pause();
    });
    toast.addEventListener("mouseleave", () => {
      hovered = false;
      resume();
    });
    toast.addEventListener("focusin", () => {
      focused = true;
      pause();
    });
    toast.addEventListener("focusout", (e) => {
      focused = toast.contains(e.relatedTarget);
      resume();
    });

    this._stack().append(toast);
    resume();
    return toast;
  }
}

customElements.define("lab-toast-host", LabToastHost);

export function toast(options) {
  let host = document.querySelector("lab-toast-host");
  if (!host) {
    host = document.createElement("lab-toast-host");
    document.body.append(host);
  }
  return host.show(options);
}
