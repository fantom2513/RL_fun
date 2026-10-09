// Global keyboard shortcuts: 1-5 switch sections, T toggles the theme.

import { SECTIONS } from './router.js';

const SECTION_KEY = /^(?:Digit|Numpad)([1-5])$/;
const TYPING_TARGETS = 'input, select, textarea, [contenteditable], [role="slider"]';

function hasModifier(event) {
  return Boolean(event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.isComposing);
}

export function sectionForKey(event) {
  const match = SECTION_KEY.exec(event?.code ?? '');
  if (!match || hasModifier(event)) return null;
  return SECTIONS[Number(match[1]) - 1];
}

export function isThemeKey(event) {
  return event?.code === 'KeyT' && !hasModifier(event);
}

export function isTypingTarget(target) {
  return Boolean(target?.closest?.(TYPING_TARGETS));
}

// Returns a stop function.
export function startKeys({ navigate, toggleTheme } = {}) {
  const onKeydown = (event) => {
    if (event.defaultPrevented || isTypingTarget(event.target)) return;

    const section = sectionForKey(event);
    if (section) {
      event.preventDefault();
      navigate?.(section);
      return;
    }
    if (isThemeKey(event)) toggleTheme?.();
  };

  window.addEventListener('keydown', onKeydown);
  return () => window.removeEventListener('keydown', onKeydown);
}
