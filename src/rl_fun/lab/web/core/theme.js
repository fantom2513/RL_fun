// Theme preference ('system' | 'light' | 'dark') stored in localStorage and applied as data-theme.

const STORAGE_KEY = 'lab.theme';

export function normalizeTheme(value) {
  return value === 'light' || value === 'dark' ? value : 'system';
}

export function effectiveTheme(preference, prefersDarkNow) {
  if (preference === 'light' || preference === 'dark') return preference;
  return prefersDarkNow ? 'dark' : 'light';
}

export function toggledTheme(preference, prefersDarkNow) {
  return effectiveTheme(preference, prefersDarkNow) === 'dark' ? 'light' : 'dark';
}

export function prefersDark() {
  // Without matchMedia (old or test environments) assume the dark theme.
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return true;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function getTheme() {
  try {
    return normalizeTheme(localStorage.getItem(STORAGE_KEY));
  } catch {
    return 'system';
  }
}

function applyTheme(value) {
  const root = document.documentElement;
  if (value === 'system') delete root.dataset.theme;
  else root.dataset.theme = value;
}

export function setTheme(preference) {
  const value = normalizeTheme(preference);
  try {
    localStorage.setItem(STORAGE_KEY, value);
  } catch {
    // Storage may be blocked; the theme still applies for this page.
  }
  applyTheme(value);
  window.dispatchEvent(
    new CustomEvent('themechange', {
      detail: { preference: value, effective: effectiveTheme(value, prefersDark()) },
    }),
  );
  return value;
}

// Applies the stored preference without writing it back.
export function initTheme() {
  const value = getTheme();
  applyTheme(value);
  return value;
}

export function toggleTheme() {
  return setTheme(toggledTheme(getTheme(), prefersDark()));
}
