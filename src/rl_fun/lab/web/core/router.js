// Hash router for the lab shell: parses '#/section/...' routes and shows the matching screen.

export const SECTIONS = ['lab', 'tracks', 'challenges', 'garage', 'settings'];

export const TITLES = {
  lab: 'Лаборатория',
  tracks: 'Трассы',
  challenges: 'Челленджи',
  garage: 'Гараж',
  settings: 'Настройки',
};

export function parseRoute(hash) {
  const parts = String(hash ?? '')
    .replace(/^#/, '')
    .split('/')
    .filter(Boolean);
  const first = parts[0];
  const valid = first !== undefined && SECTIONS.includes(first);
  return {
    name: valid ? first : 'lab',
    rest: valid ? parts.slice(1) : [],
    valid,
  };
}

export function hashFor(name, ...rest) {
  return '#/' + [name, ...rest].join('/');
}

export function showScreen(route, { focus = false } = {}) {
  for (const screen of document.querySelectorAll('.screen-pane[data-screen]')) {
    screen.hidden = screen.dataset.screen !== route.name;
  }

  const current = hashFor(route.name);
  for (const item of document.querySelectorAll('.rail-item')) {
    if (item.getAttribute('href') === current) item.setAttribute('aria-current', 'page');
    else item.removeAttribute('aria-current');
  }

  document.body.dataset.screen = route.name;
  document.title = route.name === 'lab' ? TITLES.lab : `${TITLES[route.name]} · Лаборатория`;

  if (focus) {
    const title = document.querySelector(`.screen-pane[data-screen="${route.name}"]:not([hidden]) h1`);
    title?.focus({ preventScroll: true });
  }
}

// Calls onRoute(route) after each navigation; returns a stop function.
export function startRouter(onRoute) {
  let firstCall = true;

  const handle = () => {
    const route = parseRoute(window.location.hash);
    // Rewrite unknown hashes in place; replaceState does not fire hashchange.
    if (!route.valid) history.replaceState(null, '', hashFor(route.name));
    showScreen(route, { focus: !firstCall });
    firstCall = false;
    onRoute?.(route);
  };

  window.addEventListener('hashchange', handle);
  handle();
  return () => window.removeEventListener('hashchange', handle);
}
