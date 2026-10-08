// Thin client of the lab HTTP API. Every failure is reported to the user through the error handler
// (never silently swallowed) and also thrown so callers can stop what they were doing.

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

let errorHandler = (message) => console.error(message);

export function setErrorHandler(handler) {
  errorHandler = handler;
}

export function reportError(message) {
  errorHandler(message);
}

async function request(method, path, body) {
  const options = { method, headers: { Accept: 'application/json' } };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    const error = new ApiError('Нет связи с лабораторией. Проверьте, что сервер запущен.');
    reportError(error.message);
    throw error;
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const message = (data && data.error) || `Сервер ответил ошибкой ${response.status}.`;
    reportError(message);
    throw new ApiError(message, response.status);
  }
  return data;
}

export const getCatalog = () => request('GET', '/api/catalog');
export const getTrack = (name) => request('GET', `/api/tracks/${encodeURIComponent(name)}`);
export const listRuns = () => request('GET', '/api/runs');
export const getRun = (id) => request('GET', `/api/runs/${encodeURIComponent(id)}`);
export const createRun = (config) => request('POST', '/api/runs', config);
export const deleteRun = (id) => request('DELETE', `/api/runs/${encodeURIComponent(id)}`);

// `cmd` is a command object ({cmd: 'speed', value: 4}) or just its name ('pause').
export function command(id, cmd) {
  const body = typeof cmd === 'string' ? { cmd } : cmd;
  return request('POST', `/api/runs/${encodeURIComponent(id)}/command`, body);
}

const EVENT_KINDS = ['frame', 'gen', 'status', 'notice'];
const TERMINAL = new Set(['finished', 'stopped', 'error']);

// Subscribes to a run's SSE stream. Handlers: frame, gen, status, notice (payload objects), plus
// open(), reconnecting(attempt, max) and disconnected() for the connection itself. The stream is
// re-opened a limited number of times (resuming after the generations already received); once the
// run is finished the stream is closed for good. Returns {close(), reconnect()}.
export function subscribe(id, handlers, { maxRetries = 5 } = {}) {
  let source = null;
  let timer = null;
  let retries = 0;
  let generations = 0;
  let closed = false;

  const connect = () => {
    if (closed) return;
    source = new EventSource(`/api/runs/${encodeURIComponent(id)}/stream?from=${generations}`);
    source.onopen = () => {
      retries = 0;
      handlers.open?.();
    };
    for (const kind of EVENT_KINDS) {
      source.addEventListener(kind, (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }
        if (kind === 'gen') generations += 1;
        handlers[kind]?.(payload);
        if (kind === 'status' && TERMINAL.has(payload.status)) close();
      });
    }
    source.onerror = () => {
      if (closed) return;
      source.close();
      if (retries >= maxRetries) {
        handlers.disconnected?.();
        return;
      }
      retries += 1;
      handlers.reconnecting?.(retries, maxRetries);
      timer = setTimeout(connect, Math.min(1000 * 2 ** (retries - 1), 8000));
    };
  };

  function close() {
    closed = true;
    clearTimeout(timer);
    source?.close();
  }

  function reconnect() {
    clearTimeout(timer);
    source?.close();
    closed = false;
    retries = 0;
    connect();
  }

  connect();
  return { close, reconnect };
}
