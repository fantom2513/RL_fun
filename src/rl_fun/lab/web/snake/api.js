// Snake API uses the same local server as the racing lab.
export async function request(path, body, method) {
  const response = await fetch('/api/snake/' + path, {
    method: method ?? (body === undefined ? 'GET' : 'POST'),
    headers: body === undefined ? {} : {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? 'Запрос не выполнен');
  return data;
}

export function subscribe(onState, onError) {
  const source = new EventSource('/api/snake/events');
  source.addEventListener('state', event => onState(JSON.parse(event.data)));
  source.onerror = () => onError?.();
  return () => source.close();
}
