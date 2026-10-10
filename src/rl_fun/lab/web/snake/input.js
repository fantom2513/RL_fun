// One accepted turn per server tick; keyboard control only when a board has focus.
export class TurnQueue {
  constructor() { this.queue = []; this.inflight = null; }
  push(direction, current) {
    const reference = this.queue.at(-1) ?? this.inflight ?? current;
    if (direction === reference || (direction - reference + 4) % 4 === 2) return;
    if (this.queue.length < 2) this.queue.push(direction);
  }
  take() { this.inflight = this.queue.shift() ?? null; return this.inflight; }
  ack() { this.inflight = null; }
  clear() { this.queue.length = 0; this.inflight = null; }
}

export function keyDirection(event) {
  if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing
      || event.target?.tagName !== 'CANVAS') return null;
  return ({ArrowRight: 0, KeyD: 0, ArrowDown: 1, KeyS: 1,
    ArrowLeft: 2, KeyA: 2, ArrowUp: 3, KeyW: 3})[event.code] ?? null;
}
