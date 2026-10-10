// Crisp canvas grid, colored by the lab's existing theme tokens.
export function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const style = getComputedStyle(document.documentElement);
  return {ctx, width, height, color: name => style.getPropertyValue(name).trim()};
}

export class SnakeBoard {
  constructor(canvas, accent) {
    this.canvas = canvas;
    this.accent = accent;
    this.frame = null;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    window.addEventListener('themechange', () => this.draw());
  }
  setFrame(frame) { this.frame = frame; this.draw(); }
  draw() {
    const {ctx, width, height, color} = canvasContext(this.canvas);
    ctx.fillStyle = color('--surface-sunken');
    ctx.fillRect(0, 0, width, height);
    if (!this.frame) return;
    const {size, snake, food, direction} = this.frame;
    const cell = Math.min(width, height) / size;
    const ox = (width - cell * size) / 2, oy = (height - cell * size) / 2;
    ctx.strokeStyle = color('--border');
    ctx.lineWidth = .5;
    for (let i = 0; i <= size; i++) {
      ctx.beginPath(); ctx.moveTo(ox + i * cell, oy); ctx.lineTo(ox + i * cell, oy + size * cell); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ox, oy + i * cell); ctx.lineTo(ox + size * cell, oy + i * cell); ctx.stroke();
    }
    if (food) {
      ctx.fillStyle = color('--danger');
      ctx.beginPath();
      ctx.arc(ox + (food[0] + .5) * cell, oy + (food[1] + .5) * cell, cell * .29, 0, Math.PI * 2);
      ctx.fill();
    }
    snake.forEach(([x, y], index) => {
      ctx.globalAlpha = index === 0 ? 1 : Math.max(.4, .9 - index * .015);
      ctx.fillStyle = color(this.accent);
      ctx.beginPath();
      ctx.roundRect(ox + x * cell + 1.5, oy + y * cell + 1.5, cell - 3, cell - 3, cell * .18);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    if (snake.length) {
      const [x, y] = snake[0];
      const [dx, dy] = [[1,0],[0,1],[-1,0],[0,-1]][direction];
      ctx.fillStyle = color('--surface-sunken');
      for (const sign of [-1, 1]) {
        ctx.beginPath();
        ctx.arc(ox + (x + .5 + dx * .18 - dy * sign * .18) * cell,
          oy + (y + .5 + dy * .18 + dx * sign * .18) * cell, cell * .055, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
