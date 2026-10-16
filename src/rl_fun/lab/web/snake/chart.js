import {canvasContext} from './board.js';

export function sampleHistory(history, limit = 500) {
  if (history.length <= limit) return history;
  return Array.from({length: limit}, (_, i) => history[Math.round(i * (history.length - 1) / (limit - 1))]);
}

export class SnakeChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.history = [];
    this.metric = 'score';
    new ResizeObserver(() => this.draw()).observe(canvas);
    window.addEventListener('themechange', () => this.draw());
  }
  update(history, metric) { this.history = sampleHistory(history); this.metric = metric; this.draw(); }
  draw() {
    const {ctx, width, height, color} = canvasContext(this.canvas);
    if (width < 50) return;
    const series = this.metric === 'occupancy' ? [['occupancy', '--success'], ['evaluation_occupancy', '--info']]
      : this.metric === 'win_rate' ? [['evaluation_win_rate', '--info']]
      : this.metric === 'reward' ? [['reward', '--accent'], ['mean_reward', '--info']]
      : [['mean_score', '--success'], ['evaluation_score', '--info']];
    const values = this.history.flatMap(row => series.map(([key]) => row[key]).filter(Number.isFinite));
    let low = Math.min(0, ...values), high = Math.max(1, ...values);
    if (high === low) high++;
    const left = 42, right = width - 14, top = 12, bottom = height - 28;
    ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--font-mono');
    for (let i = 0; i < 4; i++) {
      const y = top + i * (bottom - top) / 3;
      ctx.strokeStyle = color('--border'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.fillStyle = color('--text-3'); ctx.textAlign = 'right';
      const percentage = ['occupancy','win_rate'].includes(this.metric);
      ctx.fillText(((high - i * (high - low) / 3) * (percentage ? 100 : 1)).toFixed(1) + (percentage ? '%' : ''), left - 8, y + 4);
    }
    if (!this.history.length) {
      ctx.fillStyle = color('--text-3'); ctx.textAlign = 'center';
      ctx.fillText('Начните обучение — здесь появится прогресс', width / 2, height / 2);
      return;
    }
    const first = this.history[0].iteration, last = this.history.at(-1).iteration;
    for (const [key, accent] of series) {
      ctx.strokeStyle = color(accent); ctx.lineWidth = 2;
      ctx.beginPath(); let started = false;
      for (const row of this.history) {
        if (!Number.isFinite(row[key])) continue;
        const x = left + (row.iteration - first) / Math.max(1, last - first) * (right - left);
        const y = bottom - (row[key] - low) / (high - low) * (bottom - top);
        if (!started) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        started = true;
      }
      ctx.stroke();
    }
    ctx.fillStyle = color('--text-3'); ctx.textAlign = 'left';
    ctx.fillText(String(first), left, height - 8); ctx.textAlign = 'right';
    ctx.fillText(String(last), right, height - 8);
  }
}
