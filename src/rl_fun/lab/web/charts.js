// Learning curves: one line per run over generations on a HiDPI canvas, with axis ticks, a metric
// switch, a hover tooltip and an empty state. The chart does not own the data: it asks `getRuns()`
// (an array of {id, name, color, gens}) every time it draws, so live gen messages only need an
// `update()` call.

export const METRICS = [
  { id: 'best', label: 'Лучший прогресс', kind: 'percent' },
  { id: 'mean', label: 'Средний прогресс', kind: 'percent' },
  { id: 'finished', label: 'Финишировали', kind: 'count' },
  { id: 'best_fitness', label: 'Лучшая награда', kind: 'number' },
];

const FONT = '12px ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace';
const FONT_UI = '13px Bahnschrift, "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
const PAD = { left: 52, right: 14, top: 12, bottom: 28 };

function niceStep(span, target) {
  const raw = span / Math.max(1, target);
  const base = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / base;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * base;
}

function ticks(low, high, target, integer = false) {
  let step = niceStep(high - low || 1, target);
  if (integer) step = Math.max(1, Math.round(step));
  const out = [];
  for (let v = Math.ceil(low / step - 1e-9) * step; v <= high + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

const valueOf = (gen, metric) => {
  const value = gen[metric];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

export class Chart {
  constructor(canvas, getRuns) {
    this.canvas = canvas;
    this.getRuns = getRuns;
    this.ctx = canvas.getContext('2d');
    this.metric = 'best';
    this.highlight = null;
    this.hover = null;
    this.frame = 0;
    this.layout = null;
    this.width = 0;
    this.height = 0;
    this.dpr = 1;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener('pointermove', (event) => this.pointer(event));
    canvas.addEventListener('pointerleave', () => this.setHover(null));
    canvas.addEventListener('pointerdown', (event) => this.pointer(event));
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.update());
    this.resize();
  }

  setMetric(metric) {
    if (METRICS.some((m) => m.id === metric)) this.metric = metric;
    this.update();
  }

  setHighlight(id) {
    this.highlight = id;
    this.update();
  }

  // Redraws on the next animation frame; a timer backs this up for hidden panes where frames stall.
  update() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => this.draw());
    this.fallback = setTimeout(() => this.draw(), 80);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.update();
  }

  colors() {
    const style = getComputedStyle(this.canvas);
    const get = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      text: get('--text', '#212d37'),
      muted: get('--muted', '#5a6a77'),
      line: get('--line', '#bfcbd4'),
      surface: get('--surface', '#f3f6f8'),
    };
  }

  series() {
    return this.getRuns()
      .map((run) => ({
        id: run.id,
        name: run.name,
        color: run.color,
        points: run.gens
          .map((gen, index) => ({ x: typeof gen.gen === 'number' ? gen.gen : index, y: valueOf(gen, this.metric), raw: gen }))
          .filter((point) => point.y !== null),
      }))
      .filter((s) => s.points.length);
  }

  draw() {
    cancelAnimationFrame(this.frame);
    clearTimeout(this.fallback);
    this.frame = 0;
    const { ctx, width, height } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const colors = this.colors();
    const metric = METRICS.find((m) => m.id === this.metric);
    const series = this.series();
    this.layout = null;
    if (!series.length) {
      this.drawEmpty(colors);
      return;
    }

    const plot = { x: PAD.left, y: PAD.top, w: Math.max(10, width - PAD.left - PAD.right), h: Math.max(10, height - PAD.top - PAD.bottom) };
    const all = series.flatMap((s) => s.points);
    const maxX = Math.max(1, ...all.map((p) => p.x));
    let low = Math.min(...all.map((p) => p.y));
    let high = Math.max(...all.map((p) => p.y));
    if (metric.kind === 'percent') {
      low = 0;
      high = Math.max(1, high);
    } else if (metric.kind === 'count') {
      low = 0;
      high = Math.max(1, high);
    } else if (high - low < 1e-9) {
      low -= 0.5;
      high += 0.5;
    }
    const yTicks = ticks(low, high, Math.max(2, Math.floor(plot.h / 38)), metric.kind === 'count');
    const yMax = Math.max(high, yTicks[yTicks.length - 1] ?? high);
    const yMin = Math.min(low, yTicks[0] ?? low);
    const pad = metric.kind === 'number' ? (yMax - yMin) * 0.06 : 0;
    const lo = yMin - pad;
    const hi = yMax + pad * 0.5 + (metric.kind === 'percent' ? 0.02 : 0);
    const sx = (x) => plot.x + (x / maxX) * plot.w;
    const sy = (y) => plot.y + plot.h - ((y - lo) / (hi - lo)) * plot.h;
    this.layout = { plot, maxX, series, sx, sy, metric };

    ctx.font = FONT;
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;
    for (const tick of yTicks) {
      const y = Math.round(sy(tick)) + 0.5;
      ctx.strokeStyle = colors.line;
      ctx.globalAlpha = tick === 0 ? 1 : 0.55;
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.muted;
      ctx.textAlign = 'right';
      ctx.fillText(this.format(tick, metric, true), plot.x - 8, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = ticks(0, maxX, Math.max(2, Math.floor(plot.w / 70)), true);
    for (const tick of xTicks) {
      const x = Math.round(sx(tick)) + 0.5;
      if (x > width - 78) continue; // leave room for the axis caption
      ctx.strokeStyle = colors.line;
      ctx.beginPath();
      ctx.moveTo(x, plot.y + plot.h);
      ctx.lineTo(x, plot.y + plot.h + 4);
      ctx.stroke();
      ctx.fillStyle = colors.muted;
      ctx.fillText(String(tick + 1), x, plot.y + plot.h + 7);
    }
    ctx.textAlign = 'right';
    ctx.fillStyle = colors.muted;
    ctx.fillText('поколение', width - 2, plot.y + plot.h + 7 + 0.1);

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x - 4, plot.y - 4, plot.w + 8, plot.h + 8);
    ctx.clip();
    const ordered = [...series].sort((a, b) => (a.id === this.highlight) - (b.id === this.highlight));
    for (const s of ordered) this.drawLine(s, sx, sy, s.id === this.highlight, series.length > 1 && this.highlight && s.id !== this.highlight);
    ctx.restore();

    if (this.hover) this.drawHover(colors, metric);
  }

  drawLine(s, sx, sy, active, dim) {
    const { ctx } = this;
    ctx.globalAlpha = dim ? 0.55 : 1;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = active ? 2.8 : 1.9;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    s.points.forEach((p, i) => (i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y))));
    ctx.stroke();
    const last = s.points[s.points.length - 1];
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(sx(last.x), sy(last.y), active ? 4 : 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  drawEmpty(colors) {
    const { ctx, width, height } = this;
    ctx.font = FONT_UI;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = colors.muted;
    const runs = this.getRuns().length;
    ctx.fillText(
      runs ? 'Кривые появятся после первого завершённого поколения.' : 'Создайте запуск, и здесь появятся кривые обучения.',
      width / 2, height / 2,
    );
    ctx.strokeStyle = colors.line;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    ctx.setLineDash([]);
  }

  format(value, metric, axis = false) {
    if (metric.kind === 'percent') return `${Math.round(value * (axis ? 100 : 1000)) / (axis ? 1 : 10)}%`.replace('.', ',');
    if (metric.kind === 'count') return String(Math.round(value));
    const abs = Math.abs(value);
    const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
    return value.toLocaleString('ru-RU', { maximumFractionDigits: digits });
  }

  pointer(event) {
    if (!this.layout) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const { plot, maxX } = this.layout;
    if (x < plot.x - 6 || x > plot.x + plot.w + 6) {
      this.setHover(null);
      return;
    }
    const gen = Math.round(Math.min(maxX, Math.max(0, ((x - plot.x) / plot.w) * maxX)));
    this.setHover({ gen, x, y: event.clientY - rect.top });
  }

  setHover(hover) {
    this.hover = hover;
    this.update();
  }

  drawHover(colors, metric) {
    const { ctx, width } = this;
    const { plot, sx, sy, series } = this.layout;
    const gen = this.hover.gen;
    const rows = series
      .map((s) => ({ s, p: s.points.find((p) => p.x === gen) }))
      .filter((row) => row.p);
    const x = Math.round(sx(gen)) + 0.5;
    ctx.strokeStyle = colors.muted;
    ctx.globalAlpha = 0.6;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    for (const { s, p } of rows) {
      ctx.fillStyle = colors.surface;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx(p.x), sy(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    ctx.font = FONT;
    const title = `Поколение ${gen + 1} · ${metric.label.toLowerCase()}`;
    const lines = rows.length ? rows.map(({ s, p }) => `${s.name}  ${this.format(p.y, metric)}`) : ['нет данных'];
    const boxW = Math.ceil(Math.max(ctx.measureText(title).width, ...lines.map((l) => ctx.measureText(l).width + 16))) + 20;
    const boxH = 14 + 18 * (lines.length + 1);
    let bx = x + 12;
    if (bx + boxW > width - 4) bx = x - 12 - boxW;
    bx = Math.max(4, bx);
    const by = Math.min(Math.max(plot.y, this.hover.y - boxH / 2), plot.y + plot.h - boxH);
    ctx.fillStyle = colors.surface;
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(bx + 0.5, by + 0.5, boxW, boxH, 6);
    ctx.fill();
    ctx.stroke();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = colors.muted;
    ctx.fillText(title, bx + 10, by + 15);
    rows.forEach(({ s }, i) => {
      const y = by + 15 + 18 * (i + 1);
      ctx.fillStyle = s.color;
      ctx.fillRect(bx + 10, y - 4, 8, 8);
      ctx.fillStyle = colors.text;
      ctx.fillText(lines[i], bx + 24, y);
    });
    if (!rows.length) {
      ctx.fillStyle = colors.text;
      ctx.fillText(lines[0], bx + 10, by + 33);
    }
  }
}
