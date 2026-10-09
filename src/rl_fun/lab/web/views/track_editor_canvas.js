// Canvas of the track editor: draw the centerline of a closed track with the mouse. Click on empty
// ground adds a point (on a segment it inserts one), drag moves a point, right click or Delete
// removes it, arrows nudge the selected one by a metre. The road is drawn at its real width.

import { closestOnPolyline, insertPoint, selfIntersections, snapPoint } from '../core/geometry.js';

const VIEW_WIDTH = 340; // metres visible across, the workspace is centred on the origin
const VIEW_HEIGHT = 220;
const GRID = 10;
const HIT_RADIUS = 11; // pixels
const MAX_POINTS = 400;
const MAX_HISTORY = 100;

export class TrackEditorCanvas {
  constructor(canvas, { onChange = () => {} } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onChange = onChange;
    this.points = [];
    this.width = 10;
    this.selected = null;
    this.hover = null;
    this.drag = null;
    this.history = [];
    this.pixelRatio = 1;
    canvas.addEventListener('pointerdown', (event) => this.pointerDown(event));
    canvas.addEventListener('pointermove', (event) => this.pointerMove(event));
    canvas.addEventListener('pointerup', (event) => this.pointerUp(event));
    canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      this.draw();
    });
    canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      const index = this.pointAt(event);
      if (index !== null) this.removePoint(index);
    });
    canvas.addEventListener('keydown', (event) => this.keyDown(event));
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.resize();
  }

  destroy() {
    this.observer.disconnect();
  }

  // ---- state ------------------------------------------------------------------------------

  setTrack(points, width) {
    this.points = points.map(([x, y]) => [x, y]);
    this.width = width;
    this.selected = null;
    this.history = [];
    this.changed();
  }

  setPoints(points) {
    this.remember();
    this.points = points.map(([x, y]) => [x, y]);
    this.selected = null;
    this.changed();
  }

  setWidth(width) {
    this.width = width;
    this.changed();
  }

  undo() {
    const previous = this.history.pop();
    if (!previous) return;
    this.points = previous;
    this.selected = null;
    this.changed();
  }

  get canUndo() {
    return this.history.length > 0;
  }

  remember() {
    this.history.push(this.points.map(([x, y]) => [x, y]));
    if (this.history.length > MAX_HISTORY) this.history.shift();
  }

  changed() {
    this.draw();
    this.onChange();
  }

  removePoint(index) {
    this.remember();
    this.points.splice(index, 1);
    this.selected = null;
    this.changed();
  }

  // ---- geometry of the view ---------------------------------------------------------------

  get scale() {
    return Math.min(this.canvas.width / this.pixelRatio / VIEW_WIDTH, this.canvas.height / this.pixelRatio / VIEW_HEIGHT);
  }

  toScreen([x, y]) {
    const scale = this.scale;
    return [this.canvas.width / this.pixelRatio / 2 + x * scale, this.canvas.height / this.pixelRatio / 2 - y * scale];
  }

  toWorld(event) {
    const box = this.canvas.getBoundingClientRect();
    const scale = this.scale;
    const x = (event.clientX - box.left - box.width / 2) / scale;
    const y = -(event.clientY - box.top - box.height / 2) / scale;
    return [x, y];
  }

  pointAt(event) {
    const box = this.canvas.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    let best = null;
    let bestDistance = HIT_RADIUS;
    this.points.forEach((point, index) => {
      const [sx, sy] = this.toScreen(point);
      const distance = Math.hypot(sx - px, sy - py);
      if (distance <= bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  }

  // ---- input ------------------------------------------------------------------------------

  pointerDown(event) {
    if (event.button !== 0) return;
    this.canvas.focus();
    const hit = this.pointAt(event);
    if (hit !== null) {
      this.remember();
      this.selected = hit;
      this.drag = { index: hit };
    } else if (this.points.length < MAX_POINTS) {
      this.remember();
      const point = snapPoint(this.toWorld(event));
      const near = closestOnPolyline(this.points, point);
      const onRoad = near && near.distance * this.scale < HIT_RADIUS && this.points.length >= 3;
      if (onRoad) {
        this.points = insertPoint(this.points, point);
        this.selected = near.index + 1;
      } else {
        this.points.push(point);
        this.selected = this.points.length - 1;
      }
      this.drag = { index: this.selected };
    }
    this.canvas.setPointerCapture?.(event.pointerId);
    this.changed();
  }

  pointerMove(event) {
    if (this.drag) {
      this.points[this.drag.index] = snapPoint(this.toWorld(event));
      this.changed();
      return;
    }
    const hover = this.pointAt(event);
    if (hover !== this.hover) {
      this.hover = hover;
      this.canvas.style.cursor = hover === null ? 'crosshair' : 'grab';
      this.draw();
    }
  }

  pointerUp(event) {
    if (!this.drag) return;
    this.drag = null;
    this.canvas.releasePointerCapture?.(event.pointerId);
    this.draw();
  }

  keyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.code === 'KeyZ') {
      event.preventDefault();
      this.undo();
      return;
    }
    if (this.selected === null) return;
    const nudge = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (nudge) {
      event.preventDefault();
      this.remember();
      const [x, y] = this.points[this.selected];
      this.points[this.selected] = [x + nudge[0], y + nudge[1]];
      this.changed();
    } else if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault();
      this.removePoint(this.selected);
    }
  }

  // ---- drawing ----------------------------------------------------------------------------

  resize() {
    const box = this.canvas.getBoundingClientRect();
    this.pixelRatio = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(box.width * this.pixelRatio));
    this.canvas.height = Math.max(1, Math.round(box.height * this.pixelRatio));
    this.draw();
  }

  colors() {
    const style = getComputedStyle(this.canvas);
    const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      grass: read('--scene-grass', '#7a8e9c'),
      road: read('--scene-road', '#4f616e'),
      line: read('--scene-line', '#fffffc'),
      curb: read('--scene-curb', '#d62e3e'),
      accent: read('--accent', '#f5a524'),
      danger: read('--danger', '#f0566a'),
    };
  }

  tracePath(points, closed) {
    const ctx = this.ctx;
    ctx.beginPath();
    points.forEach((point, index) => {
      const [x, y] = this.toScreen(point);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (closed) ctx.closePath();
  }

  draw() {
    const ctx = this.ctx;
    const colors = this.colors();
    const width = this.canvas.width / this.pixelRatio;
    const height = this.canvas.height / this.pixelRatio;
    ctx.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
    ctx.fillStyle = colors.grass;
    ctx.fillRect(0, 0, width, height);
    this.drawGrid(ctx, colors, width, height);
    const closed = this.points.length >= 3;
    if (closed) {
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      this.tracePath(this.points, true);
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = (this.width + 1.2) * this.scale;
      ctx.stroke();
      ctx.strokeStyle = colors.road;
      ctx.lineWidth = this.width * this.scale;
      ctx.stroke();
      this.drawStart(ctx, colors);
    }
    this.tracePath(this.points, closed);
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = colors.line;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    this.drawCrossings(ctx, colors);
    this.drawHandles(ctx, colors);
  }

  drawGrid(ctx, colors, width, height) {
    const scale = this.scale;
    ctx.save();
    ctx.strokeStyle = colors.line;
    for (let x = -VIEW_WIDTH; x <= VIEW_WIDTH; x += GRID) {
      ctx.globalAlpha = x % (GRID * 5) === 0 ? 0.28 : 0.1;
      const screenX = width / 2 + x * scale;
      ctx.beginPath();
      ctx.moveTo(screenX, 0);
      ctx.lineTo(screenX, height);
      ctx.stroke();
    }
    for (let y = -VIEW_HEIGHT; y <= VIEW_HEIGHT; y += GRID) {
      ctx.globalAlpha = y % (GRID * 5) === 0 ? 0.28 : 0.1;
      const screenY = height / 2 - y * scale;
      ctx.beginPath();
      ctx.moveTo(0, screenY);
      ctx.lineTo(width, screenY);
      ctx.stroke();
    }
    ctx.restore();
  }

  // The start line across the road at the first point, and an arrow along the driving direction.
  drawStart(ctx, colors) {
    const [a, b] = [this.points[0], this.points[1]];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const direction = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
    const normal = [-direction[1], direction[0]];
    const half = this.width / 2;
    const left = this.toScreen([a[0] + normal[0] * half, a[1] + normal[1] * half]);
    const right = this.toScreen([a[0] - normal[0] * half, a[1] - normal[1] * half]);
    ctx.save();
    ctx.lineCap = 'butt';
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(...left);
    ctx.lineTo(...right);
    ctx.stroke();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = colors.curb;
    ctx.stroke();
    ctx.restore();
    const tip = this.toScreen([a[0] + direction[0] * 9, a[1] + direction[1] * 9]);
    const base = this.toScreen([a[0] + direction[0] * 3, a[1] + direction[1] * 3]);
    const wing = (sign) => this.toScreen([
      a[0] + direction[0] * 5.5 + normal[0] * 2.5 * sign,
      a[1] + direction[1] * 5.5 + normal[1] * 2.5 * sign,
    ]);
    ctx.save();
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(...base);
    ctx.lineTo(...tip);
    ctx.moveTo(...wing(1));
    ctx.lineTo(...tip);
    ctx.lineTo(...wing(-1));
    ctx.stroke();
    ctx.restore();
  }

  drawCrossings(ctx, colors) {
    const crossings = selfIntersections(this.points);
    if (!crossings.length) return;
    ctx.save();
    ctx.strokeStyle = colors.danger;
    ctx.lineWidth = 4;
    const count = this.points.length;
    for (const index of new Set(crossings.flat())) {
      ctx.beginPath();
      ctx.moveTo(...this.toScreen(this.points[index]));
      ctx.lineTo(...this.toScreen(this.points[(index + 1) % count]));
      ctx.stroke();
    }
    ctx.restore();
  }

  drawHandles(ctx, colors) {
    this.points.forEach((point, index) => {
      const [x, y] = this.toScreen(point);
      const active = index === this.selected;
      ctx.beginPath();
      ctx.arc(x, y, active ? 7 : index === this.hover ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = active ? colors.accent : colors.line;
      ctx.fill();
      ctx.strokeStyle = colors.road;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }
}
