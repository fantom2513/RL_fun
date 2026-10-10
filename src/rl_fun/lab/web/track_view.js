// Track canvas: a prerendered high-resolution background, vector F1 cars, the leader's rays,
// crash marks and a smoothed fit / follow camera. World coordinates are metres with y pointing up
// (like the pygame renderer); the screen flips y.

const CAR_LENGTH = 4.5; // metres, nose to rear wing tip
const MIN_CAR_PIXELS = 13; // keep cars legible when the whole track is in view
const BACKGROUND_MARGIN = 28; // metres of grass around the track
const BACKGROUND_MAX_SIDE = 4096; // px
const BACKGROUND_MAX_SCALE = 8; // px per metre
const EDGE_LINE_WIDTH = 0.5;
const CURB_WIDTH = 1.1;
const CURB_INSET = 0.45; // curb centre line, metres inside the road edge
const CURB_STRIPE = 2;
const DEFAULT_FOLLOW_ZOOM = 3;
const LEADER_COLOR = '#f2b01e';
const CRASH_COLOR = '#d62e3e';
const COMPACT_HUD_WIDTH = 640; // canvases narrower or lower than this get the one-line HUD plate
const COMPACT_HUD_HEIGHT = 480;
const HUD_FONT ='600 13px ui-monospace, "Cascadia Mono", Consolas, monospace';
const LABEL_FONT = '500 12px Bahnschrift, "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function mixWithWhite(hex, amount) {
  const [r, g, b] = hexToRgb(hex);
  const mix = (c) => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

function ringPath(path, ring) {
  ring.forEach(([x, y], index) => (index === 0 ? path.moveTo(x, y) : path.lineTo(x, y)));
  path.closePath();
}

// Alternating red / white stripes of CURB_STRIPE metres along the inner edge of the flagged spans.
function buildCurbPaths(track) {
  const red = new Path2D();
  const white = new Path2D();
  for (const span of track.curbs) {
    const ring = span.side === 'left' ? track.left : track.right;
    const count = ring.length;
    const inset = (index) => {
      const [px, py] = ring[index % count];
      const [cx, cy] = track.centerline[index % count];
      const length = Math.hypot(cx - px, cy - py) || 1;
      return [px + ((cx - px) / length) * CURB_INSET, py + ((cy - py) / length) * CURB_INSET];
    };
    let travelled = 0;
    let openStripe = -1;
    for (let index = span.start; index < span.stop; index += 1) {
      const [ax, ay] = inset(index);
      const [bx, by] = inset(index + 1);
      const length = Math.hypot(bx - ax, by - ay);
      if (length === 0) continue;
      let from = 0;
      while (from < length - 1e-9) {
        const stripe = Math.floor((travelled + from) / CURB_STRIPE + 1e-9);
        const boundary = (stripe + 1) * CURB_STRIPE - travelled;
        const to = Math.min(length, boundary);
        const path = stripe % 2 === 0 ? red : white;
        if (stripe !== openStripe) {
          path.moveTo(ax + ((bx - ax) * from) / length, ay + ((by - ay) * from) / length);
          openStripe = stripe;
        }
        path.lineTo(ax + ((bx - ax) * to) / length, ay + ((by - ay) * to) / length);
        from = to;
      }
      travelled += length;
    }
  }
  return { red, white };
}

// Two rows of squares across the road at the start pose.
function buildChequerPaths(track) {
  const { x, y, heading } = track.start;
  const [lx, ly] = track.left[0];
  const [rx, ry] = track.right[0];
  const width = Math.hypot(lx - rx, ly - ry);
  const columns = Math.max(4, Math.round(width / 1.1));
  const size = width / columns;
  const ux = Math.cos(heading);
  const uy = Math.sin(heading);
  const vx = -uy;
  const vy = ux;
  const dark = new Path2D();
  const light = new Path2D();
  const corner = (along, across) => [x + ux * along + vx * across, y + uy * along + vy * across];
  for (let row = 0; row < 2; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const path = (row + col) % 2 === 0 ? dark : light;
      const a0 = (row - 1) * size;
      const c0 = (col - columns / 2) * size;
      const points = [corner(a0, c0), corner(a0 + size, c0), corner(a0 + size, c0 + size), corner(a0, c0 + size)];
      path.moveTo(...points[0]);
      points.slice(1).forEach((p) => path.lineTo(...p));
      path.closePath();
    }
  }
  return { dark, light };
}

export class TrackView {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onModeChange = options.onModeChange ?? (() => {});
    this.track = null;
    this.style = null;
    this.frame = null;
    this.iterationWord = 'Пок.';
    this.laps = 1;
    this.parts = [];
    this.paths = null;
    this.background = null;
    this.bounds = null;
    this.mode = 'fit';
    this.fitZoom = 1;
    this.followZoom = DEFAULT_FOLLOW_ZOOM;
    this.pan = [0, 0];
    this.camera = { x: 0, y: 0, scale: 1 };
    this.snap = true;
    this.width = 1;
    this.height = 1;
    this.dpr = 1;
    this.crashes = [];
    this.previousStates = [];
    this.lastGeneration = -1;
    this.lastStep = -1;
    this.dirty = true;
    this.dragging = null;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement ?? canvas);
    canvas.addEventListener('wheel', (event) => this.onWheel(event), { passive: false });
    canvas.addEventListener('pointerdown', (event) => this.onPointerDown(event));
    canvas.addEventListener('pointermove', (event) => this.onPointerMove(event));
    canvas.addEventListener('pointerup', (event) => this.onPointerUp(event));
    canvas.addEventListener('pointercancel', (event) => this.onPointerUp(event));
    canvas.addEventListener('dblclick', () => this.resetView());
    this.resize();
    this.startLoop();
  }

  // ---- data -------------------------------------------------------------------------------

  // The race length: with several laps the progress reads "1,4 из 3 кр." instead of a percentage.
  setLaps(laps) {
    this.laps = laps;
  }

  progressText(progress) {
    if (this.laps <= 1) return `${Math.round(progress * 100)}%`;
    return `${progress.toFixed(1).replace('.', ',')} из ${this.laps} кр.`;
  }

  // Evolution counts generations, every other learner counts iterations.
  setLearner(learner) {
    this.iterationWord = learner === 'evolution' ? 'Пок.' : 'Итер.';
  }

  setStyle(style) {
    this.style = style;
    this.parts = style.car.map((part) => {
      const path = new Path2D();
      part.points.forEach(([x, y], index) => (index === 0 ? path.moveTo(x, y) : path.lineTo(x, y)));
      path.closePath();
      return { path, key: part.color };
    });
    this.dirty = true;
  }

  get trackName() {
    return this.track?.name ?? null;
  }

  setTrack(track, style) {
    if (style) this.setStyle(style);
    this.track = track;
    const xs = [...track.left, ...track.right].map((p) => p[0]);
    const ys = [...track.left, ...track.right].map((p) => p[1]);
    this.bounds = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
    const road = new Path2D();
    ringPath(road, track.left);
    ringPath(road, track.right);
    const edges = new Path2D();
    ringPath(edges, track.left);
    ringPath(edges, track.right);
    this.paths = { road, edges, curbs: buildCurbPaths(track), chequer: buildChequerPaths(track) };
    this.buildBackground();
    this.frame = null;
    this.resetMemory();
    this.snap = true;
    this.pan = [0, 0];
    this.dirty = true;
  }

  clearTrack() {
    this.track = null;
    this.paths = null;
    this.background = null;
    this.frame = null;
    this.resetMemory();
    this.dirty = true;
  }

  // Forget the frame and crash marks (used when another run with the same track is shown).
  reset() {
    this.frame = null;
    this.resetMemory();
    this.dirty = true;
  }

  resetMemory() {
    this.crashes = [];
    this.previousStates = [];
    this.lastGeneration = -1;
    this.lastStep = -1;
  }

  setFrame(frame) {
    if (frame.gen !== this.lastGeneration || frame.step === 0 || frame.step < this.lastStep) {
      this.crashes = [];
      this.previousStates = [];
      this.snap = true; // the leader jumps back to the start: do not glide across the track
    }
    frame.cars.forEach((car, index) => {
      if (this.previousStates[index] === 1 && car[3] === 0) this.crashes.push([car[0], car[1]]);
      this.previousStates[index] = car[3];
    });
    this.lastGeneration = frame.gen;
    this.lastStep = frame.step;
    this.frame = frame;
    this.dirty = true;
  }

  // ---- background -------------------------------------------------------------------------

  paintTrack(ctx) {
    const { style, paths } = this;
    ctx.fillStyle = style.road;
    ctx.fill(paths.road, 'evenodd');
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    ctx.lineWidth = CURB_WIDTH;
    ctx.strokeStyle = style.curb_red;
    ctx.stroke(paths.curbs.red);
    ctx.strokeStyle = style.curb_white;
    ctx.stroke(paths.curbs.white);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = EDGE_LINE_WIDTH;
    ctx.strokeStyle = style.road_line;
    ctx.stroke(paths.edges);
    ctx.fillStyle = style.chequer_light;
    ctx.fill(paths.chequer.light);
    ctx.fillStyle = style.chequer_dark;
    ctx.fill(paths.chequer.dark);
  }

  buildBackground() {
    const { minX, maxX, minY, maxY } = this.bounds;
    const left = minX - BACKGROUND_MARGIN;
    const top = maxY + BACKGROUND_MARGIN;
    const widthM = maxX - minX + 2 * BACKGROUND_MARGIN;
    const heightM = maxY - minY + 2 * BACKGROUND_MARGIN;
    const scale = Math.min(BACKGROUND_MAX_SCALE, BACKGROUND_MAX_SIDE / Math.max(widthM, heightM));
    const canvas = document.createElement('canvas');
    canvas.width = Math.ceil(widthM * scale);
    canvas.height = Math.ceil(heightM * scale);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = this.style.grass;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, -scale, -left * scale, top * scale);
    this.paintTrack(ctx);
    this.background = { canvas, scale, left, top };
  }

  // ---- size and camera --------------------------------------------------------------------

  resize() {
    const host = this.canvas.parentElement ?? this.canvas;
    const width = Math.max(1, Math.floor(host.clientWidth));
    const height = Math.max(1, Math.floor(host.clientHeight));
    const dpr = window.devicePixelRatio || 1;
    if (width === this.width && height === this.height && dpr === this.dpr && this.canvas.width) return;
    this.width = width;
    this.height = height;
    this.dpr = dpr;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.snap = true;
    this.dirty = true;
  }

  fitScale() {
    if (!this.bounds) return 1;
    const { minX, maxX, minY, maxY } = this.bounds;
    const margin = 1.08;
    return Math.min(this.width / ((maxX - minX) * margin), this.height / ((maxY - minY) * margin));
  }

  fitCenter() {
    const { minX, maxX, minY, maxY } = this.bounds;
    return [(minX + maxX) / 2, (minY + maxY) / 2];
  }

  leaderPosition() {
    const frame = this.frame;
    if (!frame || !frame.cars.length) return null;
    const car = frame.cars[frame.leader] ?? frame.cars[0];
    return [car[0], car[1]];
  }

  cameraTarget() {
    const fitScale = this.fitScale();
    if (this.mode === 'follow') {
      const [fx, fy] = this.fitCenter();
      const [x, y] = this.leaderPosition() ?? [fx, fy];
      return { x, y, scale: fitScale * this.followZoom };
    }
    const [fx, fy] = this.fitCenter();
    return { x: fx + this.pan[0], y: fy + this.pan[1], scale: fitScale * this.fitZoom };
  }

  // Moves the camera towards its target; returns true while it is still moving.
  updateCamera(dt) {
    if (!this.bounds) return false;
    const target = this.cameraTarget();
    const camera = this.camera;
    if (this.snap) {
      Object.assign(camera, target);
      this.snap = false;
      return false;
    }
    const move = 1 - Math.exp(-dt * (this.mode === 'follow' ? 9 : 12));
    const zoom = 1 - Math.exp(-dt * 12);
    camera.x += (target.x - camera.x) * move;
    camera.y += (target.y - camera.y) * move;
    camera.scale += (target.scale - camera.scale) * zoom;
    const offPixels = Math.hypot(target.x - camera.x, target.y - camera.y) * camera.scale;
    const offScale = Math.abs(target.scale - camera.scale) / target.scale;
    if (offPixels < 0.2 && offScale < 0.002) {
      Object.assign(camera, target);
      return false;
    }
    return true;
  }

  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this.dirty = true;
    this.onModeChange(mode);
  }

  toggleMode() {
    this.setMode(this.mode === 'fit' ? 'follow' : 'fit');
  }

  resetView() {
    this.fitZoom = 1;
    this.followZoom = DEFAULT_FOLLOW_ZOOM;
    this.pan = [0, 0];
    this.dirty = true;
  }

  zoomBy(factor, anchor = null) {
    if (!this.bounds) return;
    if (this.mode === 'follow') {
      this.followZoom = clamp(this.followZoom * factor, 0.5, 14);
    } else {
      const before = this.cameraTarget();
      const next = clamp(this.fitZoom * factor, 0.6, 14);
      const scale = this.fitScale() * next;
      if (anchor) {
        const ox = anchor[0] - this.width / 2;
        const oy = anchor[1] - this.height / 2;
        const wx = before.x + ox / before.scale;
        const wy = before.y - oy / before.scale;
        const [fx, fy] = this.fitCenter();
        this.pan = [wx - ox / scale - fx, wy + oy / scale - fy];
      }
      this.fitZoom = next;
    }
    this.dirty = true;
  }

  get zoomLabel() {
    return this.mode === 'follow' ? `×${this.followZoom.toFixed(1)}` : `×${this.fitZoom.toFixed(1)}`;
  }

  onWheel(event) {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    this.zoomBy(Math.exp(-event.deltaY * 0.0016), [event.clientX - rect.left, event.clientY - rect.top]);
  }

  onPointerDown(event) {
    if (this.mode !== 'fit' || event.button !== 0) return;
    this.canvas.setPointerCapture(event.pointerId);
    this.dragging = { x: event.clientX, y: event.clientY };
    this.canvas.classList.add('is-dragging');
  }

  onPointerMove(event) {
    if (!this.dragging) return;
    const scale = this.camera.scale;
    this.pan = [
      this.pan[0] - (event.clientX - this.dragging.x) / scale,
      this.pan[1] + (event.clientY - this.dragging.y) / scale,
    ];
    this.camera.x = this.fitCenter()[0] + this.pan[0];
    this.camera.y = this.fitCenter()[1] + this.pan[1];
    this.dragging = { x: event.clientX, y: event.clientY };
    this.dirty = true;
  }

  onPointerUp() {
    this.dragging = null;
    this.canvas.classList.remove('is-dragging');
  }

  // ---- drawing ----------------------------------------------------------------------------

  startLoop() {
    let last = performance.now();
    const tick = (now) => {
      requestAnimationFrame(tick);
      if ((window.devicePixelRatio || 1) !== this.dpr) this.resize();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const moving = this.updateCamera(dt);
      if (!moving && !this.dirty) return;
      this.dirty = false;
      this.draw();
    };
    requestAnimationFrame(tick);
  }

  draw() {
    const { ctx, width, height, dpr, style } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = style?.grass ?? '#7a8e9c';
    ctx.fillRect(0, 0, width, height);
    if (!this.track || !style) return;

    const { x: cx, y: cy, scale } = this.camera;
    const background = this.background;
    if (scale * dpr <= background.scale * 1.15) {
      const k = scale / background.scale;
      const sx = (background.left - cx) * scale + width / 2;
      const sy = height / 2 - (background.top - cy) * scale;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(background.canvas, sx, sy, background.canvas.width * k, background.canvas.height * k);
    } else {
      // Zoomed in past the prerendered resolution: draw the vector track directly, stay crisp.
      ctx.save();
      ctx.translate(width / 2 - cx * scale, height / 2 + cy * scale);
      ctx.scale(scale, -scale);
      this.paintTrack(ctx);
      ctx.restore();
    }

    const toScreen = (x, y) => [(x - cx) * scale + width / 2, height / 2 - (y - cy) * scale];
    if (this.frame) {
      this.drawCrashes(toScreen, scale);
      this.drawCars(toScreen, scale);
      this.drawLeader(toScreen, scale);
    }
    this.drawHud();
  }

  drawCrashes(toScreen, scale) {
    const ctx = this.ctx;
    const size = Math.max(4, 1.1 * scale);
    ctx.lineCap = 'round';
    for (const [color, width] of [['rgba(255,255,255,0.8)', 4.5], [CRASH_COLOR, 2.4]]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (const [x, y] of this.crashes) {
        const [sx, sy] = toScreen(x, y);
        if (sx < -20 || sy < -20 || sx > this.width + 20 || sy > this.height + 20) continue;
        ctx.moveTo(sx - size, sy - size);
        ctx.lineTo(sx + size, sy + size);
        ctx.moveTo(sx - size, sy + size);
        ctx.lineTo(sx + size, sy - size);
      }
      ctx.stroke();
    }
  }

  drawCar(sx, sy, heading, scale, state, leader) {
    const { ctx, dpr, style } = this;
    const k = Math.max(scale, MIN_CAR_PIXELS / CAR_LENGTH) * dpr;
    const cos = Math.cos(heading) * k;
    const sin = Math.sin(heading) * k;
    ctx.setTransform(cos, -sin, -sin, -cos, sx * dpr, sy * dpr);
    ctx.globalAlpha = state === 0 ? 0.5 : 1;
    for (const part of this.parts) {
      const colors = style.car_colors[part.key];
      let color = state === 0 ? colors.dead : colors.alive;
      if (state === 2) color = mixWithWhite(colors.alive, 0.5);
      if (leader && part.key === 'body') color = style.leader_body;
      this.ctx.fillStyle = color;
      this.ctx.fill(part.path);
    }
    ctx.globalAlpha = 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  drawCars(toScreen, scale) {
    const { cars, leader } = this.frame;
    const margin = 30;
    for (const wanted of [0, 1, 2]) {
      for (let index = 0; index < cars.length; index += 1) {
        const [x, y, heading, state] = cars[index];
        if (state !== wanted || index === leader) continue;
        const [sx, sy] = toScreen(x, y);
        if (sx < -margin || sy < -margin || sx > this.width + margin || sy > this.height + margin) continue;
        this.drawCar(sx, sy, heading, scale, state, false);
      }
    }
    const lead = cars[leader];
    if (lead) {
      const [sx, sy] = toScreen(lead[0], lead[1]);
      this.drawCar(sx, sy, lead[2], scale, lead[3] === 0 ? 0 : 1, true);
    }
  }

  drawLeader(toScreen, scale) {
    const { cars, leader, rays } = this.frame;
    const lead = cars[leader];
    if (!lead) return;
    const ctx = this.ctx;
    const [sx, sy] = toScreen(lead[0], lead[1]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.78)';
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.beginPath();
    for (const [rx, ry] of rays) {
      const [ex, ey] = toScreen(rx, ry);
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
    }
    ctx.stroke();
    ctx.beginPath();
    for (const [rx, ry] of rays) {
      const [ex, ey] = toScreen(rx, ry);
      ctx.moveTo(ex + 2, ey);
      ctx.arc(ex, ey, 2, 0, Math.PI * 2);
    }
    ctx.fill();

    const radius = Math.max(13, 3.4 * scale);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = LEADER_COLOR;
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = LEADER_COLOR;
    ctx.beginPath();
    ctx.moveTo(sx - 6, sy - radius - 13);
    ctx.lineTo(sx + 6, sy - radius - 13);
    ctx.lineTo(sx, sy - radius - 4);
    ctx.closePath();
    ctx.fill();
  }

  // On small canvases the five-row plate would cover a good part of the track: one line instead.
  drawCompactHud() {
    const ctx = this.ctx;
    const frame = this.frame;
    const text = frame
      ? `${this.iterationWord} ${frame.gen + 1} · ${frame.alive}/${frame.n} · ${this.progressText(frame.hud.progress)}`
      : 'Ждём кадр…';
    ctx.font = HUD_FONT;
    const x = 10;
    const y = 10;
    const plateWidth = Math.min(ctx.measureText(text).width + 24, Math.max(this.width - 150, 120));
    ctx.fillStyle = 'rgba(22, 31, 39, 0.84)';
    ctx.beginPath();
    ctx.roundRect(x, y, plateWidth, 26, 8);
    ctx.fill();
    ctx.fillStyle = CRASH_COLOR;
    ctx.beginPath();
    ctx.roundRect(x, y + 6, 3, 14, 2);
    ctx.fill();
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#f4f7f9';
    ctx.fillText(text, x + 12, y + 13, plateWidth - 18);
  }

  drawHud() {
    if (this.width < COMPACT_HUD_WIDTH || this.height < COMPACT_HUD_HEIGHT) {
      this.drawCompactHud();
      return;
    }
    const ctx = this.ctx;
    const frame = this.frame;
    const rows = [
      [this.iterationWord === 'Пок.' ? 'Поколение' : 'Итерация', frame ? String(frame.gen + 1) : '—'],
      ['Шаг', frame ? String(frame.step) : '—'],
      ['Живых', frame ? `${frame.alive} / ${frame.n}` : '—'],
      ['Скорость', frame ? `${frame.hud.speed.toFixed(1)} м/с` : '—'],
      ['Прогресс', frame ? this.progressText(frame.hud.progress) : '—'],
    ];
    const x = 12;
    const y = 12;
    const rowHeight = 20;
    const plateWidth = 168;
    const plateHeight = rows.length * rowHeight + 14;
    ctx.fillStyle = 'rgba(22, 31, 39, 0.84)';
    ctx.beginPath();
    ctx.roundRect(x, y, plateWidth, plateHeight, 10);
    ctx.fill();
    ctx.fillStyle = CRASH_COLOR;
    ctx.beginPath();
    ctx.roundRect(x, y + 8, 3, plateHeight - 16, 2);
    ctx.fill();
    ctx.textBaseline = 'middle';
    rows.forEach(([label, value], index) => {
      const rowY = y + 7 + rowHeight / 2 + index * rowHeight;
      ctx.font = LABEL_FONT;
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(214, 224, 232, 0.72)';
      ctx.fillText(label, x + 14, rowY);
      ctx.font = HUD_FONT;
      ctx.textAlign = 'right';
      ctx.fillStyle = '#f4f7f9';
      ctx.fillText(value, x + plateWidth - 12, rowY);
    });
  }
}
