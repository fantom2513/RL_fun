/* Prototype helpers: theme switch, simple toggles and drawing of sample scenes.
   Not production code: the real app keeps its own track_view.js / network_view.js / charts.js. */
(function () {
  'use strict';

  const params = new URLSearchParams(location.search);
  const theme = params.get('theme');
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- random ----------
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());

  // ---------- track geometry ----------
  function resample(points, step) {
    const n = points.length;
    const out = [];
    for (let i = 0; i < n; i += 1) {
      const p0 = points[(i - 1 + n) % n], p1 = points[i], p2 = points[(i + 1) % n], p3 = points[(i + 2) % n];
      const seg = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      const k = Math.max(1, Math.round(seg / step));
      for (let j = 0; j < k; j += 1) {
        const t = j / k, t2 = t * t, t3 = t2 * t;
        const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
        out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
      }
    }
    return out;
  }

  function buildTrack(def) {
    const c = resample(def.center, 1.5);
    const n = c.length;
    const w = def.width;
    const left = [], right = [], tang = [], s = [0];
    for (let i = 0; i < n; i += 1) {
      const a = c[(i - 1 + n) % n], b = c[(i + 1) % n];
      let tx = b[0] - a[0], ty = b[1] - a[1];
      const l = Math.hypot(tx, ty) || 1; tx /= l; ty /= l;
      tang.push(Math.atan2(ty, tx));
      left.push([c[i][0] - ty * w / 2, c[i][1] + tx * w / 2]);
      right.push([c[i][0] + ty * w / 2, c[i][1] - tx * w / 2]);
      if (i > 0) s.push(s[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));
    }
    const length = s[n - 1] + Math.hypot(c[0][0] - c[n - 1][0], c[0][1] - c[n - 1][1]);
    const curv = [];
    for (let i = 0; i < n; i += 1) {
      let d = tang[(i + 1) % n] - tang[(i - 1 + n) % n];
      d = ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      curv.push(d / 3);
    }
    return { c, left, right, tang, s, length, curv, width: w, n };
  }

  function at(track, dist) {
    const L = track.length;
    let d = ((dist % L) + L) % L;
    let lo = 0, hi = track.n - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (track.s[mid] <= d) lo = mid; else hi = mid - 1; }
    return lo;
  }

  function nearestOff(track, x, y) {
    let best = Infinity;
    for (let i = 0; i < track.n; i += 2) {
      const dx = track.c[i][0] - x, dy = track.c[i][1] - y;
      const d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }

  // ---------- car (racing/style.py car_polygons) ----------
  const mirror = (half) => half.concat(half.slice().reverse().filter((p) => p[1] !== 0).map((p) => [p[0], -p[1]]));
  const box = (x0, x1, y0, y1) => [[x1, y1], [x1, y0], [x0, y0], [x0, y1]];
  const CAR = [
    [box(0.8, 1.45, 0.62, 1.0), 'wheel'], [box(0.8, 1.45, -1.0, -0.62), 'wheel'],
    [box(-1.75, -0.95, 0.64, 1.05), 'wheel'], [box(-1.75, -0.95, -1.05, -0.64), 'wheel'],
    [box(1.55, 2.0, -1.0, 1.0), 'wing'], [box(-2.25, -1.8, -0.88, 0.88), 'wing'],
    [mirror([[2.25, 0], [1.75, 0.1], [0.95, 0.17], [0.75, 0.5], [-0.1, 0.52], [-0.35, 0.34], [-1.15, 0.26], [-1.85, 0.3], [-1.85, 0]]), 'body'],
    [[[0.45, 0], [0.3, 0.17], [-0.25, 0.2], [-0.35, 0], [-0.25, -0.2], [0.3, -0.17]], 'cockpit'],
  ];
  const CAR_COLORS = {
    alive: { body: '#e2e6eb', accent: '#bac1ca', wing: '#3c424c', wheel: '#12141a', cockpit: '#282e3a' },
    dead: { body: '#94a0ac', accent: '#808c98', wing: '#424e5c', wheel: '#222a34', cockpit: '#465260' },
  };

  function drawCar(ctx, x, y, heading, scale, kind) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(scale, -scale);
    ctx.rotate(heading);
    // soft drop shadow
    ctx.save();
    ctx.translate(0.25, -0.35);
    ctx.fillStyle = 'rgba(16, 24, 30, 0.28)';
    ctx.beginPath();
    ctx.ellipse(0, 0, 2.4, 1.15, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    const pal = kind === 'dead' ? CAR_COLORS.dead : CAR_COLORS.alive;
    for (const [pts, key] of CAR) {
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.closePath();
      ctx.fillStyle = key === 'body' && kind === 'leader' ? '#ffffff' : pal[key];
      ctx.fill();
    }
    ctx.restore();
  }

  // ---------- scene ----------
  function sizeCanvas(canvas) {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, W: r.width, H: r.height };
  }

  function drawScene(canvas, o) {
    const def = o.track ? (window.LAB_TRACKS[o.track] || o.trackDef) : o.trackDef;
    const track = buildTrack(def);
    const { ctx, W, H } = sizeCanvas(canvas);
    const r = rng(o.seed || 1);

    // fit
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of track.left.concat(track.right)) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    const pad = o.pad != null ? o.pad : 28;
    const padTop = o.padTop != null ? o.padTop : pad;
    const padLeft = o.padLeft != null ? o.padLeft : pad;
    const padBottom = o.padBottom != null ? o.padBottom : pad;
    let s = Math.min((W - padLeft - pad) / (maxX - minX), (H - padTop - padBottom) / (maxY - minY));
    let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    let ox = padLeft + (W - padLeft - pad) / 2, oy = padTop + (H - padTop - padBottom) / 2;
    const leaderS = (o.progress != null ? o.progress : 0.5) * track.length;
    const li = at(track, leaderS);
    if (o.follow) {
      s *= o.follow;
      cx = track.c[li][0]; cy = track.c[li][1];
    }
    const X = (x) => ox + (x - cx) * s;
    const Y = (y) => oy - (y - cy) * s;

    // grass with soft light bands (as in the reference video)
    ctx.fillStyle = css('--scene-grass') || '#7a8e9c';
    ctx.fillRect(0, 0, W, H);
    ctx.save();
    ctx.globalAlpha = 0.05;
    ctx.fillStyle = '#ffffff';
    for (let k = 0; k < 4; k += 1) {
      ctx.save();
      ctx.translate(W * (0.15 + 0.3 * k), H * (0.2 + 0.25 * (k % 2)));
      ctx.rotate(-0.5);
      ctx.fillRect(-W, -H * 0.08, W * 2, H * (0.1 + 0.05 * k));
      ctx.restore();
    }
    ctx.restore();

    const ring = (pts) => { pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.closePath(); };

    // road
    ctx.beginPath(); ring(track.left); ring(track.right);
    ctx.fillStyle = '#4f616e';
    ctx.fill('evenodd');
    // edge lines
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#fffffc';
    ctx.lineWidth = Math.max(1.4, 0.42 * s);
    ctx.beginPath(); ring(track.left); ctx.stroke();
    ctx.beginPath(); ring(track.right); ctx.stroke();

    // curbs on the inside of turns
    const curbW = 1.25;
    for (const side of [1, -1]) {
      let span = [];
      const flush = () => {
        if (span.length > 4) {
          const pts = span.map((i) => {
            const p = track.c[i], t = track.tang[i], off = side * (track.width / 2 - curbW / 2);
            return [X(p[0] - Math.sin(t) * off), Y(p[1] + Math.cos(t) * off)];
          });
          ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          ctx.lineCap = 'butt';
          ctx.lineWidth = curbW * s; ctx.strokeStyle = '#fafaf8'; ctx.setLineDash([]); ctx.stroke();
          ctx.strokeStyle = '#d62e3e'; ctx.setLineDash([2 * s, 2 * s]); ctx.stroke();
          ctx.setLineDash([]);
        }
        span = [];
      };
      for (let i = 0; i < track.n; i += 1) {
        if (side * track.curv[i] / 1.5 >= 0.02) span.push(i); else flush();
      }
      flush();
    }

    // start line (chequer)
    {
      const i = 0, p = track.c[i], t = track.tang[i];
      const nx = -Math.sin(t), ny = Math.cos(t), tx = Math.cos(t), ty = Math.sin(t);
      const cells = 8, cw = track.width / cells;
      for (let row = 0; row < 2; row += 1) {
        for (let k = 0; k < cells; k += 1) {
          const off = -track.width / 2 + k * cw;
          const along = row * cw;
          const q = [[0, 0], [cw, 0], [cw, cw], [0, cw]].map(([a, b]) => [p[0] + nx * (off + a) + tx * (along + b - cw), p[1] + ny * (off + a) + ty * (along + b - cw)]);
          ctx.beginPath(); q.forEach((v, j) => (j ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1])))); ctx.closePath();
          ctx.fillStyle = (k + row) % 2 ? '#181a20' : '#fafaf8';
          ctx.fill();
        }
      }
    }

    // trace of a demo lap, coloured by speed
    if (o.trace) {
      const upto = at(track, leaderS);
      ctx.lineCap = 'round';
      for (let i = 1; i <= upto; i += 1) {
        const k = Math.abs(track.curv[i]) * 40;
        const v = Math.max(0, Math.min(1, 1 - k));
        const off = Math.sin(i / 23) * track.width * 0.18;
        const p0 = track.c[i - 1], p1 = track.c[i], t = track.tang[i];
        ctx.strokeStyle = `hsl(${200 - v * 160}, 90%, ${62 + v * 6}%)`;
        ctx.lineWidth = Math.max(2, 0.55 * s);
        ctx.beginPath();
        ctx.moveTo(X(p0[0] - Math.sin(t) * off), Y(p0[1] + Math.cos(t) * off));
        ctx.lineTo(X(p1[0] - Math.sin(t) * off), Y(p1[1] + Math.cos(t) * off));
        ctx.stroke();
      }
    }

    const carScale = s * (o.carScale || 1);
    // crowd
    const n = o.cars || 0;
    const crashed = o.crashed || 0;
    const cars = [];
    for (let k = 0; k < crashed; k += 1) {
      const d = r() * leaderS * 0.9 + 4;
      const i = at(track, d);
      const side = r() < 0.5 ? -1 : 1;
      const off = side * track.width * (0.3 + r() * 0.14);
      const t = track.tang[i];
      cars.push({ x: track.c[i][0] - Math.sin(t) * off, y: track.c[i][1] + Math.cos(t) * off, h: t + side * (0.4 + r() * 0.6), kind: 'dead' });
    }
    for (let k = 0; k < n - crashed - 1; k += 1) {
      const back = Math.pow(r(), 1.7) * (o.spread || 60) + 3;
      const i = at(track, leaderS - back);
      const off = gauss(r) * track.width * 0.13;
      const t = track.tang[i];
      cars.push({ x: track.c[i][0] - Math.sin(t) * off, y: track.c[i][1] + Math.cos(t) * off, h: t + gauss(r) * 0.08, kind: 'alive' });
    }
    for (const car of cars) {
      if (car.kind === 'dead') drawCar(ctx, X(car.x), Y(car.y), car.h, carScale, 'dead');
    }
    for (const car of cars) {
      if (car.kind === 'dead' && o.crashMarks !== false) {
        const x = X(car.x), y = Y(car.y), m = Math.max(2.5, 0.7 * carScale);
        ctx.strokeStyle = 'rgba(214, 46, 62, 0.75)'; ctx.lineWidth = 1.4; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x - m, y - m); ctx.lineTo(x + m, y + m); ctx.moveTo(x + m, y - m); ctx.lineTo(x - m, y + m); ctx.stroke();
      }
    }
    for (const car of cars) if (car.kind === 'alive') drawCar(ctx, X(car.x), Y(car.y), car.h, carScale, 'alive');

    // leader with rays and brackets
    if (o.leader !== false && (n > 0 || o.single)) {
      const t = track.tang[li];
      const lx = track.c[li][0] - Math.sin(t) * track.width * 0.06, ly = track.c[li][1] + Math.cos(t) * track.width * 0.06;
      const rays = o.rays || [-90, -30, 0, 30, 90];
      ctx.strokeStyle = 'rgba(255, 255, 252, 0.92)';
      ctx.lineWidth = 1.2;
      for (const deg of rays) {
        const a = t + (deg * Math.PI) / 180;
        let d = 0;
        while (d < 40) { d += 0.5; if (nearestOff(track, lx + Math.cos(a) * d, ly + Math.sin(a) * d) > track.width / 2) break; }
        const ex = lx + Math.cos(a) * d, ey = ly + Math.sin(a) * d;
        ctx.beginPath(); ctx.moveTo(X(lx), Y(ly)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
        ctx.fillStyle = '#fffffc'; ctx.beginPath(); ctx.arc(X(ex), Y(ey), 2.2, 0, Math.PI * 2); ctx.fill();
      }
      drawCar(ctx, X(lx), Y(ly), t, carScale, 'leader');
      // corner brackets
      const b = Math.max(9, 3.1 * carScale), q = b * 0.45;
      const x = X(lx), y = Y(ly);
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.8; ctx.lineCap = 'square';
      ctx.beginPath();
      for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        ctx.moveTo(x + sx * b, y + sy * b - sy * q); ctx.lineTo(x + sx * b, y + sy * b); ctx.lineTo(x + sx * b - sx * q, y + sy * b);
      }
      ctx.stroke();
    }
    return { track, X, Y, s };
  }

  // ---------- network ----------
  const SVGNS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent) {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16)), pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
    return '#' + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, '0')).join('');
  }

  function drawNetwork(svg, o) {
    const W = svg.clientWidth || 360, H = svg.clientHeight || 300;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';
    const r = rng(o.seed || 3);
    const edit = o.mode === 'edit';
    const inLabels = o.inputs;
    const sizes = [inLabels.length].concat(o.hidden, [o.outputs.length]);
    const L = sizes.length;
    const fs = o.small ? 10.5 : 12;
    const gutL = o.small ? 46 : (edit ? 128 : 64), gutR = o.small ? 46 : (edit ? 120 : 58);
    const top = edit ? 84 : (o.small ? 26 : 34), bottom = edit ? 40 : (o.small ? 12 : 16);
    const xs = sizes.map((_, i) => gutL + ((W - gutL - gutR) * i) / (L - 1));
    const maxN = Math.max.apply(null, sizes);
    const gap = Math.min(edit ? (H > 700 ? 58 : 46) : (H > 420 ? 46 : 38), (H - top - bottom) / Math.max(1, maxN - 1 + (edit ? 1 : 0)));
    const nr = Math.max(4.5, Math.min(edit ? 11 : 9, gap * 0.3));
    const compact = !o.small && gap < 28, tiny = gap < 12;
    const ys = sizes.map((n) => Array.from({ length: n }, (_, j) => top + (H - top - bottom - gap * (n - 1)) / 2 + j * gap));
    const topY = Math.min.apply(null, ys.map((c) => c[0]));
    const botY = Math.max.apply(null, ys.map((c) => c[c.length - 1]));
    const off = new Set(o.disabled || []);
    // weights and forward pass
    let act = o.values || inLabels.map(() => r() * 2 - 1);
    act = act.map((v, i) => (off.has(i) ? 0 : v));
    const values = [act];
    const W8 = [];
    for (let l = 0; l < L - 1; l += 1) {
      const m = Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => gauss(r) * 0.9));
      W8.push(m);
      values.push(m.map((row) => Math.tanh(row.reduce((acc, w, i) => acc + w * values[l][i], 0) * 0.8)));
    }
    const pos = css('--net-edge-pos') || '#28c83c', neg = css('--net-edge-neg') || '#d63246';
    const gEdges = el('g', {}, svg);
    for (let l = 0; l < L - 1; l += 1) {
      const m = W8[l];
      const mx = Math.max.apply(null, m.flat().map(Math.abs));
      for (let j = 0; j < sizes[l + 1]; j += 1) {
        for (let i = 0; i < sizes[l]; i += 1) {
          if (l === 0 && off.has(i)) continue;
          const w = m[j][i], k = Math.abs(w) / mx;
          if (k < (l === 0 ? 0.22 : 0.12)) continue;
          el('line', {
            x1: xs[l], y1: ys[l][i], x2: xs[l + 1], y2: ys[l + 1][j],
            stroke: w > 0 ? pos : neg,
            'stroke-width': (0.6 + 3.4 * k * k).toFixed(2),
            'stroke-opacity': ((0.2 + 0.6 * k) * (l === 0 ? 0.7 : 1)).toFixed(2),
            'stroke-linecap': 'round',
          }, gEdges);
        }
      }
    }
    const text = (x, y, s, attrs, parent) => { const t = el('text', Object.assign({ x, y, 'font-size': fs, fill: '#3c424c', 'font-family': 'Onest Variable, sans-serif' }, attrs), parent || svg); t.textContent = s; return t; };
    const fmt = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2).replace('.', ',');
    // column titles
    const titles = sizes.map((_, i) => (i === 0 ? 'Вход' : i === L - 1 ? 'Выход' : (o.small ? `С${i}` : `Скрытый ${i}`)));
    titles.forEach((t, i) => text(xs[i], edit ? Math.max(22, topY - 78) : (o.small ? 14 : 18), t, { 'text-anchor': 'middle', fill: '#6e7782', 'font-size': fs - 0.5, 'font-weight': 500 }));
    // nodes
    const gNodes = el('g', {}, svg);
    for (let l = 0; l < L; l += 1) {
      for (let j = 0; j < sizes[l]; j += 1) {
        const v = values[l][j];
        const disabled = l === 0 && off.has(j);
        const x = xs[l], y = ys[l][j];
        const fill = disabled ? 'none' : mix('eceef0', v > 0 ? '5aff50' : 'fa3e52', Math.min(1, Math.abs(v) * 1.15));
        if (!disabled && Math.abs(v) > 0.7) el('circle', { cx: x, cy: y, r: nr + 4, fill: v > 0 ? '#5aff50' : '#fa3e52', opacity: 0.22 }, gNodes);
        const selected = edit && o.selected && o.selected[0] === l && o.selected[1] === j;
        el('circle', {
          cx: x, cy: y, r: nr, fill,
          stroke: selected ? '#c98a12' : (disabled ? '#9aa4ad' : '#3c4a52'),
          'stroke-width': selected ? 3 : 1.6,
          'stroke-dasharray': disabled ? '3 3' : 'none',
        }, gNodes);
        if (l === 0) {
          const lab = tiny ? null : text(x - nr - 8, y + (o.small || compact ? 4 : 0), inLabels[j], { 'text-anchor': 'end', 'font-weight': 500, fill: disabled ? '#8a939c' : '#3c424c', 'text-decoration': disabled ? 'line-through' : 'none' });
          if (!o.small && !compact) text(x - nr - 8, y + 13, disabled ? 'выкл.' : fmt(v), { 'text-anchor': 'end', 'font-size': 10.5, fill: '#6e7782', 'font-family': 'JetBrains Mono Variable, monospace' });
          else if (lab) lab.setAttribute('y', y + 4);
          if (edit) {
            const g = el('g', { class: 'net-btn', tabindex: 0, role: 'switch', 'aria-checked': String(!disabled), 'aria-label': `Вход ${inLabels[j]}` }, svg);
            el('rect', { x: 10, y: y - 11, width: 22, height: 22, rx: 5, fill: '#ffffff', stroke: '#dde3e7' }, g);
            el('use', { href: disabled ? '#i-eye-off' : '#i-eye', x: 14, y: y - 7, width: 14, height: 14, color: disabled ? '#8a939c' : '#3c424c' }, g);
          }
        }
        if (l === L - 1) {
          text(x + nr + 9, y + (o.small || compact ? 4 : 0), o.outputs[j], { 'font-weight': 500 });
          if (!o.small && !compact) text(x + nr + 9, y + 13, fmt(v), { 'font-size': 10.5, fill: '#6e7782', 'font-family': 'JetBrains Mono Variable, monospace' });
        }
      }
    }
    if (edit) drawEditAffordances(svg, o, { xs, ys, sizes, gap, nr, W, H, top, text, topY, botY });
  }

  function pill(svg, x, y, w, h, label, opts) {
    const g = el('g', { class: 'net-btn', tabindex: 0, role: 'button', 'aria-label': opts.aria || label }, svg);
    el('rect', { x: x - w / 2, y: y - h / 2, width: w, height: h, rx: h / 2, fill: opts.fill || '#ffffff', stroke: opts.stroke || '#cbd4da', 'stroke-dasharray': opts.dash || 'none' }, g);
    if (opts.icon) el('use', { href: '#i-' + opts.icon, x: x - 7, y: y - 7, width: 14, height: 14, color: opts.color || '#3c424c' }, g);
    if (label) { const t = el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 600, fill: opts.color || '#3c424c', 'font-family': 'JetBrains Mono Variable, monospace' }, g); t.textContent = label; }
    return g;
  }

  function drawEditAffordances(svg, o, g) {
    const { xs, ys, sizes, gap, nr, H, text, topY, botY } = g;
    const rowT = Math.max(22, topY - 78), rowP = Math.max(50, topY - 46);
    const L = sizes.length;
    // per hidden layer: stepper pill and delete
    for (let l = 1; l < L - 1; l += 1) {
      const x = xs[l];
      const y = rowP;
      const group = el('g', {}, svg);
      el('rect', { x: x - 44, y: y - 13, width: 88, height: 26, rx: 13, fill: '#ffffff', stroke: '#cbd4da' }, group);
      const minus = el('g', { class: 'net-btn', tabindex: 0, role: 'button', 'aria-label': `Убрать нейрон из слоя ${l}` }, group);
      el('rect', { x: x - 42, y: y - 11, width: 24, height: 22, rx: 11, fill: '#ffffff' }, minus);
      el('use', { href: '#i-minus', x: x - 37, y: y - 7, width: 14, height: 14, color: '#3c424c' }, minus);
      const t = el('text', { x, y: y + 4.5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, fill: '#3c424c', 'font-family': 'JetBrains Mono Variable, monospace' }, group);
      t.textContent = sizes[l];
      const plus = el('g', { class: 'net-btn', tabindex: 0, role: 'button', 'aria-label': `Добавить нейрон в слой ${l}` }, group);
      el('rect', { x: x + 18, y: y - 11, width: 24, height: 22, rx: 11, fill: '#ffffff' }, plus);
      el('use', { href: '#i-plus', x: x + 23, y: y - 7, width: 14, height: 14, color: '#3c424c' }, plus);
      const del = el('g', { class: 'net-btn', tabindex: 0, role: 'button', 'aria-label': `Удалить слой ${l}` }, svg);
      el('rect', { x: x + 50, y: y - 11, width: 22, height: 22, rx: 6, fill: '#ffffff', stroke: '#e3e8ec' }, del);
      el('use', { href: '#i-trash-2', x: x + 54, y: y - 7, width: 14, height: 14, color: '#8a939c' }, del);
      // ghost "add neuron" slot under the column
      const last = ys[l][sizes[l] - 1];
      el('circle', { cx: x, cy: last + gap, r: nr, fill: 'none', stroke: '#b5c0c8', 'stroke-dasharray': '3 3' }, svg);
      el('use', { href: '#i-plus', x: x - 6, y: last + gap - 6, width: 12, height: 12, color: '#8a939c' }, svg);
    }
    // insertion slots between columns
    for (let l = 0; l < L - 1; l += 1) {
      const x = (xs[l] + xs[l + 1]) / 2;
      const hot = o.hotSlot === l;
      if (hot) {
        el('rect', { x: x - 18, y: topY - 24, width: 36, height: botY - topY + 48, rx: 18, fill: 'rgba(242,179,61,0.10)', stroke: '#e0a12e', 'stroke-dasharray': '5 4' }, svg);
        pill(svg, x, (topY + botY) / 2, 112, 30, '', { icon: 'plus', aria: 'Вставить слой', fill: '#f2b33d', stroke: '#e0a12e', color: '#1a1406' });
        const t = el('text', { x: x + 6, y: (topY + botY) / 2 + 4.5, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 600, fill: '#1a1406', 'font-family': 'Onest Variable, sans-serif' }, svg);
        t.textContent = 'Новый слой';
        svg.lastChild.previousSibling.querySelector('use').setAttribute('x', x - 50);
      } else {
        const b = pill(svg, x, rowT - 4, 24, 24, '', { icon: 'plus', aria: 'Вставить слой здесь', dash: '3 3', stroke: '#b5c0c8', color: '#6e7782' });
        b.setAttribute('opacity', '0.9');
      }
    }
    // add ray
    const x0 = xs[0], lastIn = ys[0][sizes[0] - 1];
    el('circle', { cx: x0, cy: lastIn + gap, r: nr, fill: 'none', stroke: '#b5c0c8', 'stroke-dasharray': '3 3' }, svg);
    el('use', { href: '#i-plus', x: x0 - 6, y: lastIn + gap - 6, width: 12, height: 12, color: '#8a939c' }, svg);
    text(x0 - nr - 8, lastIn + gap + 4, 'Добавить вход', { 'text-anchor': 'end', fill: '#6e7782', 'font-size': 12 });
    // inactive outputs
    const xL = xs[L - 1];
    const lastOut = ys[L - 1][sizes[L - 1] - 1];
    (o.offOutputs || []).forEach((name, k) => {
      const y = lastOut + gap * (k + 1);
      el('circle', { cx: xL, cy: y, r: nr, fill: 'none', stroke: '#b5c0c8', 'stroke-dasharray': '3 3' }, svg);
      text(xL + nr + 9, y + 4, name, { fill: '#8a939c', 'font-weight': 500 });
      const gg = el('g', { class: 'net-btn', tabindex: 0, role: 'switch', 'aria-checked': 'false', 'aria-label': `Выход ${name}` }, svg);
      el('rect', { x: g.W - 32, y: y - 11, width: 22, height: 22, rx: 5, fill: '#ffffff', stroke: '#dde3e7' }, gg);
      el('use', { href: '#i-eye-off', x: g.W - 28, y: y - 7, width: 14, height: 14, color: '#8a939c' }, gg);
    });
    for (let j = 0; j < sizes[L - 1]; j += 1) {
      const y = ys[L - 1][j];
      const gg = el('g', { class: 'net-btn', tabindex: 0, role: 'switch', 'aria-checked': 'true', 'aria-label': `Выход ${o.outputs[j]}` }, svg);
      el('rect', { x: g.W - 32, y: y - 11, width: 22, height: 22, rx: 5, fill: '#ffffff', stroke: '#dde3e7' }, gg);
      el('use', { href: j === 0 ? '#i-lock' : '#i-eye', x: g.W - 28, y: y - 7, width: 14, height: 14, color: j === 0 ? '#8a939c' : '#3c424c' }, gg);
    }
    // selected node popover (HTML anchored by the page)
    if (o.selected && o.onSelected) {
      const [l, j] = o.selected;
      o.onSelected(xs[l], ys[l][j], nr);
    }
  }

  // ---------- charts ----------
  function curve(seed, n, top, tau, noise, start) {
    const r = rng(seed);
    const out = [];
    let best = start || 0.05;
    for (let i = 0; i < n; i += 1) {
      const target = top * (1 - Math.exp(-i / tau)) + (start || 0.05) * Math.exp(-i / tau);
      best = Math.max(best, target + (r() - 0.6) * noise);
      out.push(Math.min(1, best));
    }
    return out;
  }
  function meanCurve(best, seed) {
    const r = rng(seed);
    return best.map((v, i) => Math.max(0.02, v * (0.55 + 0.1 * Math.sin(i / 5)) + (r() - 0.5) * 0.05));
  }

  function drawChart(svg, o) {
    const W = svg.clientWidth || 600, H = svg.clientHeight || 160;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';
    const m = { l: 44, r: o.endLabels === false ? 16 : 64, t: 10, b: 26 };
    const n = Math.max.apply(null, o.series.map((s) => s.data.length));
    const xmax = o.xmax || n;
    const ymax = o.ymax || 1;
    const X = (i) => m.l + (i / (xmax - 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - v / ymax) * (H - m.t - m.b);
    const grid = css('--chart-grid'), axis = css('--chart-axis');
    const ticks = o.yticks || [0, 0.25, 0.5, 0.75, 1];
    for (const v of ticks) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: grid, 'stroke-width': 1 }, svg);
      const t = el('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: axis, 'font-family': 'JetBrains Mono Variable, monospace' }, svg);
      t.textContent = o.yfmt ? o.yfmt(v) : `${Math.round(v * 100)} %`;
    }
    const step = o.xstep || 10;
    for (let i = 0; i < xmax; i = i === 0 ? step - 1 : i + step) {
      if (o.xlabel && X(i) > W - m.r - 70) continue;
      const t = el('text', { x: X(i), y: H - 8, 'text-anchor': 'middle', 'font-size': 11, fill: axis, 'font-family': 'JetBrains Mono Variable, monospace' }, svg);
      t.textContent = String(i + 1);
    }
    if (o.xlabel) {
      const t = el('text', { x: W - m.r, y: H - 8, 'text-anchor': 'end', 'font-size': 11, fill: axis }, svg);
      t.textContent = o.xlabel;
    }
    for (const s of o.series) {
      if (s.hidden) continue;
      const d = s.data.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
      if (s.area) {
        el('path', { d: `${d}L${X(s.data.length - 1)},${Y(0)}L${X(0)},${Y(0)}Z`, fill: s.color, opacity: 0.08 }, svg);
      }
      el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash || 'none', opacity: s.dim ? 0.45 : 1 }, svg);
      const li = s.data.length - 1;
      if (!s.dash) {
        el('circle', { cx: X(li), cy: Y(s.data[li]), r: 3.5, fill: s.color, stroke: css('--surface-1'), 'stroke-width': 2 }, svg);
        if (o.endLabels !== false) {
          const t = el('text', { x: X(li) + 8, y: Y(s.data[li]) + 4 + (s.labelDy || 0), 'font-size': 12, 'font-weight': 600, fill: s.color, 'font-family': 'JetBrains Mono Variable, monospace' }, svg);
          t.textContent = o.yfmt ? o.yfmt(s.data[li]) : `${Math.round(s.data[li] * 100)} %`;
        }
      }
    }
    if (o.hover != null) {
      const i = o.hover;
      el('line', { x1: X(i), x2: X(i), y1: m.t, y2: H - m.b, stroke: axis, 'stroke-dasharray': '3 3' }, svg);
      const rows = o.series.filter((s) => !s.hidden && !s.dash && s.data[i] != null);
      rows.forEach((s) => el('circle', { cx: X(i), cy: Y(s.data[i]), r: 4, fill: s.color, stroke: css('--surface-1'), 'stroke-width': 2 }, svg));
      const bw = 176, bh = 30 + rows.length * 20;
      const bx = Math.min(X(i) + 12, W - bw - 8), by = m.t + 4;
      el('rect', { x: bx, y: by, width: bw, height: bh, rx: 8, fill: css('--surface-2'), stroke: css('--border-strong') }, svg);
      const head = el('text', { x: bx + 12, y: by + 19, 'font-size': 12, fill: css('--text-2') }, svg);
      head.textContent = `${o.xname || 'Поколение'} ${i + 1}`;
      rows.forEach((s, k) => {
        el('rect', { x: bx + 12, y: by + 30 + k * 20, width: 8, height: 8, rx: 2, fill: s.color }, svg);
        const t = el('text', { x: bx + 26, y: by + 38 + k * 20, 'font-size': 12, fill: css('--text') }, svg);
        t.textContent = s.name;
        const v = el('text', { x: bx + bw - 12, y: by + 38 + k * 20, 'font-size': 12, 'text-anchor': 'end', 'font-weight': 600, fill: css('--text'), 'font-family': 'JetBrains Mono Variable, monospace' }, svg);
        v.textContent = o.yfmt ? o.yfmt(s.data[i]) : `${Math.round(s.data[i] * 100)} %`;
      });
    }
  }

  function drawSpark(svg, data, color) {
    const W = 96, H = 24;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const d = data.map((v, i) => `${i ? 'L' : 'M'}${((i / (data.length - 1)) * (W - 4) + 2).toFixed(1)},${(H - 3 - v * (H - 6)).toFixed(1)}`).join('');
    el('path', { d, fill: 'none', stroke: color, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, svg);
  }

  // ---------- interactions ----------
  function wireToggles(root) {
    root.querySelectorAll('[data-seg]').forEach((seg) => {
      seg.addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (!b || !seg.contains(b) || b.disabled) return;
        seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      });
    });
    root.querySelectorAll('[aria-pressed]').forEach((b) => {
      if (b.closest('[data-seg]')) return;
      b.addEventListener('click', () => b.setAttribute('aria-pressed', String(b.getAttribute('aria-pressed') !== 'true')));
    });
    root.querySelectorAll('[role="radiogroup"]').forEach((g) => {
      g.addEventListener('click', (e) => {
        const b = e.target.closest('[role="radio"]');
        if (!b || b.getAttribute('aria-disabled') === 'true') return;
        g.querySelectorAll('[role="radio"]').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
      });
    });
    root.querySelectorAll('[role="tablist"]').forEach((tl) => {
      tl.addEventListener('click', (e) => {
        const b = e.target.closest('[role="tab"]');
        if (!b) return;
        tl.querySelectorAll('[role="tab"]').forEach((x) => {
          x.setAttribute('aria-selected', String(x === b));
          const panel = x.getAttribute('aria-controls');
          if (panel && document.getElementById(panel)) document.getElementById(panel).hidden = x !== b;
        });
      });
    });
    root.querySelectorAll('input[type="range"].range').forEach((input) => {
      const sync = () => {
        const p = ((input.value - input.min) / (input.max - input.min)) * 100;
        input.style.setProperty('--p', `${p}%`);
        const out = input.closest('.param') && input.closest('.param').querySelector('.value-field input');
        if (out && input.dataset.fmt !== 'none') out.value = input.dataset.fmt === 'int' ? input.value : String(input.value).replace('.', ',');
      };
      input.addEventListener('input', sync);
      const p = ((input.value - input.min) / (input.max - input.min)) * 100;
      input.style.setProperty('--p', `${p}%`);
    });
    document.addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea, select')) return;
      if (e.key === 't' || e.key === 'е') {
        const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
        document.documentElement.dataset.theme = cur === 'light' ? 'dark' : 'light';
        window.dispatchEvent(new Event('resize'));
      }
    });
  }

  window.Proto = { drawScene, drawNetwork, drawChart, drawSpark, curve, meanCurve, rng, css, params, wireToggles, buildTrack };

  window.addEventListener('DOMContentLoaded', () => {
    wireToggles(document);
    if (typeof window.render === 'function') {
      const run = () => window.render();
      document.fonts && document.fonts.ready ? document.fonts.ready.then(run) : run();
      let t;
      window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(run, 80); });
    }
  });
})();
