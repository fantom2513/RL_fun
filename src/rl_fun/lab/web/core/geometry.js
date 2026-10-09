// Pure 2D geometry helpers for the track editor. Points are [x, y] arrays.
// A closed polyline's segment i runs from points[i] to points[(i + 1) % n].

function normZero(v) {
  return v + 0; // turns -0 into 0
}

function round1(v) {
  return normZero(Math.round(v * 10) / 10);
}

function round3(v) {
  return normZero(Math.round(v * 1000) / 1000);
}

function cross(ux, uy, vx, vy) {
  return ux * vy - uy * vx;
}

export function snapPoint(point, step = 1) {
  return [
    normZero(Math.round(point[0] / step) * step),
    normZero(Math.round(point[1] / step) * step),
  ];
}

export function polylineLength(points) {
  const n = points.length;
  if (n < 2) return 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const a = points[i];
    const b = points[(i + 1) % n];
    total += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return total;
}

export function signedArea(points) {
  const n = points.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % n];
    sum += x1 * y2 - x2 * y1;
  }
  return sum / 2;
}

// Proper crossing only: touching endpoints and collinear overlaps do not count.
export function segmentsCross(a, b, c, d) {
  const d1 = cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]);
  const d2 = cross(b[0] - a[0], b[1] - a[1], d[0] - a[0], d[1] - a[1]);
  const d3 = cross(d[0] - c[0], d[1] - c[1], a[0] - c[0], a[1] - c[1]);
  const d4 = cross(d[0] - c[0], d[1] - c[1], b[0] - c[0], b[1] - c[1]);
  return d1 * d2 < 0 && d3 * d4 < 0;
}

export function selfIntersections(points) {
  const n = points.length;
  const pairs = [];
  if (n < 4) return pairs;
  for (let i = 0; i < n; i++) {
    // j = i + 1 is adjacent; starting at i + 2 skips it.
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue; // wrap-around adjacency
      const a = points[i];
      const b = points[(i + 1) % n];
      const c = points[j];
      const d = points[(j + 1) % n];
      if (segmentsCross(a, b, c, d)) pairs.push([i, j]);
    }
  }
  return pairs;
}

export function closestOnPolyline(points, point) {
  const n = points.length;
  if (n < 2) return null;
  let best = null;
  for (let i = 0; i < n; i++) {
    const a = points[i];
    const b = points[(i + 1) % n];
    const abx = b[0] - a[0];
    const aby = b[1] - a[1];
    const len2 = abx * abx + aby * aby;
    let t = 0;
    if (len2 > 0) {
      t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / len2;
      t = Math.min(1, Math.max(0, t));
    }
    const qx = a[0] + t * abx;
    const qy = a[1] + t * aby;
    const distance = Math.hypot(point[0] - qx, point[1] - qy);
    if (best === null || distance < best.distance) {
      best = { index: i, t, point: [qx, qy], distance };
    }
  }
  return best;
}

export function chaikin(points, rounds = 1, maxPoints = 400) {
  let current = points.map((p) => [p[0], p[1]]);
  if (current.length < 3) return current;
  for (let r = 0; r < rounds; r++) {
    const n = current.length;
    if (2 * n > maxPoints) break;
    const next = [];
    for (let i = 0; i < n; i++) {
      const [ax, ay] = current[i];
      const [bx, by] = current[(i + 1) % n];
      next.push([round3(0.75 * ax + 0.25 * bx), round3(0.75 * ay + 0.25 * by)]);
      next.push([round3(0.25 * ax + 0.75 * bx), round3(0.25 * ay + 0.75 * by)]);
    }
    current = next;
  }
  return current;
}

export function insertPoint(points, point) {
  const copy = points.map((p) => [p[0], p[1]]);
  const newPoint = [point[0], point[1]];
  const hit = closestOnPolyline(points, point);
  if (hit === null) {
    copy.push(newPoint);
    return copy;
  }
  copy.splice(hit.index + 1, 0, newPoint);
  return copy;
}

function sample(count, fn) {
  const pts = [];
  for (let k = 0; k < count; k++) {
    const t = (2 * Math.PI * k) / count;
    const [x, y] = fn(t);
    pts.push([round1(x), round1(y)]);
  }
  return pts;
}

export const TEMPLATES = {
  oval: () => sample(16, (t) => [70 * Math.cos(t), 40 * Math.sin(t)]),
  trefoil: () =>
    sample(24, (t) => {
      const r = 55 + 12 * Math.cos(3 * t);
      return [r * Math.cos(t), 0.8 * r * Math.sin(t)];
    }),
  bean: () =>
    sample(20, (t) => [75 * Math.cos(t), 38 * Math.sin(t) + 16 * Math.cos(t) ** 2]),
};
