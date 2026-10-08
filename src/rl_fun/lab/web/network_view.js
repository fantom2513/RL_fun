// SVG graph of the leader's network: the browser twin of the pygame NetworkPanel in
// rl_fun/racing/fleet_view.py. Columns "Вход / Скрытый N / Выход", nodes coloured by the sign of the
// activation (stronger = more saturated, a glow when strongly active), S-shaped edges coloured by
// the sign of the weight with thickness and opacity by |w|, weak edges hidden (8 % of the layer's
// largest |w|, 15 % for the input layer), values printed under the input and output labels.
//
// The SVG elements are created once per structure (layer sizes + labels) and positioned once per
// container size; every frame only changes attributes whose value actually changed, and at most
// once per animation frame however many SSE frames arrive.

const SVG_NS = 'http://www.w3.org/2000/svg';

const NODE_RADIUS = 11;
const MIN_NODE_RADIUS = 4;
const TITLE_Y = 16;
const PLOT_TOP = 32;
const PLOT_BOTTOM_PAD = 8;
const MIN_GUTTER = 54;
const CHAR_WIDTH = 7.2; // average label glyph width at 13 px, used to size the label gutters
const EDGE_CURVE = 0.42; // share of the column gap used as horizontal tangent: 0 = straight lines
const EDGE_THRESHOLD = 0.08;
const EDGE_THRESHOLD_FIRST = 0.15;
const EDGE_FADE_FIRST = 0.6;
const GLOW_THRESHOLD = 0.5;
const INLINE_VALUE_SPACING = 30; // px between nodes below which a value shares its label's line
const VALUE_WIDTH = 34; // width reserved for a value like "−0.42" in 11 px monospace
const SHORT_TITLE_GAP = 70; // px between columns below which hidden-layer titles are abbreviated

const plural = (count, one, few, many) => {
  const last = count % 10;
  if (count % 100 >= 11 && count % 100 <= 14) return many;
  if (last === 1) return one;
  return last >= 2 && last <= 4 ? few : many;
};

const LEGEND_ENTRIES = [
  ['positive', 'Зелёная линия — вес положительный (усиливает)'],
  ['negative', 'Красная линия — вес отрицательный (ослабляет)'],
  ['thickness', 'Толщина линии — сила связи'],
  ['node', 'Узел: зелёный — плюс, красный — минус, яркость — сила сигнала'],
];

function svg(name, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  node.append(...children);
  return node;
}

const setAttr = (node, name, value) => {
  const text = String(value);
  if (node.getAttribute(name) !== text) node.setAttribute(name, text);
};

const minus = (text) => text.replace('-', '−');

// SVG elements have no `hidden` property, only the attribute (the stylesheet hides [hidden]).
const setHidden = (node, hidden) => node.toggleAttribute('hidden', hidden);

export function columnTitles(layerCount) {
  if (layerCount <= 1) return ['Вход'].slice(0, layerCount);
  const hidden = Array.from({ length: layerCount - 2 }, (_, index) => `Скрытый ${index + 1}`);
  return ['Вход', ...hidden, 'Выход'];
}

// Labels of a model's inputs and outputs as the game prints them: «↑ −90°», «Скор.», «Руль»…
export function modelLabels(model, catalog) {
  const prefix = catalog.ray?.prefix ?? 'ray:';
  const scalar = new Map(catalog.inputs.map((item) => [item.id, item.label]));
  const outputs = new Map(catalog.outputs.map((item) => [item.id, item.label]));
  return {
    inputs: model.inputs.map((id) =>
      id.startsWith(prefix) ? `↑ ${minus(String(Number(id.slice(prefix.length))))}°` : (scalar.get(id) ?? id),
    ),
    outputs: model.outputs.map((id) => outputs.get(id) ?? id),
  };
}

function legendSwatch(kind) {
  const edge = (cls, y0, y1, width) =>
    svg('path', { class: cls, d: `M3 ${y0} C16 ${y0} 20 ${y1} 33 ${y1}`, 'stroke-width': width, fill: 'none', 'stroke-linecap': 'round' });
  const node = (cls, x) =>
    svg('g', {}, svg('circle', { class: 'net-ring', cx: x, cy: 10, r: 7 }), svg('circle', { class: `net-tint ${cls}`, cx: x, cy: 10, r: 5 }));
  const content = {
    positive: [edge('net-edge pos', 15, 5, 2.4)],
    negative: [edge('net-edge neg', 5, 15, 2.4)],
    thickness: [0.9, 3.1, 5.3].map((width, index) =>
      svg('path', { class: 'net-edge neutral', d: `M3 ${4 + index * 6} H33`, 'stroke-width': width, 'stroke-linecap': 'round' }),
    ),
    node: [node('pos', 10), node('neg', 26)],
  }[kind];
  return svg('svg', { class: 'net-swatch', viewBox: '0 0 36 20', width: 36, height: 20, 'aria-hidden': 'true' }, ...content);
}

export class NetworkView {
  // `body` holds the graph, `legend` is an empty <ul> that receives the explanations.
  constructor(body, legend) {
    this.body = body;
    this.svg = svg('svg', { class: 'net-svg', role: 'img', preserveAspectRatio: 'none' });
    this.message = document.createElement('p');
    this.message.className = 'net-message';
    this.message.setAttribute('role', 'status');
    this.body.replaceChildren(this.svg, this.message);
    this.fillLegend(legend);

    this.labels = { inputs: [], outputs: [] };
    this.net = null;
    this.key = '';
    this.sizes = [];
    this.layers = [];
    this.edgeLayers = [];
    this.titles = [];
    this.width = 0;
    this.height = 0;
    this.frameRequest = 0;
    this.setMessage(null);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.body);
    this.resize();
  }

  fillLegend(legend) {
    legend.replaceChildren(
      ...LEGEND_ENTRIES.map(([kind, text]) => {
        const item = document.createElement('li');
        const label = document.createElement('span');
        label.textContent = text;
        item.append(legendSwatch(kind), label);
        return item;
      }),
    );
  }

  // ---- public API -------------------------------------------------------------------------

  // Input and output captions of the shown run; changing them rebuilds the graph.
  setLabels(labels) {
    const next = { inputs: [...(labels?.inputs ?? [])], outputs: [...(labels?.outputs ?? [])] };
    if (JSON.stringify(next) === JSON.stringify(this.labels)) return;
    this.labels = next;
    this.key = '';
    if (this.net) this.schedule();
  }

  // The latest `frame.net` ({matrices, activations}); null keeps whatever is drawn.
  setNet(net) {
    if (!net?.activations?.length) return;
    this.net = net;
    this.schedule();
  }

  // Forgets the drawn network (another run is shown) and removes the graph from the screen.
  clear() {
    this.net = null;
    this.key = '';
    setHidden(this.svg, true);
  }

  // Text over the panel (empty, loading and error states); null shows the graph again.
  setMessage(text) {
    this.message.hidden = !text;
    this.message.textContent = text ?? '';
    setHidden(this.svg, Boolean(text) || !this.net);
  }

  // ---- scheduling -------------------------------------------------------------------------

  schedule() {
    if (this.frameRequest) return;
    this.frameRequest = requestAnimationFrame(() => {
      this.frameRequest = 0;
      this.flush();
    });
  }

  flush() {
    const net = this.net;
    if (!net) return;
    const sizes = net.activations.map((layer) => layer.length);
    const key = `${sizes.join('-')}|${this.labels.inputs.join(',')}|${this.labels.outputs.join(',')}`;
    if (key !== this.key) {
      this.key = key;
      this.build(sizes);
    }
    if (this.message.hidden) setHidden(this.svg, false);
    this.update(net);
  }

  resize() {
    const { clientWidth: width, clientHeight: height } = this.body;
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    if (this.layers.length) this.layout();
  }

  // ---- structure (once per change) --------------------------------------------------------

  build(sizes) {
    this.sizes = sizes;
    const titles = columnTitles(sizes.length);
    const edgeGroup = svg('g', { class: 'net-edges' });
    const nodeGroup = svg('g', { class: 'net-nodes' });
    const textGroup = svg('g', { class: 'net-texts' });

    this.titles = titles.map((title) => {
      const node = svg('text', { class: 'net-title', 'text-anchor': 'middle' });
      node.textContent = title;
      textGroup.append(node);
      return node;
    });

    this.edgeLayers = sizes.slice(0, -1).map((inCount, layer) => {
      const outCount = sizes[layer + 1];
      const edges = [];
      for (let out = 0; out < outCount; out += 1) {
        for (let from = 0; from < inCount; from += 1) {
          const path = svg('path', { class: 'net-edge pos', fill: 'none', 'stroke-linecap': 'round', 'stroke-opacity': 0 });
          edgeGroup.append(path);
          edges.push({ path, out, from });
        }
      }
      return edges;
    });

    this.layers = sizes.map((count, layer) => {
      const isInput = layer === 0;
      const isOutput = layer === sizes.length - 1;
      const labels = isInput ? this.labels.inputs : isOutput ? this.labels.outputs : null;
      return Array.from({ length: count }, (_, index) => {
        const glowOuter = svg('circle', { class: 'net-glow pos', 'fill-opacity': 0 });
        const glowInner = svg('circle', { class: 'net-glow pos', 'fill-opacity': 0 });
        const ring = svg('circle', { class: 'net-ring' });
        const tint = svg('circle', { class: 'net-tint pos', 'fill-opacity': 0 });
        nodeGroup.append(svg('g', {}, glowOuter, glowInner, ring, tint));
        const node = { glowOuter, glowInner, ring, tint, name: null, value: null, text: '' };
        if (labels) {
          node.name = svg('text', { class: 'net-label', 'text-anchor': isInput ? 'end' : 'start' });
          node.name.textContent = labels[index] ?? (isInput ? `Вход ${index + 1}` : `Выход ${index + 1}`);
          node.value = svg('text', { class: 'net-value', 'text-anchor': isInput ? 'end' : 'start' });
          textGroup.append(node.name, node.value);
        }
        return node;
      });
    });

    this.svg.replaceChildren(edgeGroup, nodeGroup, textGroup);
    const [first, last] = [sizes[0], sizes.at(-1)];
    const summary = sizes.length > 1
      ? `${first} ${plural(first, 'вход', 'входа', 'входов')} → ${sizes.slice(1, -1).map((n) => `${n} → `).join('')}${last} ${plural(last, 'выход', 'выхода', 'выходов')}`
      : '';
    this.svg.setAttribute('aria-label', `Нейросеть лидера: ${summary}. Значения обновляются вместе с трассой.`);
    this.layout();
  }

  // Room to the left of the inputs / right of the outputs for their captions.
  gutter(labels, inline) {
    const longest = labels.length ? Math.max(...labels.map((label) => label.length)) : 0;
    const value = inline ? VALUE_WIDTH + 4 : 0;
    return Math.max(MIN_GUTTER, NODE_RADIUS + 10 + longest * CHAR_WIDTH + value);
  }

  // Positions of everything for the current size; runs on structure change and resize only.
  layout() {
    const { width, height, sizes } = this;
    if (!sizes.length || width < 40 || height < 40) return;
    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const top = PLOT_TOP;
    const bottom = Math.max(top + 40, height - PLOT_BOTTOM_PAD);
    const spacings = sizes.map((count) => (bottom - top) / count);
    // With many inputs a two-line caption no longer fits between neighbours: the value moves
    // next to the node on the caption's line.
    const inline = spacings.map((spacing) => spacing < INLINE_VALUE_SPACING);
    const left = this.gutter(this.labels.inputs, inline[0]);
    const right = this.gutter(this.labels.outputs, inline.at(-1));
    const span = Math.max(width - left - right, 1);
    const xs = sizes.map((_, layer) => (sizes.length === 1 ? width / 2 : left + (span * layer) / (sizes.length - 1)));

    this.positions = sizes.map((count, layer) => {
      const spacing = spacings[layer];
      return { x: xs[layer], spacing, inline: inline[layer], ys: Array.from({ length: count }, (_, index) => top + (index + 0.5) * spacing) };
    });

    // Narrow columns would make "Скрытый 2" and its neighbour collide: abbreviate them then.
    const crowded = sizes.length > 2 && span / (sizes.length - 1) < SHORT_TITLE_GAP;
    this.titles.forEach((title, layer) => {
      const full = columnTitles(sizes.length)[layer];
      const text = crowded ? full.replace('Скрытый', 'Скр.') : full;
      if (title.textContent !== text) title.textContent = text;
      setAttr(title, 'x', xs[layer]);
      setAttr(title, 'y', TITLE_Y);
    });

    this.layers.forEach((nodes, layer) => {
      const { x, spacing, ys, inline } = this.positions[layer];
      const radius = Math.max(MIN_NODE_RADIUS, Math.min(NODE_RADIUS, spacing / 2 - 2));
      const scale = inline ? Math.max(0.72, Math.min(1, spacing / 18)) : Math.max(0.72, Math.min(1, spacing / 28));
      const isInput = layer === 0;
      nodes.forEach((node, index) => {
        const y = ys[index];
        for (const [circle, r] of [
          [node.glowOuter, radius + 5],
          [node.glowInner, radius + 2.5],
          [node.ring, radius],
          [node.tint, Math.max(radius - 2.2, 1)],
        ]) {
          setAttr(circle, 'cx', x.toFixed(1));
          setAttr(circle, 'cy', y.toFixed(1));
          setAttr(circle, 'r', r.toFixed(1));
        }
        if (node.name) {
          const anchor = isInput ? x - radius - 8 : x + radius + 8;
          const side = isInput ? -1 : 1;
          // [element, dx from the anchor, dy from the node centre, font size]
          const places = inline
            ? [
                [node.value, 0, 4 * scale, 11 * scale],
                [node.name, side * (VALUE_WIDTH + 4), 4.5 * scale, 13 * scale],
              ]
            : [
                [node.name, 0, -1.5 * scale, 13 * scale],
                [node.value, 0, 11 * scale, 11 * scale],
              ];
          for (const [text, dx, dy, size] of places) {
            setAttr(text, 'x', (anchor + dx).toFixed(1));
            setAttr(text, 'y', (y + dy).toFixed(1));
            setAttr(text, 'font-size', size.toFixed(1));
          }
        }
      });
    });

    this.edgeLayers.forEach((edges, layer) => {
      const from = this.positions[layer];
      const to = this.positions[layer + 1];
      const reach = (to.x - from.x) * EDGE_CURVE;
      for (const edge of edges) {
        const y0 = from.ys[edge.from].toFixed(1);
        const y1 = to.ys[edge.out].toFixed(1);
        const x0 = from.x.toFixed(1);
        const x1 = to.x.toFixed(1);
        setAttr(edge.path, 'd', `M${x0} ${y0}C${(from.x + reach).toFixed(1)} ${y0} ${(to.x - reach).toFixed(1)} ${y1} ${x1} ${y1}`);
      }
    });
  }

  // ---- values (every frame; only attributes that changed are touched) -----------------------

  update(net) {
    this.layers.forEach((nodes, layer) => {
      const activations = net.activations[layer];
      nodes.forEach((node, index) => {
        const value = Number(activations[index]) || 0;
        const strength = Math.min(Math.abs(value), 1);
        const sign = value >= 0 ? 'pos' : 'neg';
        setAttr(node.tint, 'class', `net-tint ${sign}`);
        setAttr(node.tint, 'fill-opacity', (strength ** 0.7).toFixed(2));
        const glow = strength > GLOW_THRESHOLD;
        setAttr(node.glowOuter, 'class', `net-glow ${sign}`);
        setAttr(node.glowInner, 'class', `net-glow ${sign}`);
        setAttr(node.glowOuter, 'fill-opacity', glow ? 0.14 : 0);
        setAttr(node.glowInner, 'fill-opacity', glow ? 0.22 : 0);
        if (node.value) {
          const text = minus(value.toFixed(2));
          if (text !== node.text) {
            node.text = text;
            node.value.textContent = text;
          }
        }
      });
    });

    this.edgeLayers.forEach((edges, layer) => {
      const matrix = net.matrices[layer];
      if (!matrix) return;
      let largest = 0;
      for (const row of matrix) for (const weight of row) largest = Math.max(largest, Math.abs(weight));
      const threshold = layer === 0 ? EDGE_THRESHOLD_FIRST : EDGE_THRESHOLD;
      const fade = layer === 0 ? EDGE_FADE_FIRST : 1;
      for (const edge of edges) {
        const weight = matrix[edge.out]?.[edge.from] ?? 0;
        const strength = largest > 0 ? Math.abs(weight) / largest : 0;
        if (strength < threshold || largest <= 0) {
          setAttr(edge.path, 'stroke-opacity', 0);
          continue;
        }
        setAttr(edge.path, 'class', `net-edge ${weight > 0 ? 'pos' : 'neg'}`);
        setAttr(edge.path, 'stroke-width', (0.9 + 4.4 * strength).toFixed(1));
        setAttr(edge.path, 'stroke-opacity', (((70 + 150 * strength) / 255) * fade).toFixed(2));
      }
    });
  }
}
