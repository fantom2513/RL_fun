// Number formatting by the interface rules (DESIGN.md §9.1): decimal comma, U+2212 minus,
// a no-break space before units and between thousands, integer percentages.

export const MINUS = '−';
export const NBSP = ' ';

function withUnit(text, unit) {
  return unit ? `${text}${NBSP}${unit}` : text;
}

function exponent(value) {
  const [mantissa, power] = value.toExponential().split('e');
  const m = Number(mantissa);
  const head = Number.isInteger(m) ? String(m) : m.toFixed(1).replace('.', ',');
  return `${head}e${Number(power) < 0 ? MINUS : ''}${Math.abs(Number(power))}`.replace(/^-/, MINUS);
}

// number(value, digits, unit): digits is a count of decimals or 'auto' (up to 3 significant
// decimals, exponent form for very small or very large values).
export function number(value, digits = 'auto', unit = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const v = Number(value);
  if (digits === 'auto') {
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return withUnit(exponent(v), unit);
    digits = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3;
    const trimmed = Number(v.toFixed(digits));
    return withUnit(number(trimmed, decimalsOf(trimmed)), unit);
  }
  const fixed = Math.abs(v).toFixed(digits);
  const [int, frac] = fixed.split('.');
  const grouped = int.length > 4 ? int.replace(/\B(?=(\d{3})+(?!\d))/g, NBSP) : int;
  const negative = v < 0 && Number(fixed) !== 0;
  return withUnit(`${negative ? MINUS : ''}${grouped}${frac ? `,${frac}` : ''}`, unit);
}

function decimalsOf(value) {
  const text = String(value);
  return text.includes('.') ? text.split('.')[1].length : 0;
}

// percent(0.613) -> "61 %"
export function percent(fraction, digits = 0) {
  return number(fraction * 100, digits, '%');
}

// Parse what a person typed into a number field: accepts a comma, U+2212 and spaces.
export function parse(text) {
  const clean = String(text).trim().replace(/[\s ]/g, '').replace(MINUS, '-').replace(',', '.');
  if (clean === '' || clean === '-') return NaN;
  return Number(clean);
}

// Logarithmic slider scale: positions 0…1 map onto min…max with equal steps per decade.
export function logScale(min, max) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  return {
    toValue(position) {
      const p = Math.min(1, Math.max(0, position));
      if (p === 0) return min;
      if (p === 1) return max;
      return 10 ** (lo + p * (hi - lo));
    },
    toPosition(value) {
      return (Math.log10(Math.min(max, Math.max(min, value))) - lo) / (hi - lo);
    },
    ticks() {
      const out = [];
      for (let e = Math.ceil(lo); e <= Math.floor(hi); e += 1) out.push(10 ** e);
      return out;
    },
  };
}

// "10⁻⁵" style label for a power of ten.
export function powerLabel(value) {
  const sup = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const e = Math.round(Math.log10(value));
  return `10${String(e).split('').map((c) => sup[c]).join('')}`;
}
