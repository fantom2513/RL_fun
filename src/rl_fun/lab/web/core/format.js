// Russian number formatting for the lab UI.

const MINUS = "−";
const NBSP = " ";

function groupThousands(digits) {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, NBSP);
}

function withUnit(text, unit) {
  return unit ? text + NBSP + unit : text;
}

function autoNumber(value) {
  if (value === 0) return "0";
  const sign = value < 0 ? MINUS : "";
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e6) {
    const [mantissa, exponent] = abs.toExponential(3).split("e");
    const trimmed = mantissa.includes(".") ? mantissa.replace(/\.?0+$/, "") : mantissa;
    const expSign = Number(exponent) < 0 ? MINUS : "";
    return `${sign}${trimmed.replace(".", ",")}e${expSign}${Math.abs(Number(exponent))}`;
  }
  const rounded = Number(abs.toPrecision(4));
  const decimals = Math.max(0, 3 - Math.floor(Math.log10(rounded)));
  const [whole, frac = ""] = rounded.toFixed(decimals).split(".");
  const fracTrimmed = frac.replace(/0+$/, "");
  return sign + groupThousands(whole) + (fracTrimmed ? "," + fracTrimmed : "");
}

export function number(value, digits, unit) {
  if (digits === "auto") return withUnit(autoNumber(value), unit);
  const fixed = Math.abs(value).toFixed(digits);
  const sign = value < 0 && Number(fixed) !== 0 ? MINUS : "";
  const [whole, frac] = fixed.split(".");
  const body = groupThousands(whole) + (frac !== undefined ? "," + frac : "");
  return withUnit(sign + body, unit);
}

export function percent(fraction) {
  const points = Math.round(fraction * 100);
  const sign = points < 0 ? MINUS : "";
  return sign + groupThousands(String(Math.abs(points))) + NBSP + "%";
}

// Slider position in [0, 1] <-> value on a log scale between lo and hi.
export function logScale(lo, hi) {
  const logLo = Math.log(lo);
  const span = Math.log(hi) - logLo;
  return {
    toValue(pos) {
      if (pos === 0) return lo;
      if (pos === 1) return hi;
      return Math.exp(logLo + pos * span);
    },
    toPosition(value) {
      return (Math.log(value) - logLo) / span;
    },
  };
}
