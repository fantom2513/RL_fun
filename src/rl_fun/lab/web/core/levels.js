// The game layer: levels (challenges) with a goal, constraints and three star thresholds.
// A level is judged from any run that fits its constraints, so there is nothing to attach to a
// run: start the level (the form gets its track and limits) and train as usual.

export const STEP_SECONDS = 1 / 30;
const RAY_PREFIX = 'ray:';

// A level is a race: `laps` laps on `track` within `maxSteps` simulation steps (30 steps = 1 s), so a
// slow solution simply does not finish. Metric `race_seconds` is the fastest finish of the whole
// race in seconds, `finish_iter` the number of iterations until a car first finished it; the three
// `stars` are the limits for 1, 2 and 3 stars (smaller is better).
export const LEVELS = [
  {
    id: 'oval-two', group: 'Первые шаги', title: 'Два круга', track: 'oval', laps: 2, maxSteps: 1100,
    metric: 'race_seconds', stars: [33.5, 31, 30],
    goal: 'Проедьте два круга по овалу за 33,5 секунды. Лучшие сети делают это за 29,7 с.',
    hint: 'Хватит настроек по умолчанию, чтобы проехать; за последние звёзды учите дольше.',
  },
  {
    id: 'wavy-two', group: 'Первые шаги', title: 'Волны', track: 'wavy', laps: 2, maxSteps: 1260,
    metric: 'race_seconds', stars: [42, 38.5, 37.2], requires: { level: 'oval-two', stars: 1 },
    goal: 'Два круга по волнам за 42 секунды: повороты в обе стороны.',
    hint: 'Если машинки не учатся, увеличьте популяцию или попробуйте PPO.',
  },
  {
    id: 'circuit-two', group: 'Первые шаги', title: 'Гран-при', track: 'circuit', laps: 2, maxSteps: 2100,
    metric: 'race_seconds', stars: [70, 64, 61.5], requires: { level: 'wavy-two', stars: 1 },
    goal: 'Два круга Гран-при за 70 секунд. Предел машинки — 61,2 с.',
    hint: 'Награда за прогресс и центрирование помогает не вылетать на поворотах.',
  },
  {
    id: 'oval-three', group: 'Скорость', title: 'Овал: три круга', track: 'oval', laps: 3, maxSteps: 1500, maxIterations: 40,
    metric: 'race_seconds', stars: [50, 46.5, 45.3], requires: { level: 'oval-two', stars: 2 },
    goal: 'Три круга по овалу за 50 секунд, обучение — не дольше 40 итераций.',
    hint: 'Нужны езда на предельной скорости и быстрое обучение: больше популяция, выше вес «скорости» в награде.',
  },
  {
    id: 'wavy-three', group: 'Скорость', title: 'Волны: три круга', track: 'wavy', laps: 3, maxSteps: 1900,
    metric: 'race_seconds', stars: [63, 58.5, 56.8], requires: { level: 'wavy-two', stars: 2 },
    goal: 'Три круга по волнам за 63 секунды.',
    hint: 'Срезайте повороты по внутренней стороне, не вылетая с трассы.',
  },
  {
    id: 'circuit-three', group: 'Скорость', title: 'Квалификация', track: 'circuit', laps: 3, maxSteps: 3050,
    metric: 'race_seconds', stars: [101, 94, 91.5], requires: { level: 'circuit-two', stars: 2 },
    goal: 'Три круга Гран-при за 101 секунду. Предел — 91,8 с.',
    hint: 'Обучайте дольше: время падает, пока сеть учится срезать углы.',
  },
  {
    id: 'quick-learner', group: 'Мастер', title: 'Скороучка', track: 'wavy', laps: 2, maxSteps: 1260, maxIterations: 30,
    metric: 'finish_iter', stars: [30, 15, 8], requires: { level: 'wavy-two', stars: 1 },
    goal: 'Научите машинку проехать два круга по волнам за 42 секунды, потратив на обучение как можно меньше итераций.',
    hint: 'Больше популяция и сильнее мутация учат быстрее: подберите, что работает лучше.',
  },
  {
    id: 'sharp-eye', group: 'Мастер', title: 'Три луча', track: 'circuit', laps: 2, maxSteps: 2400,
    metric: 'race_seconds', stars: [80, 70, 64], maxRays: 3, requires: { level: 'circuit-two', stars: 1 },
    goal: 'Проедьте Гран-при дважды за 80 секунд, имея всего три луча дальномера.',
    hint: 'Лучи вперёд и по бокам; скорость как вход помогает поворачивать вовремя.',
  },
  {
    id: 'tiny-brain', group: 'Мастер', title: 'Минимализм', track: 'oval', laps: 2, maxSteps: 1100, maxHidden: 4,
    metric: 'race_seconds', stars: [36, 32, 30.5], requires: { level: 'oval-three', stars: 1 },
    goal: 'Два круга по овалу за 36 секунд сетью, у которой в скрытых слоях не больше четырёх нейронов.',
    hint: 'Один скрытый слой из 3–4 нейронов уже умеет рулить.',
  },
  {
    id: 'rl-pilot', group: 'Мастер', title: 'Пилот с подкреплением', track: 'circuit', laps: 2, maxSteps: 2200,
    metric: 'race_seconds', stars: [72, 66, 62.5], learner: 'ppo', requires: { level: 'circuit-two', stars: 1 },
    goal: 'Два круга Гран-при обучателем PPO за 72 секунды.',
    hint: 'PPO учится на каждом шаге: итерации идут быстро.',
  },
];

export const byId = (id) => LEVELS.find((level) => level.id === id) ?? null;

const isRay = (name) => name.startsWith(RAY_PREFIX);

// Does a run's config satisfy the level's track and limits?
export function fits(level, config) {
  if (!config || config.track !== level.track) return false;
  if ((config.laps ?? 1) !== level.laps || config.max_steps > level.maxSteps) return false;
  if (level.learner && (config.learner ?? 'evolution') !== level.learner) return false;
  const inputs = config.model?.inputs ?? [];
  if (level.maxRays != null && inputs.filter(isRay).length > level.maxRays) return false;
  if (level.maxHidden != null) {
    const hidden = (config.model?.hidden ?? []).reduce((sum, size) => sum + size, 0);
    if (hidden > level.maxHidden) return false;
  }
  return true;
}

// Numbers a run has reached so far: iterations to the first finished race and the fastest race in seconds.
export function metrics(gens) {
  let firstLap = null;
  let fastest = null;
  gens.forEach((gen, index) => {
    if (firstLap === null && gen.finished > 0) firstLap = index + 1;
    if (gen.best_lap_steps != null) {
      const seconds = gen.best_lap_steps * STEP_SECONDS;
      if (fastest === null || seconds < fastest) fastest = seconds;
    }
  });
  return { finish_iter: firstLap, race_seconds: fastest };
}

export function starsFor(level, value) {
  if (value == null) return 0;
  return level.stars.filter((limit) => value <= limit).length;
}

// Result of a run on a level: whether it counts, the value of the level's metric and the stars.
export function evaluate(level, run) {
  if (!run.config || !fits(level, run.config)) return { counts: false, value: null, stars: 0 };
  // A training budget counts only the first `maxIterations` iterations of the run.
  const gens = level.maxIterations ? run.gens.slice(0, level.maxIterations) : run.gens;
  const value = metrics(gens)[level.metric];
  return { counts: true, value, stars: starsFor(level, value) };
}

// Best stars per level over all runs, merged into the saved ones (stars are never taken away).
// Levels are in unlock order, and a locked level earns nothing until the one it needs has stars.
export function bestStars(runs, saved = {}) {
  const result = { ...saved };
  for (const level of LEVELS) {
    if (!isUnlocked(level, result)) continue;
    for (const run of runs) {
      const { stars } = evaluate(level, run);
      if (stars > (result[level.id] ?? 0)) result[level.id] = stars;
    }
  }
  return result;
}

// The best result any run has reached on the level so far (the metric value, not the stars).
export function bestValue(level, runs) {
  let best = null;
  for (const run of runs) {
    const { counts, value } = evaluate(level, run);
    if (counts && value != null && (best === null || value < best)) best = value;
  }
  return best;
}

export function isUnlocked(level, stars) {
  return !level.requires || (stars[level.requires.level] ?? 0) >= level.requires.stars;
}

export const totalStars = (stars) => Object.values(stars).reduce((sum, count) => sum + count, 0);

// A starting config for a level: the level's track, learner and limits on top of the defaults.
export function startConfig(level, defaults) {
  const config = structuredClone(defaults);
  config.track = level.track;
  config.laps = level.laps;
  config.max_steps = level.maxSteps;
  if (level.maxIterations) config.generations = level.maxIterations;
  config.name = level.title.slice(0, 40);
  if (level.learner) config.learner = level.learner;
  if (level.maxRays != null) {
    const scalars = config.model.inputs.filter((name) => !isRay(name));
    const angles = level.maxRays >= 3 ? [-45, 0, 45] : [0];
    config.model.inputs = [...angles.map((angle) => `${RAY_PREFIX}${angle}`), ...scalars];
  }
  if (level.maxHidden != null) config.model.hidden = [Math.min(4, level.maxHidden)];
  return config;
}

const STORAGE_KEY = 'lab.stars';

export function loadStars() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    return data && typeof data === 'object' ? data : {};
  } catch {
    return {};
  }
}

export function saveStars(stars) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stars));
  } catch {
    // storage may be unavailable; stars are then kept only until the page is closed
  }
}
