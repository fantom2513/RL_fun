// The game layer: levels (challenges) with a goal, constraints and three star thresholds.
// A level is judged from any run that fits its constraints, so there is nothing to attach to a
// run: start the level (the form gets its track and limits) and train as usual.

export const STEP_SECONDS = 1 / 30;
const RAY_PREFIX = 'ray:';

// metric `finish_iter`: iterations until the first finished lap (fewer is better);
// metric `lap_seconds`: the fastest lap in seconds (less is better). `stars` are the limits for 1, 2, 3.
export const LEVELS = [
  {
    id: 'oval-first', group: 'Первые шаги', title: 'Первый круг', track: 'oval', metric: 'finish_iter',
    stars: [80, 40, 20], goal: 'Научите машинку проехать круг по овалу.',
    hint: 'Хватит настроек по умолчанию. Чем меньше итераций до первого круга, тем больше звёзд.',
  },
  {
    id: 'wavy-first', group: 'Первые шаги', title: 'Волны', track: 'wavy', metric: 'finish_iter',
    stars: [120, 60, 30], requires: { level: 'oval-first', stars: 1 },
    goal: 'Проедьте круг по трассе с плавными поворотами в обе стороны.',
    hint: 'Если машинки не учатся, увеличьте популяцию или попробуйте PPO.',
  },
  {
    id: 'circuit-first', group: 'Первые шаги', title: 'Гран-при', track: 'circuit', metric: 'finish_iter',
    stars: [200, 100, 40], requires: { level: 'wavy-first', stars: 1 },
    goal: 'Пройдите первый круг на длинной трассе с резкими поворотами.',
    hint: 'Награда за прогресс и центрирование помогает не вылетать на поворотах.',
  },
  {
    id: 'oval-fast', group: 'Скорость', title: 'Скоростной овал', track: 'oval', metric: 'lap_seconds',
    stars: [20, 17, 15.8], requires: { level: 'oval-first', stars: 1 },
    goal: 'Проедьте овал как можно быстрее: пределу в 15,2 с мешает только скорость машинки.',
    hint: 'Вес «скорости» в награде и дальше обучение после первого круга сокращают время.',
  },
  {
    id: 'wavy-fast', group: 'Скорость', title: 'Быстрые волны', track: 'wavy', metric: 'lap_seconds',
    stars: [27.3, 22.4, 20.3], requires: { level: 'wavy-first', stars: 2 },
    goal: 'Лучший круг по волнам: предел — 19,5 с.', hint: 'Нужно срезать повороты, не вылетая с трассы.',
  },
  {
    id: 'circuit-fast', group: 'Скорость', title: 'Квалификация', track: 'circuit', metric: 'lap_seconds',
    stars: [43, 35.2, 31.8], requires: { level: 'circuit-first', stars: 2 },
    goal: 'Лучший круг на Гран-при: предел — 30,6 с.',
    hint: 'Обучайте дольше первого круга: время продолжает падать, пока сеть учится срезать углы.',
  },
  {
    id: 'sharp-eye', group: 'Мастер', title: 'Три луча', track: 'circuit', metric: 'lap_seconds',
    stars: [60, 45, 36], maxRays: 3, requires: { level: 'circuit-first', stars: 1 },
    goal: 'Проедьте Гран-при, имея всего три луча дальномера.',
    hint: 'Лучи вперёд и по бокам, скорость как вход помогает поворачивать вовремя.',
  },
  {
    id: 'tiny-brain', group: 'Мастер', title: 'Минимализм', track: 'oval', metric: 'finish_iter',
    stars: [150, 80, 40], maxHidden: 4, requires: { level: 'oval-fast', stars: 1 },
    goal: 'Проедьте овал сетью, у которой в скрытых слоях не больше четырёх нейронов.',
    hint: 'Один скрытый слой из 3–4 нейронов уже умеет рулить.',
  },
  {
    id: 'rl-pilot', group: 'Мастер', title: 'Пилот с подкреплением', track: 'circuit', metric: 'finish_iter',
    stars: [80, 40, 25], learner: 'ppo', requires: { level: 'circuit-first', stars: 1 },
    goal: 'Пройдите Гран-при обучателем PPO.', hint: 'PPO учится на каждом шаге: итерации идут быстро.',
  },
];

export const byId = (id) => LEVELS.find((level) => level.id === id) ?? null;

const isRay = (name) => name.startsWith(RAY_PREFIX);

// Does a run's config satisfy the level's track and limits?
export function fits(level, config) {
  if (!config || config.track !== level.track) return false;
  if (level.learner && (config.learner ?? 'evolution') !== level.learner) return false;
  const inputs = config.model?.inputs ?? [];
  if (level.maxRays != null && inputs.filter(isRay).length > level.maxRays) return false;
  if (level.maxHidden != null) {
    const hidden = (config.model?.hidden ?? []).reduce((sum, size) => sum + size, 0);
    if (hidden > level.maxHidden) return false;
  }
  return true;
}

// Numbers a run has reached so far: iterations to the first lap and the fastest lap in seconds.
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
  return { finish_iter: firstLap, lap_seconds: fastest };
}

export function starsFor(level, value) {
  if (value == null) return 0;
  return level.stars.filter((limit) => value <= limit).length;
}

// Result of a run on a level: whether it counts, the value of the level's metric and the stars.
export function evaluate(level, run) {
  if (!run.config || !fits(level, run.config)) return { counts: false, value: null, stars: 0 };
  const value = metrics(run.gens)[level.metric];
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

export function isUnlocked(level, stars) {
  return !level.requires || (stars[level.requires.level] ?? 0) >= level.requires.stars;
}

export const totalStars = (stars) => Object.values(stars).reduce((sum, count) => sum + count, 0);

// A starting config for a level: the level's track, learner and limits on top of the defaults.
export function startConfig(level, defaults) {
  const config = structuredClone(defaults);
  config.track = level.track;
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
