// Challenges screen: the levels grouped by theme, with stars, goals and a start button.

import { h, icon } from '../core/dom.js';
import { number } from '../core/format.js';
import { LEVELS, byId, evaluate, isUnlocked, totalStars } from '../core/levels.js';

const UNITS = { finish_iter: 'итер.', lap_seconds: 'с' };
const GOAL_TITLE = { finish_iter: 'Итераций до первого круга', lap_seconds: 'Лучший круг, с' };
const TRACK_NAMES = { oval: 'Овал', wavy: 'Волны', circuit: 'Гран-при' };
const LEARNER_NAMES = { ppo: 'только PPO', evolution: 'только эволюция' };

const formatValue = (level, value) => (level.metric === 'lap_seconds' ? number(value, 2) : number(value, 0));

function constraints(level) {
  const items = [TRACK_NAMES[level.track] ?? level.track];
  if (level.learner) items.push(LEARNER_NAMES[level.learner] ?? level.learner);
  if (level.maxRays != null) items.push(`лучей не больше ${level.maxRays}`);
  if (level.maxHidden != null) items.push(`скрытых нейронов не больше ${level.maxHidden}`);
  return items;
}

function starRow(count, large = false) {
  const stars = [0, 1, 2].map((index) => {
    const star = icon('star');
    if (index < count) star.classList.add('on');
    return star;
  });
  return h('span', { class: `stars${large ? ' stars-lg' : ''}`, role: 'img', 'aria-label': `Звёзд: ${count} из 3` }, ...stars);
}

// The best result any run has reached on the level so far (the metric value, not the stars).
function bestValue(level, runs) {
  let best = null;
  for (const run of runs) {
    const { counts, value } = evaluate(level, run);
    if (counts && value != null && (best === null || value < best)) best = value;
  }
  return best;
}

function card(level, stars, runs, onStart) {
  const unlocked = isUnlocked(level, stars);
  const earned = stars[level.id] ?? 0;
  const value = bestValue(level, runs);
  const limits = level.stars.map((limit, index) => `${'★'.repeat(index + 1)} ≤ ${formatValue(level, limit)}`).join(' · ');
  const requirement = level.requires
    ? `Откроется после «${byId(level.requires.level).title}»: ${level.requires.stars} из 3 звёзд.`
    : '';
  const start = h('button', {
    type: 'button', class: 'btn btn-sm', disabled: !unlocked,
    'aria-label': `Начать попытку: ${level.title}`, onclick: () => onStart(level),
  }, 'Начать попытку');
  return h('article', { class: `card level${unlocked ? '' : ' is-locked'}`, 'data-level': level.id },
    h('div', { class: 'card-head' },
      h('span', { class: 'card-title' }, unlocked ? null : icon('lock'), level.title),
      starRow(earned)),
    h('div', { class: 'card-body level-body' },
      h('p', null, level.goal),
      h('div', { class: 'level-tags' }, ...constraints(level).map((text) => h('span', { class: 'tag' }, text))),
      h('p', { class: 'muted level-limits' }, `${GOAL_TITLE[level.metric]}: ${limits}`),
      value != null ? h('p', { class: 'level-best' }, `Сейчас лучший результат: ${formatValue(level, value)} ${UNITS[level.metric]}`) : null,
      unlocked ? h('p', { class: 'muted' }, level.hint) : h('p', { class: 'muted' }, requirement)),
    h('div', { class: 'card-foot' }, start));
}

export function renderChallenges(container, { stars, runs, onStart }) {
  const groups = [...new Set(LEVELS.map((level) => level.group))];
  container.replaceChildren(
    ...groups.map((group) => h('section', { class: 'level-group' },
      h('h2', { class: 'level-group-title' }, group),
      h('div', { class: 'level-grid' },
        ...LEVELS.filter((level) => level.group === group).map((level) => card(level, stars, runs, onStart))))));
}

export const starSummary = (stars) => `Звёзд: ${totalStars(stars)} из ${LEVELS.length * 3}`;
