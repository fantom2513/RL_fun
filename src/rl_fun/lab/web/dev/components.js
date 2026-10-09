import { h, icon } from '/static/core/dom.js';
import '/static/ui/param.js';
import '/static/ui/seg.js';
import '/static/ui/stepper.js';
import { toast } from '/static/ui/toast.js';

const root = document.getElementById('showcase');
const section = (title, ...children) =>
  root.append(h('section', { class: 'sc' }, h('h2', null, title), ...children));
const row = (...children) => h('div', { class: 'sc-row' }, ...children);
const options = (...items) => JSON.stringify(items.map(([value, label]) => ({ value, label })));

// The scheme switch flips data-theme on the root element.
const applyTheme = (value) => {
  document.documentElement.dataset.theme = value === 'light' ? 'light' : 'dark';
};
document.getElementById('scheme').addEventListener('seg-change', (event) => applyTheme(event.detail.value));
applyTheme('dark');

section('Кнопки (.btn)',
  row(
    h('button', { class: 'btn' }, 'Обычная'),
    h('button', { class: 'btn btn-primary' }, 'Главная'),
    h('button', { class: 'btn btn-ghost' }, 'Тихая'),
    h('button', { class: 'btn btn-danger' }, 'Опасная'),
    h('button', { class: 'btn btn-icon', 'aria-label': 'Играть' }, icon('play')),
    h('button', { class: 'btn btn-sm' }, 'Малая 28'),
    h('button', { class: 'btn btn-lg btn-primary' }, 'Большая 40'),
    h('button', { class: 'btn btn-primary is-loading' }, 'Загрузка'),
    h('button', { class: 'btn', disabled: true }, 'Недоступна'),
    h('button', { class: 'btn btn-primary', disabled: true }, 'Главная недоступна'),
  ),
);

section('Сегменты (.seg), переключатель, чипы',
  row(
    h('lab-seg', { label: 'Размер', value: 'b', options: options(['a', 'Один'], ['b', 'Два'], ['c', 'Три']) }),
    h('lab-seg', { label: 'Малый', size: 'sm', value: 'x', options: options(['x', '1×'], ['y', '2×'], ['z', 'max']) }),
    h('lab-seg', { label: 'Моноширинный', mono: true, value: 'x', options: options(['x', '0,5'], ['y', '1']) }),
    h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: true }), h('span', { class: 'track' }), 'Включено'),
    h('label', { class: 'switch' }, h('input', { type: 'checkbox' }), h('span', { class: 'track' }), 'Выключено'),
    h('label', { class: 'switch' }, h('input', { type: 'checkbox', disabled: true }), h('span', { class: 'track' }), 'Недоступно'),
  ),
  row(
    h('button', { class: 'chip', 'aria-pressed': 'false' }, 'Выкл'),
    h('button', { class: 'chip', 'aria-pressed': 'true' }, 'Вкл'),
    h('button', { class: 'chip is-locked', 'aria-pressed': 'true' }, icon('lock'), 'Закреплён'),
    h('button', { class: 'chip chip-dashed' }, icon('plus'), 'Добавить'),
  ),
);

section('Параметры (.param), поле числа, шаговый счётчик',
  h('div', { class: 'sc-grid' },
    h('lab-param', { label: 'Мутация', hint: 'Какая доля весов меняется у каждого потомка', min: 0.01, max: 1, step: 0.01, value: 0.4, default: 0.2, live: true }),
    h('lab-param', { label: 'Скорость обучения', hint: 'Логарифмическая шкала', min: 1e-5, max: 1e-2, value: 1e-3, default: 3e-4, scale: 'log', digits: 'auto' }),
    h('lab-param', { label: 'Время', compact: true, min: 0, max: 60, step: 1, value: 36.9, default: 36.9, unit: 'с' }),
    h('lab-param', { label: 'С ошибкой', min: 0, max: 10, value: 12, default: 5, error: 'Значение вне диапазона' }),
    h('lab-param', { label: 'Недоступно', min: 0, max: 10, value: 5, default: 5, disabled: true }),
  ),
  h('div', { class: 'range-scale' }, h('span', null, '1e−5'), h('span', null, '1e−2')),
  row(
    h('span', { class: 'value-field' }, h('input', { value: '0,20', 'aria-label': 'Значение' }), h('span', { class: 'unit' }, 'с')),
    h('lab-stepper', { label: 'Нейроны', min: 1, max: 12, value: 5 }),
    h('lab-stepper', { label: 'На минимуме', min: 1, max: 12, value: 1 }),
    h('lab-stepper', { label: 'На максимуме', min: 1, max: 12, value: 12 }),
    h('input', { class: 'range', type: 'range', 'aria-label': 'Ползунок' }),
  ),
);

section('Поля, список, вкладки',
  h('div', { class: 'sc-grid' },
    h('div', { class: 'field' },
      h('label', { class: 'field-label', for: 'f1' }, 'Название'),
      h('input', { class: 'input', id: 'f1', value: 'Овал' }),
      h('span', { class: 'field-hint' }, 'Подсказка под полем')),
    h('div', { class: 'field' },
      h('label', { class: 'field-label', for: 'f2' }, 'Трасса'),
      h('select', { class: 'select', id: 'f2' }, h('option', null, 'Овал'), h('option', null, 'Волны'), h('option', null, 'Гран-при'))),
  ),
  h('div', { class: 'tabs', role: 'tablist' },
    h('button', { role: 'tab', 'aria-selected': 'true' }, 'Кривые'),
    h('button', { role: 'tab', 'aria-selected': 'false' }, 'Сеть'),
    h('button', { role: 'tab', 'aria-selected': 'false' }, 'Журнал')),
);

section('Метки, точки, подсказки',
  row(...['tag-live', 'tag-ok', 'tag-warn', 'tag-err', 'tag-best', 'tag-info'].map((c) => h('span', { class: `tag ${c}` }, c.slice(4)))),
  row(
    h('span', null, h('i', { class: 'dot dot-running' }), ' идёт'),
    h('span', null, h('i', { class: 'dot dot-paused' }), ' пауза'),
    h('span', null, h('i', { class: 'dot dot-finished' }), ' готово'),
    h('span', null, h('i', { class: 'dot dot-error' }), ' ошибка'),
    h('kbd', { class: 'kbd' }, 'Пробел'),
    h('div', { class: 'tooltip sc-static', role: 'tooltip' }, 'Подсказка ', h('kbd', { class: 'kbd' }, 'Esc')),
  ),
);

const runTab = (colour, name, best, state, selected) =>
  h('button', { class: 'run-tab', role: 'tab', 'aria-selected': String(selected), style: `--c: var(${colour})` },
    h('span', { class: 'run-tab-name' }, name), h('span', { class: 'run-tab-best' }, best),
    h('span', { class: 'run-tab-meta' }, h('i', { class: `dot dot-${state}` }), state === 'running' ? 'идёт' : 'пауза'),
    h('span', { class: 'run-tab-cap' }, 'лучший'));

section('Вкладка запуска (.run-tab), карточки, группа',
  row(runTab('--run-1', 'Запуск 1', '0,98', 'running', true), runTab('--run-2', 'Запуск 2', '0,41', 'paused', false)),
  h('div', { class: 'sc-grid' },
    h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('span', { class: 'card-title' }, 'Карточка')), h('div', { class: 'card-body' }, 'Обычная карточка раздела')),
    h('button', { class: 'tpl-card', role: 'radio', 'aria-checked': 'true' }, h('b', null, 'Классика'), h('span', null, 'Пять входов, два слоя'), h('span', { class: 'mono' }, '6 · 5 · 2')),
    h('button', { class: 'tpl-card', role: 'radio', 'aria-checked': 'false' }, h('b', null, 'Крупная'), h('span', null, 'Больше нейронов'), h('span', { class: 'mono' }, '6 · 12 · 8 · 2')),
    h('button', { class: 'learner-card', role: 'radio', 'aria-checked': 'true' }, icon('brain'), h('b', null, 'PPO ', h('span', { class: 'tag tag-info' }, 'RL')), h('span', null, 'Policy gradient')),
    h('button', { class: 'learner-card', role: 'radio', 'aria-checked': 'false', 'aria-disabled': 'true' }, icon('dna'), h('b', null, 'Эволюция'), h('span', null, 'Недоступно')),
    h('button', { class: 'track-card', role: 'radio', 'aria-checked': 'false' }, h('div', { class: 'thumb' }), h('b', null, 'Овал'), h('span', null, '1 круг')),
  ),
  h('details', { class: 'group', open: true },
    h('summary', null, icon('chevron-right', { size: 16 }), 'Группа параметров'),
    h('div', { class: 'group-body' },
      h('lab-param', { label: 'Первый', min: 0, max: 1, step: 0.1, value: 0.5, default: 0.5 }),
      h('lab-param', { label: 'Второй', min: 0, max: 1, step: 0.1, value: 0.7, default: 0.5 }))),
);

section('Выноски, пустое состояние, скелет',
  h('div', { class: 'callout' }, icon('info'), h('span', null, 'Нейтральная выноска')),
  h('div', { class: 'callout callout-warn' }, icon('triangle-alert'), h('span', null, 'Предупреждение')),
  h('div', { class: 'empty' }, h('b', null, 'Пока пусто'), h('span', null, 'Создай первый запуск')),
  h('div', { class: 'skeleton', style: 'height: 48px' }),
);

section('Таблица, звёзды, прогресс',
  h('table', { class: 'table' },
    h('thead', null, h('tr', null, h('th', { 'aria-sort': 'none' }, 'Запуск'), h('th', { 'aria-sort': 'descending' }, 'Прогресс'))),
    h('tbody', null,
      h('tr', null, h('td', null, 'Запуск 1'), h('td', null, h('span', { class: 'cell-best' }, '1,00'))),
      h('tr', { 'aria-selected': 'true' }, h('td', null, 'Запуск 2'), h('td', null, '0,41')))),
  row(
    h('span', { class: 'stars' }, icon('star'), icon('star')),
    h('span', { class: 'stars stars-lg' }, icon('star'), icon('star')),
    h('div', { class: 'progress', style: 'width: 200px' }, h('i', { style: 'width: 60%' })),
  ),
);
document.querySelectorAll('.stars .icon:first-child').forEach((el) => el.classList.add('on'));

section('Над миром: стекло (.glass)',
  h('div', { class: 'sc-stage' },
    h('div', { class: 'glass', style: 'padding: 8px 12px' }, 'Панель HUD'),
    h('lab-seg', { label: 'Над миром', glass: true, value: 'a', options: options(['a', 'Мир'], ['b', 'Сеть']) }),
    h('button', { class: 'glass-btn', 'aria-label': 'Пауза' }, icon('pause')),
  ),
);

section('Доска сети (.board), меню, всплывающее',
  h('div', { class: 'board sc-board' },
    h('div', { class: 'popover popover-board sc-static' },
      h('button', { class: 'menu-item' }, icon('plus'), 'Добавить нейрон', h('kbd', { class: 'kbd' }, 'N')),
      h('button', { class: 'menu-item danger' }, icon('trash-2'), 'Удалить слой')),
  ),
  h('div', { class: 'popover sc-static' }, h('button', { class: 'menu-item' }, icon('copy'), 'Копировать')),
);

section('Диалог, легенда, тосты',
  h('div', { class: 'dialog sc-static', role: 'dialog', 'aria-label': 'Пример', style: 'width: 360px; padding: 16px' }, 'Диалог'),
  h('div', { class: 'legend', style: '--c: var(--run-1)' }, h('button', { 'aria-pressed': 'true' }, h('i', { class: 'swatch' }), 'Запуск 1')),
  row(
    h('button', { class: 'btn', onclick: () => toast({ text: 'Сохранено', kind: 'success' }) }, 'Тост: успех'),
    h('button', { class: 'btn', onclick: () => toast({ text: 'Что-то пошло не так', kind: 'error' }) }, 'Тост: ошибка'),
    h('button', { class: 'btn', onclick: () => toast({ text: 'Мутация применена', action: { label: 'Отменить', onClick: () => {} } }) }, 'Тост с действием'),
  ),
);

section('Варианты в чистой разметке',
  row(
    h('div', { class: 'seg seg-sm' }, h('button', { 'aria-pressed': 'true' }, 'seg-sm'), h('button', null, 'Второй')),
    h('div', { class: 'seg seg-mono' }, h('button', { 'aria-pressed': 'true' }, 'seg-mono'), h('button', null, '0,5')),
    h('div', { class: 'sc-stage' }, h('div', { class: 'seg seg-glass' }, h('button', { 'aria-pressed': 'true' }, 'seg-glass'), h('button', null, 'Сеть'))),
  ),
  h('div', { class: 'param param-compact is-changed' },
    h('label', { class: 'param-label' }, 'param-compact is-changed'),
    h('span', { class: 'param-value' }, h('span', { class: 'value-field' }, h('input', { value: '0,40', 'aria-label': 'Значение' })))),
  h('div', { class: 'toast toast-error sc-static', role: 'alert' }, icon('triangle-alert'), h('span', { class: 'toast-text' }, 'toast-error: не исчезает сам')),
);
