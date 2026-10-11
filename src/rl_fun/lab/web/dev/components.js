// Dev showcase of the lab components (/static/dev/components.html). Not linked from the app.
// ?theme=light|dark picks the scheme for screenshots; T toggles it; the header segments too.
import { h, hydrateIcons, icon } from '../core/dom.js';
import '../ui/param.js';
import '../ui/seg.js';
import '../ui/stepper.js';
import { toast } from '../ui/toast.js';

const root = document.documentElement;

function applyTheme(theme) {
  if (theme === 'light' || theme === 'dark') root.dataset.theme = theme;
  else delete root.dataset.theme;
  const seg = document.getElementById('theme');
  if (seg) seg.value = theme;
}

function currentScheme() {
  if (root.dataset.theme) return root.dataset.theme;
  return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

hydrateIcons();
applyTheme(new URLSearchParams(location.search).get('theme') ?? 'system');

document.getElementById('theme').addEventListener('change', (event) => applyTheme(event.detail.value));

document.addEventListener('keydown', (event) => {
  if (event.target instanceof Element && event.target.closest('input, select, textarea, lab-seg, lab-stepper')) return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === 't' || event.key === 'T' || event.key === 'е' || event.key === 'Е') {
    applyTheme(currentScheme() === 'light' ? 'dark' : 'light');
  }
});

document.getElementById('toast-demo').addEventListener('click', () => {
  toast({
    text: 'Мутация 0,20 применится со следующего поколения',
    action: { label: 'Отменить', run: () => toast({ text: 'Изменение отменено', kind: 'info' }) },
  });
});

document.getElementById('params').addEventListener('change', (event) => {
  const param = event.target.closest('lab-param');
  if (!param?.hasAttribute('live')) return;
  param.setAttribute('applying', '');
  setTimeout(() => param.removeAttribute('applying'), 700);
});

document.getElementById('star-demo').addEventListener('click', () => {
  const star = document.querySelector('.stars-lg .icon:not(.on)');
  if (!star) return;
  star.classList.add('on', 'is-new');
  toast({ text: 'Новая звезда: «Быстрее 37 секунд»' });
});

function openDialog(opener) {
  const close = () => {
    scrim.remove();
    opener.focus();
  };
  const cancel = h('button', { class: 'btn btn-ghost', type: 'button', onclick: close }, 'Отмена');
  const scrim = h('div', { class: 'scrim', onclick: (e) => e.target === scrim && close() },
    h('div', { class: 'dialog', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'demo-title', style: { width: 'min(520px, 100%)' } },
      h('div', { class: 'dialog-head' }, h('h3', { id: 'demo-title', class: 'dialog-title' }, 'Сохранить трассу «Петля»?')),
      h('div', { class: 'dialog-body' }, h('p', {}, 'Изменения не сохранены. Esc — остаться в редакторе.')),
      h('div', { class: 'dialog-foot' },
        cancel,
        h('button', { class: 'btn', type: 'button', onclick: close }, 'Не сохранять'),
        h('button', { class: 'btn btn-primary', type: 'button', onclick: () => { close(); toast('Трасса «Петля» сохранена'); } }, icon('save'), 'Сохранить'))));
  scrim.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') close();
    if (event.key !== 'Tab') return;
    const items = [...scrim.querySelectorAll('button')];
    const index = items.indexOf(document.activeElement);
    if (event.shiftKey && index <= 0) { event.preventDefault(); items.at(-1).focus(); }
    if (!event.shiftKey && index === items.length - 1) { event.preventDefault(); items[0].focus(); }
  });
  document.body.append(scrim);
  cancel.focus();
}

document.getElementById('dialog-demo').addEventListener('click', (event) => openDialog(event.currentTarget));
