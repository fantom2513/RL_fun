import { h } from '../core/dom.js';

export function confirmRunRemoval(runs) {
  if (!runs.length) return Promise.resolve(false);
  const origin = document.activeElement;
  const dialog = h('dialog', {
    class: 'dialog run-delete-dialog',
    'aria-labelledby': 'run-delete-title',
    'aria-describedby': 'run-delete-description',
  });
  const cancel = h('button', {
    type: 'button', class: 'btn', autofocus: true,
    onclick: () => dialog.close(),
  }, 'Отмена');
  dialog.append(
    h('h2', { id: 'run-delete-title' }, runs.length === 1 ? 'Удалить заезд?' : `Удалить заезды (${runs.length})?`),
    h('p', { id: 'run-delete-description' }, 'Заезды исчезнут из истории, графиков и гаража.'),
    h('ul', { class: 'run-delete-list' }, runs.map((run) => h('li', null, run.name))),
    h('p', { class: 'run-delete-note' }, 'Заработанные звёзды и открытые уровни сохранятся.'),
    h('div', { class: 'run-delete-actions' }, cancel,
      h('button', {
        type: 'button', class: 'btn btn-danger',
        onclick: () => dialog.close('delete'),
      }, runs.length === 1 ? 'Удалить заезд' : `Удалить заезды (${runs.length})`)),
  );
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => {
      const accepted = dialog.returnValue === 'delete';
      dialog.remove();
      if (origin?.isConnected) origin.focus({ preventScroll: true });
      resolve(accepted);
    }, { once: true });
    document.body.append(dialog);
    dialog.showModal();
    cancel.focus();
  });
}
