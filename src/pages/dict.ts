import { CATEGORIES, SIGNS, clipUrl, posterUrl, type Category, type Sign } from '../data/signs';
import { normalize } from '../translate/normalize';
import { h, icon } from '../ui/dom';
import { t } from '../i18n';

export function mountDict(root: HTMLElement) {
  let cat: Category | 'all' = 'all';
  const search = h('input', { type: 'search', class: 'search', placeholder: t().dictSearch, dir: 'auto' });
  const grid = h('ul', { class: 'dict-grid' });
  const count = h('p', { class: 'hint', role: 'status' });
  const dialog = h('dialog', { class: 'sign-dialog' });

  const open = (s: Sign) => {
    const video = h('video', { src: clipUrl(s.id), poster: posterUrl(s.id), autoplay: true, loop: true, muted: true, playsinline: true, class: 'clip front' });
    video.playbackRate = 0.75;
    dialog.replaceChildren(
      h('div', { class: 'stage signer' }, video),
      h('h2', {}, s.ar),
      h('p', { class: 'hint' }, s.label !== s.ar ? s.label : ''),
      h('p', { class: 'hint', dir: 'ltr' }, s.en),
      h('form', { method: 'dialog' }, h('button', { class: 'btn' }, t().close)),
    );
    dialog.showModal();
  };
  dialog.addEventListener('close', () => dialog.replaceChildren());

  const tabs = h(
    'div',
    { class: 'tabs', role: 'tablist' },
    ...(['all', ...CATEGORIES] as const).map((c) =>
      h('button', { role: 'tab', 'data-cat': c, onclick: () => ((cat = c), render()) }, c === 'all' ? t().allCats : t().cats[c]),
    ),
  );

  function render() {
    tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.cat === cat)));
    const q = normalize(search.value);
    const rank = (s: Sign) => CATEGORIES.indexOf(s.cat);
    const list = SIGNS.filter(
      (s) =>
        (cat === 'all' || s.cat === cat) &&
        (!q || [s.ar, s.label, s.en, ...s.aliases].some((f) => normalize(f).includes(q))),
    ).sort((a, b) => rank(a) - rank(b) || a.id - b.id);
    count.textContent = list.length ? `${list.length} ${t().signs}` : t().noResults;
    grid.replaceChildren(
      ...list.map((s) =>
        h(
          'li',
          {},
          h(
            'button',
            { class: 'dict-item', onclick: () => open(s) },
            h('img', { src: posterUrl(s.id), alt: '', loading: 'lazy', width: 160, height: 160 }),
            h('span', {}, s.ar),
            icon('play'),
          ),
        ),
      ),
    );
  }
  search.addEventListener('input', render);
  render();

  root.replaceChildren(
    h('div', { class: 'page page-dict' }, h('header', { class: 'page-head' }, h('h1', {}, t().dictTitle)), search, tabs, count, grid, dialog),
  );
  return () => dialog.close();
}
