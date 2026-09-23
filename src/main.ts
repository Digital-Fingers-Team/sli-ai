import './styles.css';
import { h, icon } from './ui/dom';
import { lang, setLang, t } from './i18n';
import { mountSign } from './pages/sign';
import { mountText } from './pages/text';
import { mountTalk } from './pages/talk';
import { mountDict } from './pages/dict';
import { mountAbout } from './pages/about';

type Route = 'sign' | 'text' | 'talk' | 'dict' | 'about';
const ROUTES: Record<Route, (root: HTMLElement, q?: string) => () => void> = {
  sign: mountSign,
  text: mountText,
  talk: mountTalk,
  dict: mountDict,
  about: mountAbout,
};
const NAV: [Route, string, () => string][] = [
  ['sign', 'hand', () => t().navSign],
  ['text', 'type', () => t().navText],
  ['talk', 'talk', () => t().navTalk],
  ['dict', 'book', () => t().navDict],
  ['about', 'info', () => t().navAbout],
];

let cleanup: (() => void) | null = null;

function parse(): { route: Route; q: string } {
  const [path, query = ''] = location.hash.replace(/^#\/?/, '').split('?');
  const route = (path in ROUTES ? path : 'sign') as Route;
  return { route, q: new URLSearchParams(query).get('q') ?? '' };
}

function theme(): 'light' | 'dark' | null {
  try {
    return localStorage.getItem('sli-theme') as 'light' | 'dark' | null;
  } catch {
    return null;
  }
}

function applyTheme(v: 'light' | 'dark' | null) {
  if (v) document.documentElement.dataset.theme = v;
  else delete document.documentElement.dataset.theme;
}

function render() {
  cleanup?.();
  const { route, q } = parse();
  const app = document.getElementById('app')!;
  const main = h('main', { id: 'main', tabindex: -1 });
  const dark =
    document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);

  app.replaceChildren(
    h('a', { class: 'skip', href: '#main' }, lang() === 'ar' ? 'انتقل إلى المحتوى' : 'Skip to content'),
    h(
      'header',
      { class: 'topbar' },
      h('a', { class: 'brand', href: '#/sign' }, h('img', { src: 'icons/icon.svg', alt: '', width: 32, height: 32 }), h('span', {}, t().appName), h('small', {}, t().tagline)),
      h(
        'nav',
        { class: 'nav', 'aria-label': 'Main' },
        ...NAV.map(([r, ic, label]) =>
          h('a', { href: `#/${r}`, 'aria-current': r === route ? 'page' : undefined }, icon(ic), h('span', {}, label())),
        ),
      ),
      h(
        'div',
        { class: 'prefs' },
        h('button', { class: 'btn ghost small', onclick: () => (setLang(lang() === 'ar' ? 'en' : 'ar'), render()) }, t().langToggle),
        h(
          'button',
          {
            class: 'btn ghost small icon-only',
            'aria-label': t().themeToggle,
            title: t().themeToggle,
            onclick: () => {
              const next = dark ? 'light' : 'dark';
              try {
                localStorage.setItem('sli-theme', next);
              } catch {
                /* ignore */
              }
              applyTheme(next);
              render();
            },
          },
          icon(dark ? 'sun' : 'moon'),
        ),
      ),
    ),
    main,
  );
  cleanup = ROUTES[route](main, q);
  document.title = `${t().appName} — ${NAV.find(([r]) => r === route)![2]()}`;
}

setLang(lang());
applyTheme(theme());
addEventListener('hashchange', render);
render();

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
