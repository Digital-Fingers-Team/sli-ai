// The sentence being built from recognised signs. Consecutive letters join into one
// fingerspelled word; every word can be swapped for one of the model's alternatives.

import { signById } from '../data/signs';
import type { Commit } from '../recognition/decoder';
import type { Guess } from '../recognition/topk';
import { t } from '../i18n';
import { h } from './dom';

interface Token {
  id: number;
  alternatives: Guess[];
}

const isLetter = (id: number) => signById(id).cat === 'letters';

/** Text of a token list: letters run together, everything else is space-separated. */
export function sentenceText(ids: number[]): string {
  let out = '';
  let prevLetter = false;
  for (const id of ids) {
    const s = signById(id);
    const letter = s.cat === 'letters';
    const text = letter ? s.ar.replace('ـ', '') : s.ar;
    out += (out && !(letter && prevLetter) ? ' ' : '') + text;
    prevLetter = letter;
  }
  return out;
}

export class Sentence {
  readonly el: HTMLElement;
  private tokens: Token[] = [];
  private list = h('div', { class: 'sentence-words', 'aria-live': 'polite' });
  private menu: HTMLElement | null = null;

  constructor(private onChange: (text: string) => void = () => {}) {
    this.el = h('div', { class: 'sentence' }, this.list);
    this.render();
  }

  add(c: Commit) {
    this.tokens.push({ id: c.id, alternatives: c.alternatives });
    this.render();
  }

  undo() {
    this.tokens.pop();
    this.render();
  }

  clear() {
    this.tokens = [];
    this.render();
  }

  text() {
    return sentenceText(this.tokens.map((tk) => tk.id));
  }

  get length() {
    return this.tokens.length;
  }

  private openMenu(i: number, anchor: HTMLElement) {
    this.menu?.remove();
    const tk = this.tokens[i];
    this.menu = h(
      'div',
      { class: 'alt-menu', role: 'menu' },
      h('div', { class: 'alt-title' }, t().alternatives),
      ...tk.alternatives.map((g) =>
        h(
          'button',
          {
            role: 'menuitem',
            onclick: () => {
              tk.alternatives = [{ id: tk.id, p: 0 }, ...tk.alternatives.filter((a) => a.id !== g.id)];
              tk.id = g.id;
              this.render();
            },
          },
          signById(g.id).ar,
          h('small', {}, `${Math.round(g.p * 100)}%`),
        ),
      ),
      h(
        'button',
        {
          role: 'menuitem',
          class: 'danger',
          onclick: () => {
            this.tokens.splice(i, 1);
            this.render();
          },
        },
        t().remove,
      ),
    );
    anchor.after(this.menu);
    const close = (e: Event) => {
      if (this.menu && !this.menu.contains(e.target as Node) && e.target !== anchor) {
        this.menu.remove();
        this.menu = null;
        document.removeEventListener('pointerdown', close);
      }
    };
    document.addEventListener('pointerdown', close);
  }

  private render() {
    this.menu?.remove();
    this.menu = null;
    if (!this.tokens.length) {
      this.list.replaceChildren(h('p', { class: 'empty' }, t().emptySentence));
    } else {
      this.list.replaceChildren(
        ...this.tokens.map((tk, i) => {
          const letter = isLetter(tk.id);
          const joined = letter && i > 0 && isLetter(this.tokens[i - 1].id);
          const b = h(
            'button',
            {
              class: `word${letter ? ' letter' : ''}${joined ? ' joined' : ''}`,
              'aria-haspopup': 'menu',
              'data-alts': tk.alternatives.map((a) => signById(a.id).ar).join('|'),
            },
            signById(tk.id).ar,
          );
          b.addEventListener('click', () => this.openMenu(i, b));
          return b;
        }),
      );
    }
    this.onChange(this.text());
  }
}
