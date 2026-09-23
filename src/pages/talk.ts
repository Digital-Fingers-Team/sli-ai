// Two people, one screen: the deaf person signs (their words are spoken aloud), the hearing
// person speaks or types (their words are played as signs). Both appear in one transcript.

import { Capture } from '../ui/capture';
import { Sentence } from '../ui/sentence';
import { h, icon } from '../ui/dom';
import { speak } from '../speech/tts';
import { plan } from '../translate/planner';
import { textToSign } from './text';
import { t } from '../i18n';

export function mountTalk(root: HTMLElement) {
  const log = h('ol', { class: 'transcript', 'aria-live': 'polite' });
  const empty = h('p', { class: 'empty' }, t().talkEmpty);
  let hearing: ReturnType<typeof textToSign>;

  const add = (who: 'deaf' | 'hearing', text: string) => {
    empty.remove();
    const replay =
      who === 'hearing'
        ? h('button', { class: 'btn ghost small', onclick: () => hearing.player.load(plan(text)) }, icon('hand'), t().showInSign)
        : h('button', { class: 'btn ghost small', onclick: () => speak(text) }, icon('speaker'));
    log.append(h('li', { class: `msg ${who}` }, h('span', { class: 'who' }, who === 'deaf' ? t().talkDeaf : t().talkHearing), h('p', { dir: 'auto' }, text), replay));
    log.scrollTop = log.scrollHeight;
  };

  const sendBtn = h('button', { class: 'btn primary', disabled: true }, icon('speaker'), t().talkSend);
  const sentence = new Sentence((text) => (sendBtn.disabled = !text));
  sendBtn.addEventListener('click', () => {
    const text = sentence.text();
    if (!text) return;
    add('deaf', text);
    void speak(text);
    sentence.clear();
  });
  const capture = new Capture((c) => sentence.add(c));
  hearing = textToSign('', (text) => add('hearing', text));

  root.replaceChildren(
    h(
      'div',
      { class: 'page page-talk' },
      h('header', { class: 'page-head' }, h('h1', {}, t().talkTitle)),
      h(
        'div',
        { class: 'talk-grid' },
        h(
          'section',
          { class: 'side deaf' },
          h('h2', {}, t().talkDeaf),
          capture.el,
          h('div', { class: 'panel output' }, sentence.el, h('div', { class: 'actions' }, sendBtn,
            h('button', { class: 'btn ghost', onclick: () => sentence.undo() }, icon('undo'), t().undo))),
        ),
        h('section', { class: 'side log' }, log, empty),
        h('section', { class: 'side hearing' }, h('h2', {}, t().talkHearing), hearing.form, hearing.player.el),
      ),
    ),
  );
  return () => {
    capture.destroy();
    hearing.destroy();
    hearing.player.stop();
  };
}
