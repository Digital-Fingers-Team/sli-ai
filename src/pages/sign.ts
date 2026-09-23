import { Capture } from '../ui/capture';
import { Sentence } from '../ui/sentence';
import { h, icon } from '../ui/dom';
import { speak } from '../speech/tts';
import { signById } from '../data/signs';
import { t } from '../i18n';

export function mountSign(root: HTMLElement) {
  let autoSpeak = (() => {
    try {
      return localStorage.getItem('sli-autospeak') === '1';
    } catch {
      return false;
    }
  })();

  const speakBtn = h('button', { class: 'btn primary', onclick: () => speak(sentence.text()) }, icon('speaker'), t().speakSentence);
  const sentence = new Sentence((text) => {
    speakBtn.disabled = !text;
  });
  const capture = new Capture((c) => {
    sentence.add(c);
    const s = signById(c.id);
    if (autoSpeak && s.cat !== 'letters') void speak(s.ar);
  });

  const auto = h('input', { type: 'checkbox', id: 'autospeak' });
  auto.checked = autoSpeak;
  auto.addEventListener('change', () => {
    autoSpeak = auto.checked;
    try {
      localStorage.setItem('sli-autospeak', autoSpeak ? '1' : '0');
    } catch {
      /* ignore */
    }
  });

  root.replaceChildren(
    h(
      'div',
      { class: 'page page-sign' },
      h('header', { class: 'page-head' }, h('h1', {}, t().signTitle), h('p', {}, t().signHint)),
      h(
        'div',
        { class: 'split' },
        capture.el,
        h(
          'section',
          { class: 'panel output' },
          sentence.el,
          h('p', { class: 'hint' }, t().tapToFix),
          h(
            'div',
            { class: 'actions' },
            speakBtn,
            h('button', { class: 'btn ghost', onclick: () => sentence.undo() }, icon('undo'), t().undo),
            h('button', { class: 'btn ghost', onclick: () => sentence.clear() }, icon('clear'), t().clear),
          ),
          h('label', { class: 'switch', for: 'autospeak' }, auto, h('span', {}, t().autoSpeak)),
        ),
      ),
    ),
  );
  return () => capture.destroy();
}
