import { SignPlayer } from '../ui/player';
import { h, icon } from '../ui/dom';
import { plan } from '../translate/planner';
import { listen, sttSupported } from '../speech/stt';
import { lang, t } from '../i18n';

const EXAMPLES = ['السلام عليكم', 'أنا طبيب في المستشفى', 'شكراً يا صديقي', 'أمي وأبي في البيت', 'عندي صداع وحمى', 'أحمد 25'];

/** Text/voice input wired to a sign player. Reused by the conversation page. */
export function textToSign(initial = '', onRun?: (text: string) => void) {
  const player = new SignPlayer();
  const input = h('textarea', { class: 'text-input', rows: 2, placeholder: t().textPlaceholder, dir: 'auto' });
  input.value = initial;
  const note = h('p', { class: 'hint', role: 'status' });
  let stopListening: (() => void) | null = null;

  const run = () => {
    player.load(plan(input.value));
    if (input.value.trim()) onRun?.(input.value.trim());
  };
  const micBtn = h('button', { class: 'btn ghost', onclick: () => toggleMic() });
  const renderMic = () =>
    micBtn.replaceChildren(icon(stopListening ? 'stop' : 'mic'), stopListening ? t().stopListening : t().listen);

  function toggleMic() {
    if (stopListening) {
      stopListening();
      return;
    }
    if (!sttSupported()) {
      note.textContent = t().sttUnsupported;
      return;
    }
    let before = input.value ? input.value + ' ' : '';
    let finals = '';
    note.textContent = t().listening;
    stopListening = listen(lang(), {
      onInterim: (s) => (input.value = before + finals + s),
      onFinal: (s) => {
        finals += s + ' ';
        input.value = (before + finals).trim();
        run();
        // In a conversation every utterance becomes its own message.
        if (onRun) before = finals = input.value = '';
      },
      onError: (code) => (note.textContent = code === 'unsupported' ? t().sttUnsupported : `${t().sttError} (${code})`),
      onEnd: () => {
        stopListening = null;
        if (note.textContent === t().listening) note.textContent = '';
        renderMic();
      },
    });
    renderMic();
  }
  renderMic();

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      run();
    }
  });

  const form = h(
    'div',
    { class: 'panel composer' },
    input,
    h(
      'div',
      { class: 'actions' },
      h('button', { class: 'btn primary', onclick: run }, icon('hand'), t().translate),
      micBtn,
    ),
    note,
  );
  if (initial) run();
  return { player, form, input, run, destroy: () => stopListening?.() };
}

export function mountText(root: HTMLElement, query = '') {
  const tts = textToSign(query);
  const examples = h(
    'div',
    { class: 'examples' },
    h('span', {}, t().examples),
    ...EXAMPLES.map((ex) =>
      h('button', { class: 'chip', onclick: () => ((tts.input.value = ex), tts.run()) }, ex),
    ),
  );
  root.replaceChildren(
    h(
      'div',
      { class: 'page page-text' },
      h('header', { class: 'page-head' }, h('h1', {}, t().textTitle)),
      h('div', { class: 'split' }, h('div', {}, tts.form, examples), tts.player.el),
    ),
  );
  return () => {
    tts.destroy();
    tts.player.stop();
  };
}
