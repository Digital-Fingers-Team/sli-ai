// "Teach the app your hand": the user signs each letter once, and the letter model is
// fine-tuned on the device to their hand (on a signer it never saw: 80% → 90% of letters,
// tests/unit/letters.test.ts). The tuned weights stay in this browser only.

import { Engine, loadModels } from '../recognition/engine';
import { signingHand } from '../recognition/interpreter';
import { augment, featurise, recordingWindows, type HandFrame } from '../recognition/letters';
import { SIGNS, clipUrl, posterUrl } from '../data/signs';
import { drawOverlay } from '../ui/overlay';
import { lang } from '../i18n';
import { h, icon } from '../ui/dom';

const LETTERS = SIGNS.filter((s) => s.cat === 'letters');
const STEADY_MS = 600; // hand in view this long before recording starts
const RECORD_MS = 1800;
const MIN_HAND_FRAMES = 8;
const MIN_LETTERS = 30; // tuning on too few letters would bias the model towards them

const copy = {
  ar: {
    title: 'علّم التطبيق يدك',
    intro:
      'أشِر بكل حرف مرة واحدة (نحو دقيقتين)، ويتعلم التطبيق شكل يدك أنت. في تجاربنا على شخص لم يره النموذج من قبل، ارتفعت دقة الحروف من 80% إلى 90%. يبقى كل شيء على جهازك.',
    saved: 'التطبيق متعلّم يدك على هذا الجهاز.',
    reset: 'انسَ يدي',
    start: 'ابدأ',
    doLetter: (l: string) => `أشِر بالحرف «${l}» وثبّت يدك`,
    waitHand: 'ارفع يدك أمام الكاميرا…',
    recording: 'ثبّت يدك…',
    again: 'لم نرَ يدك جيدًا، أعد المحاولة.',
    skip: 'تخطَّ',
    redo: 'أعِد الحرف السابق',
    progress: (n: number, of: number) => `${n} من ${of}`,
    training: 'يتعلم التطبيق يدك…',
    tooFew: (n: number) => `سُجّل ${n} حرفًا فقط؛ نحتاج ${MIN_LETTERS} على الأقل. أعِد الحروف التي تخطيتها.`,
    done: 'تم! جرّب التهجئة الآن.',
    tryIt: 'جرّب وضع الحروف',
  },
  en: {
    title: 'Teach the app your hand',
    intro:
      'Sign each letter once (about two minutes) and the app learns your own hand. In our tests on a person the model had never seen, letters went from 80% to 90% right. Everything stays on your device.',
    saved: 'The app has learned your hand on this device.',
    reset: 'Forget my hand',
    start: 'Start',
    doLetter: (l: string) => `Sign the letter “${l}” and hold it`,
    waitHand: 'Raise your hand in front of the camera…',
    recording: 'Hold it…',
    again: 'We did not see your hand well; try again.',
    skip: 'Skip',
    redo: 'Redo the previous letter',
    progress: (n: number, of: number) => `${n} of ${of}`,
    training: 'Learning your hand…',
    tooFew: (n: number) => `Only ${n} letters were recorded; at least ${MIN_LETTERS} are needed. Redo the ones you skipped.`,
    done: 'Done! Try spelling now.',
    tryIt: 'Try Letters mode',
  },
};

export function mountTeach(root: HTMLElement) {
  const c = copy[lang()];
  const video = h('video', { class: 'camera', muted: true, playsinline: true });
  const canvas = h('canvas', { class: 'overlay' });
  const stage = h('div', { class: 'stage camera-stage idle' }, video, canvas);
  const clip = h('video', { class: 'teach-clip', muted: true, playsinline: true, loop: true, autoplay: true });
  const glyph = h('div', { class: 'teach-glyph' });
  const prompt = h('p', { class: 'teach-prompt', role: 'status' });
  const bar = h('div', { class: 'teach-bar' }, h('span'));
  const count = h('p', { class: 'hint' });
  const startBtn = h('button', { class: 'btn primary' }, icon('camera'), c.start);
  const skipBtn = h('button', { class: 'btn ghost' }, c.skip);
  const redoBtn = h('button', { class: 'btn ghost' }, icon('undo'), c.redo);
  const controls = h('div', { class: 'actions' }, skipBtn, redoBtn);
  const status = h('div', { class: 'teach-status' });
  const card = h('section', { class: 'panel teach-card' }, glyph, clip, prompt, bar, count, controls);

  const recordings = new Map<number, (HandFrame | null)[]>();
  let index = 0;
  let phase: 'off' | 'wait' | 'rec' | 'train' | 'done' = 'off';
  let handSince = 0;
  let recStart = 0;
  let frames: (HandFrame | null)[] = [];

  const engine = new Engine(video, {
    onStatus: (s) => {
      stage.classList.toggle('idle', s !== 'running');
      if (s === 'running') {
        const { videoWidth: w, videoHeight: hh } = video;
        if (w && hh) stage.style.aspectRatio = `${w} / ${hh}`;
        showLetter();
      }
    },
    onFrame: (raw) => {
      const hand = signingHand(raw);
      drawOverlay(canvas, raw, !!hand);
      const now = performance.now();
      if (phase === 'wait') {
        if (!hand) handSince = 0;
        else if (!handSince) handSince = now;
        else if (now - handSince >= STEADY_MS) {
          phase = 'rec';
          recStart = now;
          frames = [];
          prompt.textContent = c.recording;
        }
      } else if (phase === 'rec') {
        frames.push(hand ? { label: hand.label, lms: hand.lms } : null);
        setBar((now - recStart) / RECORD_MS);
        if (now - recStart >= RECORD_MS) finishLetter();
      }
    },
  });
  engine.setMode('letters'); // hands only: fastest

  function setBar(f: number) {
    (bar.firstChild as HTMLElement).style.width = `${Math.min(100, Math.round(f * 100))}%`;
  }

  function showLetter() {
    if (index >= LETTERS.length) return void train();
    const s = LETTERS[index];
    phase = 'wait';
    handSince = 0;
    glyph.textContent = s.ar;
    prompt.textContent = `${c.doLetter(s.ar)} — ${c.waitHand}`;
    clip.poster = posterUrl(s.id);
    clip.src = clipUrl(s.id);
    void clip.play().catch(() => {});
    count.textContent = c.progress(index + 1, LETTERS.length);
    redoBtn.disabled = index === 0;
    setBar(0);
  }

  function finishLetter() {
    const got = frames.filter((f) => f).length;
    if (got < MIN_HAND_FRAMES) {
      phase = 'wait';
      handSince = 0;
      prompt.textContent = c.again;
      setBar(0);
      return;
    }
    recordings.set(LETTERS[index].id, frames);
    index++;
    showLetter();
  }

  skipBtn.onclick = () => {
    if (phase !== 'wait' && phase !== 'rec') return;
    recordings.delete(LETTERS[index].id);
    index++;
    showLetter();
  };
  redoBtn.onclick = () => {
    if (index === 0 || (phase !== 'wait' && phase !== 'rec')) return;
    index--;
    showLetter();
  };

  async function train() {
    phase = 'train';
    engine.stop();
    controls.hidden = true;
    if (recordings.size < MIN_LETTERS) {
      prompt.textContent = c.tooFew(recordings.size);
      index = LETTERS.findIndex((s) => !recordings.has(s.id));
      controls.hidden = false;
      startBtn.hidden = false;
      return;
    }
    const { letters } = await loadModels();
    const m = letters.model;
    const classIndex = new Map(m.classes.map((id, i) => [id, i]));
    // Windows at the camera's rate and at half of it, as the model was trained on 10-30 fps.
    const examples = [...recordings].flatMap(([id, fr]) =>
      [1, 2].flatMap((stride) => recordingWindows(fr, m.window, m.mirror, stride, 0, [0, 1]).map((s) => ({ s, y: classIndex.get(id)! }))),
    );
    glyph.textContent = '';
    clip.removeAttribute('src');
    clip.load();
    prompt.textContent = c.training;
    const net = letters.base.clone();
    await net.fineTune(
      examples.length,
      (i, rand) => ({ x: featurise(augment(examples[i].s, rand)), y: examples[i].y }),
      { epochs: 30, lr: 5e-4 },
      setBar,
    );
    letters.setPersonal(net);
    phase = 'done';
    prompt.textContent = c.done;
    count.textContent = '';
    card.append(
      h(
        'a',
        {
          class: 'btn primary',
          href: '#/sign',
          onclick: () => {
            try {
              localStorage.setItem('sli-mode', 'letters');
            } catch {
              /* ignore */
            }
          },
        },
        icon('hand'),
        c.tryIt,
      ),
    );
    renderStatus(true);
  }

  function renderStatus(personal: boolean) {
    status.replaceChildren();
    if (!personal) return;
    status.append(
      h('p', { class: 'callout' }, c.saved),
      h(
        'button',
        {
          class: 'btn ghost',
          onclick: async () => {
            (await loadModels()).letters.resetPersonal();
            renderStatus(false);
          },
        },
        icon('clear'),
        c.reset,
      ),
    );
  }

  startBtn.onclick = () => {
    startBtn.hidden = true;
    controls.hidden = false;
    if (index >= LETTERS.length) index = 0;
    void engine.start();
  };
  controls.hidden = true;
  prompt.textContent = '';

  root.replaceChildren(
    h(
      'div',
      { class: 'page page-teach' },
      h('header', { class: 'page-head' }, h('h1', {}, c.title), h('p', {}, c.intro)),
      status,
      h('div', { class: 'split' }, h('div', { class: 'capture' }, stage, h('div', { class: 'capture-actions' }, startBtn)), card),
    ),
  );
  // Show whether this device already has a tuned model (loading also warms the models up).
  loadModels()
    .then(({ letters }) => renderStatus(letters.personal))
    .catch(() => {});
  return () => {
    engine.stop();
    clip.removeAttribute('src');
  };
}
