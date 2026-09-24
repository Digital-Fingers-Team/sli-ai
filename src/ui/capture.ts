// Camera stage + skeleton overlay + live guess, feeding a Sentence. Modes: auto (words and
// spelled letters mixed), words only, letters only (see recognition/interpreter.ts).

import { Engine, type EngineStatus, type SignMode } from '../recognition/engine';
import type { Commit } from '../recognition/decoder';
import type { Guess } from '../recognition/topk';
import { signById } from '../data/signs';
import { t } from '../i18n';
import { drawOverlay } from './overlay';
import { FramingCoach, type FramingIssue } from '../recognition/framing';
import { h, icon } from './dom';
import type { Sentence } from './sentence';

const MODES: SignMode[] = ['auto', 'words', 'letters'];

function storedMode(): SignMode {
  try {
    const m = localStorage.getItem('sli-mode') as SignMode | null;
    return m && MODES.includes(m) ? m : 'auto';
  } catch {
    return 'auto';
  }
}

export class Capture {
  readonly el: HTMLElement;
  private engine: Engine;
  private video = h('video', { class: 'camera', muted: true, playsinline: true });
  private canvas = h('canvas', { class: 'overlay' });
  private stage: HTMLElement;
  private status = h('div', { class: 'cam-status', role: 'status' });
  private live = h('div', { class: 'live-guess', 'aria-hidden': 'true' });
  private toggleBtn: HTMLButtonElement;
  private running = false;
  private handSeen = false;
  private fpsEl = h('span', { class: 'fps' });
  private mode: SignMode = storedMode();
  private modeButtons: HTMLButtonElement[];
  private hint = h('p', { class: 'mode-hint' });
  private tip = h('div', { class: 'framing-tip', role: 'status' });
  private coach = new FramingCoach();
  private issue: FramingIssue | null = null;
  private lastCommitted = -1; // not previewed again while it is still being held

  constructor(
    private sentence: Sentence,
    private onCommit: (c: Commit) => void = () => {},
  ) {
    this.toggleBtn = h('button', { class: 'btn primary', onclick: () => this.toggle() });
    this.modeButtons = MODES.map((m) =>
      h('button', { type: 'button', 'data-mode': m, onclick: () => this.setMode(m) }, t().modes[m]),
    );
    this.stage = h(
      'div',
      { class: 'stage camera-stage idle' },
      this.video,
      this.canvas,
      this.live,
      this.tip,
      h('div', { class: 'stage-bar' }, this.status, this.fpsEl),
    );
    this.el = h(
      'div',
      { class: 'capture' },
      h('div', { class: 'mode-switch', role: 'group', 'aria-label': t().modeLabel }, ...this.modeButtons),
      this.stage,
      this.hint,
      h('div', { class: 'capture-actions' }, this.toggleBtn),
    );
    this.engine = new Engine(this.video, {
      onStatus: (s, d) => this.setStatus(s, d),
      onFrame: (raw, fps) => {
        const seen = raw.hands.length > 0;
        if (seen !== this.handSeen) {
          this.handSeen = seen;
          this.stage.classList.toggle('signing', seen);
          this.status.textContent = seen ? t().handSeen : t().noHand;
        }
        drawOverlay(this.canvas, raw, seen);
        this.showFraming(this.coach.push(performance.now(), raw));
        this.fpsEl.textContent = `${Math.round(fps)} ${t().fps}`;
      },
      onLive: (g) => this.showLive(g),
      onCommit: (c) => {
        this.lastCommitted = c.id;
        this.flash(c);
        this.sentence.add(c);
        this.onCommit(c);
      },
      onSegmentEnd: () => {
        this.lastCommitted = -1;
        this.live.textContent = '';
        this.sentence.setPending(null);
        this.sentence.endWord();
      },
    });
    this.setMode(this.mode);
    this.renderButton();
    this.status.textContent = t().cameraOff;
  }

  /** Tells the signer when a hand is cut off or they are too close for two-handed signs. */
  private showFraming(issue: FramingIssue | null) {
    if (issue === this.issue) return;
    this.issue = issue;
    this.stage.classList.toggle('framing-bad', !!issue);
    if (!issue) {
      this.tip.textContent = '';
      return;
    }
    const upright = this.video.videoHeight > this.video.videoWidth;
    this.tip.textContent = t().framing[issue] + (upright ? ` ${t().framingRotate}` : '');
  }

  private setMode(mode: SignMode) {
    this.mode = mode;
    try {
      localStorage.setItem('sli-mode', mode);
    } catch {
      /* ignore */
    }
    this.modeButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    this.hint.textContent = t().modeHints[mode];
    this.engine.setMode(mode);
    this.sentence.setPending(null);
    this.sentence.endWord();
    this.lastCommitted = -1;
    this.live.textContent = '';
  }

  private showLive(guesses: Guess[]) {
    const top = guesses[0];
    // Words show once the model leans one way; letters are only offered when fairly sure.
    const letter = signById(top.id).cat === 'letters';
    const shown = top.id !== this.lastCommitted && top.p >= (letter ? 0.45 : 0.3);
    if (!shown && this.live.classList.contains('committed')) return;
    this.sentence.setPending(shown ? top.id : null);
    this.live.textContent = shown ? signById(top.id).ar : '';
    this.live.classList.remove('committed');
  }

  private flash(c: Commit) {
    this.live.textContent = signById(c.id).ar;
    this.live.classList.remove('committed');
    void this.live.offsetWidth;
    this.live.classList.add('committed');
  }

  private setStatus(s: EngineStatus, detail?: string) {
    this.stage.classList.toggle('idle', s !== 'running');
    if (s === 'loading') {
      this.status.textContent = detail ? `${t().loadingModels} ${t().loadingStep[detail] ?? ''}` : t().loadingModels;
      this.toggleBtn.disabled = true;
      return;
    }
    this.toggleBtn.disabled = false;
    this.running = s === 'running';
    if (s === 'running') {
      const { videoWidth: w, videoHeight: hh } = this.video;
      if (w && hh) this.stage.style.aspectRatio = `${w} / ${hh}`;
      this.status.textContent = t().noHand;
    } else if (s === 'error') {
      this.status.textContent =
        detail === 'NotAllowedError' ? t().cameraDenied : detail === 'NotFoundError' ? t().cameraMissing : `${t().cameraError} (${detail})`;
      this.stage.classList.add('error');
    } else {
      this.status.textContent = t().cameraOff;
      this.canvas.getContext('2d')?.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.coach.reset();
      this.showFraming(null);
      this.live.textContent = '';
      this.fpsEl.textContent = '';
    }
    this.renderButton();
  }

  private renderButton() {
    this.toggleBtn.replaceChildren(icon(this.running ? 'stop' : 'camera'), this.running ? t().stopCamera : t().startCamera);
  }

  toggle() {
    if (this.running) this.engine.stop();
    else {
      this.stage.classList.remove('error');
      void this.engine.start();
    }
  }

  destroy() {
    this.engine.stop();
  }
}
