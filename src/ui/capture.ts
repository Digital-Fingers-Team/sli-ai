// Camera stage + skeleton overlay + live guess. Emits committed signs.

import { Engine, type EngineStatus } from '../recognition/engine';
import type { Commit } from '../recognition/decoder';
import type { Guess } from '../recognition/topk';
import { signById } from '../data/signs';
import { t } from '../i18n';
import { drawOverlay } from './overlay';
import { h, icon } from './dom';

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

  constructor(private onCommit: (c: Commit) => void) {
    this.toggleBtn = h('button', { class: 'btn primary', onclick: () => this.toggle() });
    this.stage = h(
      'div',
      { class: 'stage camera-stage idle' },
      this.video,
      this.canvas,
      this.live,
      h('div', { class: 'stage-bar' }, this.status, this.fpsEl),
    );
    this.el = h('div', { class: 'capture' }, this.stage, h('div', { class: 'capture-actions' }, this.toggleBtn));
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
        this.fpsEl.textContent = `${Math.round(fps)} ${t().fps}`;
      },
      onLive: (g) => this.showLive(g),
      onCommit: (c) => {
        this.flash(c);
        this.onCommit(c);
      },
      onSegmentEnd: () => (this.live.textContent = ''),
    });
    this.renderButton();
    this.status.textContent = t().cameraOff;
  }

  private showLive(guesses: Guess[]) {
    const top = guesses[0];
    this.live.textContent = top.p > 0.25 ? `${t().maybe}: ${signById(top.id).ar}` : '';
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
