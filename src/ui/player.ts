// Plays a translation plan as a continuous run of signer clips, with the word being signed
// highlighted underneath. Two <video> elements alternate so the next clip is already loaded.

import { clipUrl, posterUrl, signById } from '../data/signs';
import { sequence, type PlanItem } from '../translate/planner';
import { t } from '../i18n';
import { h, icon } from './dom';

const SPEEDS = [0.5, 0.75, 1, 1.25];

// Clips are fetched whole (not streamed) so the service worker can cache them for offline use
// and the next clip is fully in memory before it is needed.
const blobs = new Map<number, Promise<string>>();
function clipSrc(id: number): Promise<string> {
  let p = blobs.get(id);
  if (!p) {
    p = fetch(clipUrl(id))
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((b) => URL.createObjectURL(b));
    p.catch(() => blobs.delete(id));
    blobs.set(id, p);
  }
  return p;
}

export class SignPlayer {
  readonly el: HTMLElement;
  private videos: [HTMLVideoElement, HTMLVideoElement];
  private front = 0;
  private items: PlanItem[] = [];
  private steps: { id: number; item: number }[] = [];
  private pos = -1;
  private playing = false;
  private speed = 0.75;
  private caption: HTMLElement;
  private sub: HTMLElement;
  private chips: HTMLElement;
  private playBtn: HTMLButtonElement;
  private speedBtn: HTMLButtonElement;

  constructor() {
    const mk = () => {
      const v = h('video', { muted: true, playsinline: true, preload: 'auto', class: 'clip' });
      v.muted = true;
      v.addEventListener('ended', () => this.next());
      v.addEventListener('error', () => this.playing && this.next());
      return v;
    };
    this.videos = [mk(), mk()];
    this.caption = h('div', { class: 'player-word', 'aria-live': 'polite' });
    this.sub = h('div', { class: 'player-sub' });
    this.chips = h('div', { class: 'plan' });
    this.playBtn = h('button', { class: 'btn', onclick: () => this.toggle() });
    this.speedBtn = h('button', { class: 'btn ghost', onclick: () => this.cycleSpeed() });
    this.el = h(
      'section',
      { class: 'player' },
      h('div', { class: 'stage signer' }, ...this.videos, h('div', { class: 'stage-caption' }, this.caption, this.sub)),
      h(
        'div',
        { class: 'player-controls' },
        this.playBtn,
        h('button', { class: 'btn ghost', onclick: () => this.play(0) }, icon('replay'), t().replay),
        this.speedBtn,
      ),
      this.chips,
    );
    this.render();
  }

  load(items: PlanItem[], autoplay = true) {
    this.items = items;
    this.steps = sequence(items);
    this.pos = -1;
    this.renderChips();
    if (this.steps.length) {
      this.videos[this.front].poster = posterUrl(this.steps[0].id);
      if (autoplay) this.play(0);
    } else {
      this.stop();
      this.caption.textContent = '';
      this.sub.textContent = '';
    }
    this.render();
  }

  play(from = this.pos < 0 || this.pos >= this.steps.length ? 0 : this.pos) {
    if (!this.steps.length) return;
    this.playing = true;
    this.show(from);
  }

  stop() {
    this.playing = false;
    this.videos.forEach((v) => v.pause());
    this.render();
  }

  private toggle() {
    if (this.playing) this.stop();
    else this.play();
  }

  private cycleSpeed() {
    this.speed = SPEEDS[(SPEEDS.indexOf(this.speed) + 1) % SPEEDS.length];
    this.videos.forEach((v) => (v.playbackRate = this.speed));
    this.render();
  }

  private async show(i: number) {
    this.pos = i;
    this.renderStep();
    const step = this.steps[i];
    let src: string;
    try {
      src = await clipSrc(step.id);
    } catch {
      this.next();
      return;
    }
    if (this.pos !== i || !this.playing) return;
    const cur = this.videos[this.front];
    const other = this.videos[1 - this.front];
    if (other.dataset.src === src && other.readyState >= 2) {
      // The preloaded clip is ready: swap instantly.
      this.front = 1 - this.front;
      cur.classList.remove('front');
      cur.pause();
    } else if (cur.dataset.src !== src) {
      cur.dataset.src = src;
      cur.src = src;
    }
    const v = this.videos[this.front];
    v.classList.add('front');
    v.playbackRate = this.speed;
    v.currentTime = 0;
    v.play().catch(() => this.next());
    this.preload(i + 1);
  }

  private async preload(i: number) {
    const step = this.steps[i];
    if (!step) return;
    const src = await clipSrc(step.id).catch(() => '');
    const back = this.videos[1 - this.front];
    if (src && back.dataset.src !== src && !back.classList.contains('front')) {
      back.dataset.src = src;
      back.src = src;
      back.load();
    }
  }

  private next() {
    if (!this.playing) return;
    if (this.pos + 1 < this.steps.length) this.show(this.pos + 1);
    else {
      this.playing = false;
      this.render();
    }
  }

  private renderStep() {
    const step = this.steps[this.pos];
    const item = this.items[step.item];
    const sign = signById(step.id);
    this.caption.textContent = sign.ar;
    this.sub.textContent = item.kind === 'spell' ? `${t().spelled}: ${item.text}` : item.kind === 'number' ? item.text : '';
    this.chips.querySelectorAll('.chip').forEach((c, i) => c.classList.toggle('current', i === step.item));
    this.render();
  }

  private renderChips() {
    this.chips.replaceChildren(
      ...this.items.map((it, i) =>
        h(
          'button',
          {
            class: `chip ${it.kind}`,
            title: it.kind === 'spell' && it.missing.length ? `${t().noSign}: ${it.missing.join(' ')}` : undefined,
            onclick: () => this.play(this.steps.findIndex((s) => s.item === i)),
          },
          it.kind === 'word' ? signById(it.id).ar : it.text,
          it.kind === 'spell' ? h('small', {}, t().spelled) : null,
        ),
      ),
    );
  }

  private render() {
    this.playBtn.replaceChildren(icon(this.playing ? 'pause' : 'play'), this.playing ? t().pause : t().play);
    this.playBtn.disabled = !this.steps.length;
    this.speedBtn.textContent = `${t().speed} ×${this.speed}`;
  }
}
