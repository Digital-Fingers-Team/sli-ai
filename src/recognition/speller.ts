// Turns per-frame letter probabilities into typed letters. A letter is typed once the same
// shape has been held for holdMs; typing it again needs the hand to leave that shape first
// (or drop out of view). Lowering the hands for gapMs ends the word.

import type { Commit, DecoderEvents } from './decoder';
import type { Guess } from './topk';

export interface SpellerOptions {
  holdMs: number;
  minP: number; // smoothed probability needed to hold a letter
  releaseP: number; // below this the last letter counts as released
  smoothing: number; // weight of the newest frame in the running average
  gapMs: number;
}

export const DEFAULT_SPELLER: SpellerOptions = {
  holdMs: 350,
  minP: 0.6,
  releaseP: 0.3,
  smoothing: 0.5,
  gapMs: 700,
};

export class Speller {
  private avg: Float32Array | null = null;
  private candidate = -1;
  private since = 0;
  private last = -1; // index of the last typed letter
  private released = true;
  private lastHandAt = -Infinity;
  private inWord = false;

  constructor(
    private classes: number[],
    private events: DecoderEvents = {},
    private opts: SpellerOptions = DEFAULT_SPELLER,
  ) {}

  reset() {
    this.avg = null;
    this.candidate = -1;
    this.last = -1;
    this.released = true;
    this.inWord = false;
  }

  push(t: number, probs: Float32Array | null) {
    if (!probs) {
      this.avg = null;
      this.candidate = -1;
      // A one- or two-frame tracking dropout is not a release; otherwise held letters double.
      if (t - this.lastHandAt > 250) this.released = true;
      if (this.inWord && t - this.lastHandAt > this.opts.gapMs) {
        this.inWord = false;
        this.last = -1;
        this.events.onSegmentEnd?.(t);
      }
      return;
    }
    this.lastHandAt = t;
    const a = this.opts.smoothing;
    if (!this.avg) this.avg = Float32Array.from(probs);
    else for (let i = 0; i < probs.length; i++) this.avg[i] = a * probs[i] + (1 - a) * this.avg[i];
    const avg = this.avg;

    const order = Array.from(avg.keys()).sort((x, y) => avg[y] - avg[x]);
    const top = order[0];
    const guesses: Guess[] = order.slice(0, 5).map((i) => ({ id: this.classes[i], p: avg[i] }));
    this.events.onLive?.(guesses, t);

    if (this.last >= 0 && avg[this.last] < this.opts.releaseP) this.released = true;
    if (avg[top] < this.opts.minP) {
      this.candidate = -1;
      return;
    }
    if (top !== this.candidate) {
      this.candidate = top;
      this.since = t;
    }
    if (t - this.since >= this.opts.holdMs && (top !== this.last || this.released)) {
      this.last = top;
      this.released = false;
      this.inWord = true;
      const [first, ...alternatives] = guesses;
      const commit: Commit = { id: first.id, p: first.p, alternatives, at: t };
      this.events.onCommit?.(commit);
    }
  }
}
