// Turns a continuous stream of per-frame features into committed signs.
//
// A segment starts when a hand appears and ends when no hand has been seen for gapMs; the
// whole segment is then classified. While signing, a sign is committed early only if the
// same answer stays very confident for about a second (so letters can be spelled without
// lowering the hands). Windows keep frames where the hand was momentarily lost: the KArSL
// training clips contain them too, and dropping them costs accuracy.
//
// Defaults were chosen with tests/replay (3-sign sentences from held-out KArSL videos):
// see training/README.md for the numbers behind each value.

import { tsnSample } from './features';
import type { Guess } from './topk';

export interface DecoderOptions {
  stepMs: number; // how often to classify while signing
  windowsMs: number[]; // window lengths tried at each step; the most confident wins
  minFrames: number; // frames with a hand needed before a window is worth classifying
  gapMs: number; // hand missing this long ends the segment
  commitP: number; // probability needed on consecutive checks
  commitRuns: number; // consecutive checks needed
  finalP: number; // lower bar for the single check at the end of a segment
  maxBufferMs: number;
  tailMinMs: number; // after an early commit, the rest of the segment must last this long to be a sign
  tailP: number; // ...and be this confident
  preRollMs: number; // frames before the hand first appeared that belong to the sign
  postRollMs: number; // frames after the hand was last seen that belong to the sign
}

export const DEFAULT_OPTIONS: DecoderOptions = {
  stepMs: 250,
  windowsMs: [900, 1600, 2600],
  minFrames: 5,
  gapMs: 450,
  commitP: 0.9,
  commitRuns: 4,
  finalP: 0.4,
  maxBufferMs: 5000,
  tailMinMs: 700,
  tailP: 0.8,
  preRollMs: 0,
  postRollMs: 0,
};

export interface Commit {
  id: number;
  p: number;
  alternatives: Guess[];
  at: number;
}

export interface DecoderEvents {
  onLive?(guesses: Guess[], at: number): void;
  onCommit?(commit: Commit): void;
  onSegmentEnd?(at: number): void;
}

type Classify = (sequence: Float32Array) => Promise<Guess[]>;

interface Frame {
  t: number;
  f: Float32Array;
  hand: boolean;
}

const handFrames = (win: Frame[]) => win.reduce((n, fr) => n + (fr.hand ? 1 : 0), 0);

export class Decoder {
  private frames: Frame[] = [];
  private windowStart = -1; // earliest frame the next sign can use
  private lastHandAt = -Infinity;
  private lastStepAt = -Infinity;
  private busy = false;
  private runId = -1;
  private runCount = 0;
  private lastCommitted = -1;
  private inSegment = false;

  constructor(
    private classify: Classify,
    private events: DecoderEvents = {},
    private opts: DecoderOptions = DEFAULT_OPTIONS,
  ) {}

  reset() {
    this.frames = [];
    this.inSegment = false;
    this.runId = -1;
    this.runCount = 0;
    this.lastCommitted = -1;
    this.windowStart = -1;
  }

  /** Feed one frame. Resolves once any classification it triggered has finished. */
  async push(t: number, features: Float32Array, hasHand: boolean): Promise<void> {
    this.frames.push({ t, f: features, hand: hasHand });
    while (this.frames.length && t - this.frames[0].t > this.opts.maxBufferMs) this.frames.shift();
    if (hasHand) {
      if (!this.inSegment) {
        this.inSegment = true;
        this.windowStart = t - this.opts.preRollMs;
        this.runId = -1;
        this.runCount = 0;
        this.lastCommitted = -1;
      }
      this.lastHandAt = t;
      if (t - this.lastStepAt >= this.opts.stepMs && !this.busy) {
        this.lastStepAt = t;
        await this.step(t);
      }
      return;
    }
    if (this.inSegment && t - this.lastHandAt > this.opts.gapMs) {
      this.inSegment = false;
      await this.finish(t);
      this.events.onSegmentEnd?.(t);
    }
  }

  private windows(t: number): Frame[][] {
    const since = this.frames.filter((fr) => fr.t >= this.windowStart);
    const out: Frame[][] = [];
    for (const w of this.opts.windowsMs) {
      const win = since.filter((fr) => t - fr.t <= w);
      if (handFrames(win) >= this.opts.minFrames && !out.some((o) => o.length === win.length)) out.push(win);
    }
    return out;
  }

  private async best(wins: Frame[][]): Promise<Guess[] | null> {
    let best: Guess[] | null = null;
    for (const win of wins) {
      const guesses = await this.classify(tsnSample(win.map((fr) => fr.f)));
      if (!best || guesses[0].p > best[0].p) best = guesses;
    }
    return best;
  }

  private async step(t: number) {
    const wins = this.windows(t);
    if (!wins.length) return;
    this.busy = true;
    try {
      const guesses = await this.best(wins);
      if (!guesses) return;
      this.events.onLive?.(guesses, t);
      const top = guesses[0];
      if (top.p < this.opts.commitP) {
        this.runCount = 0;
        this.runId = -1;
        return;
      }
      if (top.id === this.runId) this.runCount++;
      else {
        this.runId = top.id;
        this.runCount = 1;
      }
      // Holding the same sign should not repeat it; a different confident sign clears the lock.
      if (top.id !== this.lastCommitted && this.runCount >= this.opts.commitRuns) {
        this.commit(guesses, t);
      }
    } finally {
      this.busy = false;
    }
  }

  private commit(guesses: Guess[], t: number) {
    const [top, ...alternatives] = guesses;
    this.lastCommitted = top.id;
    this.windowStart = t; // the next sign starts after this one
    this.runCount = 0;
    this.events.onCommit?.({ id: top.id, p: top.p, alternatives, at: t });
  }

  /** End of segment: classify whatever came after the last commit (or the whole segment). */
  private async finish(t: number) {
    const end = this.lastHandAt + this.opts.postRollMs;
    const since = this.frames.filter((fr) => fr.t >= this.windowStart && fr.t <= end);
    if (handFrames(since) < this.opts.minFrames) return;
    // After an early commit the leftover is usually the end of that same sign.
    const afterCommit = this.lastCommitted !== -1;
    if (afterCommit && since[since.length - 1].t - since[0].t < this.opts.tailMinMs) return;
    this.busy = true;
    try {
      const guesses = await this.best([since]);
      const bar = afterCommit ? this.opts.tailP : this.opts.finalP;
      if (guesses && guesses[0].p >= bar && guesses[0].id !== this.lastCommitted) this.commit(guesses, t);
    } finally {
      this.busy = false;
    }
  }
}
