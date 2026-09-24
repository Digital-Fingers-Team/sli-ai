// Frame-by-frame interpretation of landmarks into signs, in one of three modes:
//   words   - the word decoder only (the sequence model, 502 signs);
//   letters - the speller only (the hand-shape model, see letters.ts);
//   auto    - both, so a sentence can mix signed words and spelled names. Letters are held
//             still and words move, so the speller only types while the hand is still and
//             raised. Words have still moments too, so the word model (which also knows the
//             letter signs) can veto a letter while it confidently sees a word. When one of
//             them commits, the other skips the frames already used.
// The browser engine and tests/replay both drive this class, so what is measured is what runs.

import { frameFeatures, type Point, type RawFrame } from './features';
import { Decoder, DEFAULT_OPTIONS, type Commit, type DecoderEvents, type DecoderOptions } from './decoder';
import { DEFAULT_SPELLER, Speller, type SpellerOptions } from './speller';
import type { LetterClassifier, LetterStream } from './letters';
import type { Guess } from './topk';

export type SignMode = 'auto' | 'words' | 'letters';

export interface InterpreterOptions {
  decoder: DecoderOptions;
  speller: SpellerOptions;
  stillSpeed: number; // palm lengths per second below which the hand counts as still
  motionSmoothing: number; // weight of the newest frame in the speed average
  minHeight: number; // wrist height over the shoulder line, in shoulder widths, to spell (hands in the lap are not letters)
  wordVetoP: number; // a letter is not typed while the word model sees a non-letter sign this confidently
}

export const DEFAULT_INTERPRETER: InterpreterOptions = {
  decoder: DEFAULT_OPTIONS,
  speller: DEFAULT_SPELLER,
  stillSpeed: 1, // letter holds: median 0.27; words: 4.9 (signer 01, 15 fps)
  motionSmoothing: 0.5,
  minHeight: -0.6, // letter holds: 5th percentile -0.46
  wordVetoP: 0.5,
};

const isLetterSign = (id: number) => id >= 31 && id <= 69; // KArSL classes 32-70

type Hand = RawFrame['hands'][number];

/** The hand doing the spelling: with two in view, the raised one. */
export function signingHand(raw: RawFrame): Hand | undefined {
  if (raw.hands.length < 2) return raw.hands[0];
  return raw.hands.reduce((a, b) => (b.lms[0][1] < a.lms[0][1] ? b : a));
}

/** Wrist height above the shoulder line, in shoulder widths. */
function height(hand: Hand, pose: NonNullable<RawFrame['pose']>): number {
  const [l, r] = [pose[11], pose[12]];
  const width = Math.max(Math.hypot(l[0] - r[0], l[1] - r[1]), 1e-6);
  return ((l[1] + r[1]) / 2 - hand.lms[0][1]) / width;
}

const palm = (lms: Point[]) => Math.max(Math.hypot(lms[9][0] - lms[0][0], lms[9][1] - lms[0][1]), 1e-6);

export class Interpreter {
  private decoder: Decoder;
  private speller: Speller;
  private prev: { t: number; lms: Point[] } | null = null;
  private letterStream: LetterStream;
  private speed = 0; // palm lengths per second, smoothed

  constructor(
    classify: (seq: Float32Array) => Promise<Guess[]>,
    letters: LetterClassifier,
    private events: DecoderEvents,
    private mode: SignMode = 'auto',
    private opts: InterpreterOptions = DEFAULT_INTERPRETER,
  ) {
    this.decoder = new Decoder(
      classify,
      {
        onLive: (g, at) => {
          this.wordLive = g;
          // In auto mode a held, confident letter takes over the preview.
          if (this.mode !== 'auto' || !this.spelling) this.events.onLive?.(g, at);
        },
        onCommit: (c) => {
          this.speller.reset();
          this.events.onCommit?.(c);
        },
        onSegmentEnd: (at) => {
          this.wordLive = null;
          this.events.onSegmentEnd?.(at);
        },
      },
      opts.decoder,
    );
    this.letterStream = letters.stream();
    this.speller = new Speller(
      letters.classes,
      {
        onLive: (g, at) => {
          this.spelling = this.still && this.raised && g[0].p >= opts.speller.minP && !this.wordVeto();
          if (this.mode === 'letters' || this.spelling) this.events.onLive?.(g, at);
        },
        onCommit: (c: Commit) => {
          if (this.mode === 'auto' && this.wordVeto()) return;
          this.decoder.consume(c.at, c.id);
          this.events.onCommit?.(c);
        },
        onSegmentEnd: (at) => {
          if (this.mode === 'letters') this.events.onSegmentEnd?.(at);
        },
      },
      opts.speller,
    );
  }

  private still = false;
  private raised = true;
  private spelling = false;
  private wordLive: Guess[] | null = null; // the word decoder's latest view of the same frames

  private wordVeto() {
    const top = this.wordLive?.[0];
    return !!top && !isLetterSign(top.id) && top.p >= this.opts.wordVetoP;
  }

  setMode(mode: SignMode) {
    this.mode = mode;
    this.reset();
  }

  reset() {
    this.decoder.reset();
    this.speller.reset();
    this.letterStream.reset();
    this.prev = null;
    this.speed = 0;
    this.spelling = false;
    this.wordLive = null;
  }

  /** Whether pose and face are needed for this frame (the letter model uses the hand only). */
  get needsBody() {
    return this.mode !== 'letters';
  }

  async push(t: number, raw: RawFrame): Promise<void> {
    const hand = signingHand(raw);
    this.trackMotion(t, hand);
    this.raised = !hand || !raw.pose || height(hand, raw.pose) >= this.opts.minHeight;
    const holding = this.mode === 'letters' || (this.still && this.raised);
    if (this.mode !== 'words') this.speller.push(t, this.letterStream.push(hand ?? null), holding);
    if (this.mode !== 'letters') await this.decoder.push(t, frameFeatures(raw), raw.hands.length > 0);
  }

  private trackMotion(t: number, hand: Hand | undefined) {
    if (!hand) {
      this.prev = null;
      this.still = false;
      return;
    }
    if (this.prev && t > this.prev.t) {
      const scale = palm(hand.lms);
      let d = 0;
      hand.lms.forEach((p, i) => (d += Math.hypot(p[0] - this.prev!.lms[i][0], p[1] - this.prev!.lms[i][1])));
      const v = d / hand.lms.length / scale / ((t - this.prev.t) / 1000);
      const a = this.opts.motionSmoothing;
      this.speed = a * v + (1 - a) * this.speed;
      this.still = this.speed < this.opts.stillSpeed;
    } else {
      this.speed = 0;
      this.still = false;
    }
    this.prev = { t, lms: hand.lms };
  }
}
