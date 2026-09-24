// Live fingerspelling: a small per-frame hand-shape model, trained on KArSL's letter videos by
// training/train_letters.py. It sees one frame at a time, so letters show up while the hand is
// still up instead of after the sign ends. handFeatures must match letter_features() there.

import type { Point } from './features';

interface Layer {
  rows: number; // inputs
  cols: number; // outputs
  w: number[]; // rows * cols, row-major
  b: number[];
}

export interface LetterModelFile {
  classes: number[]; // sign ids, in output order
  mirror: string; // handedness label mirrored onto the other hand ('' = none)
  extra: boolean; // fingertip distances appended
  layers: Layer[];
}

const TIPS = [0, 4, 8, 12, 16, 20];
const PAIRS = TIPS.flatMap((a, i) => TIPS.slice(i + 1).map((b) => [a, b] as const));

/**
 * 21 points relative to the wrist, scaled by the wrist→middle-knuckle length, then (with
 * `extra`) the distances between fingertips and wrist. `flip` mirrors a left hand onto the
 * right-handed layout the model was trained on.
 */
export function handFeatures(lms: Point[], flip: boolean, extra: boolean): Float32Array {
  const [wx, wy, wz] = lms[0];
  const m = lms[9];
  const scale = Math.max(Math.hypot(m[0] - wx, m[1] - wy), 1e-6);
  const out = new Float32Array(63 + (extra ? PAIRS.length : 0));
  lms.forEach((p, i) => {
    out[i * 3] = ((flip ? -1 : 1) * (p[0] - wx)) / scale;
    out[i * 3 + 1] = (p[1] - wy) / scale;
    out[i * 3 + 2] = (p[2] - wz) / scale;
  });
  if (extra) {
    PAIRS.forEach(([a, b], k) => {
      out[63 + k] = Math.hypot(out[a * 3] - out[b * 3], out[a * 3 + 1] - out[b * 3 + 1], out[a * 3 + 2] - out[b * 3 + 2]);
    });
  }
  return out;
}

export class LetterClassifier {
  private constructor(private model: LetterModelFile) {}

  static async load(base: string): Promise<LetterClassifier> {
    const res = await fetch(`${base}models/letters.json`);
    if (!res.ok) throw new Error(`letters model: HTTP ${res.status}`);
    return new LetterClassifier(await res.json());
  }

  get classes() {
    return this.model.classes;
  }

  /** Probabilities over `classes` for one hand; `leftHanded` mirrors it first. */
  predict(hand: { label: string; lms: Point[] }, leftHanded = false): Float32Array {
    const flip = leftHanded !== (this.model.mirror !== '' && hand.label === this.model.mirror);
    let x = handFeatures(hand.lms, flip, this.model.extra);
    this.model.layers.forEach((layer, k) => {
      const y = new Float32Array(layer.cols);
      for (let j = 0; j < layer.cols; j++) {
        let s = layer.b[j];
        for (let i = 0; i < layer.rows; i++) s += x[i] * layer.w[i * layer.cols + j];
        y[j] = k < this.model.layers.length - 1 ? Math.max(0, s) : s;
      }
      x = y;
    });
    let max = -Infinity;
    for (const v of x) max = Math.max(max, v);
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] = Math.exp(x[i] - max);
    for (let i = 0; i < x.length; i++) x[i] /= sum;
    return x;
  }
}
