// The letter model's small MLP: inference, and the on-device fine-tuning behind "teach the app
// your hand". Mirrors train() in training/train_letters_seq.py (ReLU hidden layers, softmax,
// Adam with L2 weight decay, cosine learning rate, batches of 256).

export interface LayerFile {
  rows: number;
  cols: number;
  w: number[]; // rows * cols, row-major
  b: number[];
}

interface Layer {
  rows: number;
  cols: number;
  w: Float32Array;
  b: Float32Array;
}

export class LetterNet {
  constructor(readonly layers: Layer[]) {}

  static fromFile(layers: LayerFile[]): LetterNet {
    return new LetterNet(layers.map((l) => ({ rows: l.rows, cols: l.cols, w: Float32Array.from(l.w), b: Float32Array.from(l.b) })));
  }

  clone(): LetterNet {
    return new LetterNet(this.layers.map((l) => ({ ...l, w: l.w.slice(), b: l.b.slice() })));
  }

  get inputs() {
    return this.layers[0].rows;
  }

  /** Activations of every layer for one input (the last is the logits). */
  private forwardAll(x: Float32Array): Float32Array[] {
    const acts = [x];
    this.layers.forEach((l, k) => {
      const inp = acts[acts.length - 1];
      const out = l.b.slice();
      for (let i = 0; i < l.rows; i++) {
        const v = inp[i];
        if (v === 0) continue;
        const row = i * l.cols;
        for (let j = 0; j < l.cols; j++) out[j] += v * l.w[row + j];
      }
      if (k < this.layers.length - 1) for (let j = 0; j < l.cols; j++) if (out[j] < 0) out[j] = 0;
      acts.push(out);
    });
    return acts;
  }

  /** Class probabilities. */
  predict(x: Float32Array): Float32Array {
    const acts = this.forwardAll(x);
    return softmax(acts[acts.length - 1]);
  }

  toFile(): LayerFile[] {
    return this.layers.map((l) => ({ rows: l.rows, cols: l.cols, w: Array.from(l.w), b: Array.from(l.b) }));
  }

  /**
   * Fine-tunes in place on labelled examples. `sample(i, rand)` returns example i's input,
   * freshly augmented, so each epoch sees a new variation. Yields to the page between epochs.
   */
  async fineTune(
    count: number,
    sample: (i: number, rand: Rand) => { x: Float32Array; y: number },
    opts: { epochs: number; lr: number; seed?: number; weightDecay?: number; batch?: number },
    onProgress?: (done: number) => void,
  ): Promise<void> {
    const rand = makeRand(opts.seed ?? 1);
    const wd = opts.weightDecay ?? 1e-4;
    const batch = opts.batch ?? 256;
    const params = this.layers.flatMap((l) => [l.w, l.b]);
    const m1 = params.map((p) => new Float32Array(p.length));
    const m2 = params.map((p) => new Float32Array(p.length));
    const grads = params.map((p) => new Float32Array(p.length));
    let step = 0;
    for (let epoch = 0; epoch < opts.epochs; epoch++) {
      const data = Array.from({ length: count }, (_, i) => sample(i, rand));
      const order = shuffle(count, rand);
      const rate = opts.lr * 0.5 * (1 + Math.cos((Math.PI * epoch) / opts.epochs));
      for (let start = 0; start < count; start += batch) {
        const idx = order.slice(start, start + batch);
        grads.forEach((g) => g.fill(0));
        for (const i of idx) this.backprop(data[i].x, data[i].y, grads, 1 / idx.length);
        step++;
        params.forEach((p, k) => {
          const g = grads[k];
          const isWeight = k % 2 === 0;
          const a = m1[k];
          const v = m2[k];
          const c1 = 1 - 0.9 ** step;
          const c2 = 1 - 0.999 ** step;
          for (let j = 0; j < p.length; j++) {
            const gj = g[j] + (isWeight ? wd * p[j] : 0);
            a[j] = 0.9 * a[j] + 0.1 * gj;
            v[j] = 0.999 * v[j] + 0.001 * gj * gj;
            p[j] -= (rate * (a[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8);
          }
        });
      }
      onProgress?.((epoch + 1) / opts.epochs);
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  /** Adds scale × the cross-entropy gradient of one example to `grads` ([w0, b0, w1, b1, ...]). */
  private backprop(x: Float32Array, y: number, grads: Float32Array[], scale: number) {
    const acts = this.forwardAll(x);
    let g = softmax(acts[acts.length - 1]);
    g[y] -= 1;
    for (let k = this.layers.length - 1; k >= 0; k--) {
      const l = this.layers[k];
      const inp = acts[k];
      const gw = grads[2 * k];
      const gb = grads[2 * k + 1];
      for (let j = 0; j < l.cols; j++) gb[j] += g[j] * scale;
      for (let i = 0; i < l.rows; i++) {
        const v = inp[i] * scale;
        if (v === 0) continue;
        const row = i * l.cols;
        for (let j = 0; j < l.cols; j++) gw[row + j] += v * g[j];
      }
      if (k === 0) break;
      const back = new Float32Array(l.rows);
      for (let i = 0; i < l.rows; i++) {
        if (inp[i] <= 0) continue; // ReLU of the previous layer
        const row = i * l.cols;
        let s = 0;
        for (let j = 0; j < l.cols; j++) s += l.w[row + j] * g[j];
        back[i] = s;
      }
      g = back;
    }
  }
}

export function softmax(x: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of x) max = Math.max(max, v);
  const out = new Float32Array(x.length);
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += out[i] = Math.exp(x[i] - max);
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

export interface Rand {
  (): number; // uniform [0, 1)
  normal(sd: number): number;
  uniform(lo: number, hi: number): number;
}

/** Seeded generator (mulberry32) so fine-tuning is reproducible. */
export function makeRand(seed: number): Rand {
  let s = seed >>> 0;
  const r = (() => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }) as Rand;
  r.normal = (sd) => sd * Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  r.uniform = (lo, hi) => lo + (hi - lo) * r();
  return r;
}

function shuffle(n: number, rand: Rand): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
