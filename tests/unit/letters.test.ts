// The browser letter model must see exactly what training/train_letters_seq.py trained it on,
// and its on-device fine-tuning must reproduce the calibration gain measured in Python.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  LetterClassifier,
  augment,
  featurise,
  recordingWindows,
  windowFrom,
  type HandFrame,
  type LetterModelFile,
} from '../../src/recognition/letters';
import { LetterNet } from '../../src/recognition/letternet';

const load = <T>(path: string): T => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const model = load<LetterModelFile>('../../public/models/letters-seq.json');

describe('letter model parity with Python', () => {
  const parity = load<{ cases: { frames: (HandFrame | null)[]; ends: number[]; features: number[][]; probs: number[][] }[] }>(
    '../fixtures/letters-seq-parity.json',
  );
  const clf = new LetterClassifier(model);

  it('builds the same windows and features from a frame stream', () => {
    for (const c of parity.cases) {
      c.ends.forEach((end, k) => {
        const s = windowFrom(c.frames.slice(0, end + 1), model.window, model.mirror)!;
        const got = featurise(s);
        expect(got.length).toBe(c.features[k].length);
        got.forEach((v, i) => expect(v).toBeCloseTo(c.features[k][i], 3));
      });
    }
  });

  it('gives the same probabilities', () => {
    for (const c of parity.cases) {
      c.ends.forEach((end, k) => {
        const got = clf.predict(windowFrom(c.frames.slice(0, end + 1), model.window, model.mirror)!);
        got.forEach((v, i) => expect(v).toBeCloseTo(c.probs[k][i], 3));
      });
    }
  });

  it('streams: a frame without a hand gives no letter', () => {
    const stream = clf.stream();
    const f = parity.cases[0].frames.find((x) => x)!;
    expect(stream.push(f)).not.toBeNull();
    expect(stream.push(null)).toBeNull();
  });
});

type Video = { sign: number; frames: (HandFrame | null)[] };

describe('teach the app your hand (calibration)', () => {
  const fx = load<{ model: LetterModelFile; span: [number, number]; calibration: Video[]; test: Video[]; python: { base: number; finetuned: number; epochs: number; lr: number } }>(
    '../fixtures/letters-calib.json',
  );
  const index = new Map(fx.model.classes.map((c, i) => [c, i]));

  /** Per-video accuracy on 15 fps windows, probabilities averaged over the video (as in Python). */
  function accuracy(net: LetterNet) {
    let ok = 0;
    let n = 0;
    for (const v of fx.test) {
      const wins = recordingWindows(v.frames, fx.model.window, fx.model.mirror, 2, 0, fx.span);
      if (!wins.length) continue;
      const mean = new Float32Array(fx.model.classes.length);
      for (const w of wins) net.predict(featurise(w)).forEach((p, i) => (mean[i] += p));
      ok += mean.indexOf(Math.max(...mean)) === index.get(v.sign) ? 1 : 0;
      n++;
    }
    return ok / n;
  }

  it('starts where Python did on a signer the model never saw', () => {
    expect(accuracy(LetterNet.fromFile(fx.model.layers))).toBeCloseTo(fx.python.base, 2);
  });

  it('fine-tuning on one recording per letter gains as much as in Python', async () => {
    const base = LetterNet.fromFile(fx.model.layers);
    const before = accuracy(base);
    // Windows at 30, 15 and 10 fps from each recording, like dataset(strides=(1, 2, 3)).
    const examples = fx.calibration.flatMap((v) =>
      [1, 2, 3].flatMap((stride) =>
        recordingWindows(v.frames, fx.model.window, fx.model.mirror, stride, 0, fx.span).map((s) => ({ s, y: index.get(v.sign)! })),
      ),
    );
    const tuned = base.clone();
    await tuned.fineTune(examples.length, (i, rand) => ({ x: featurise(augment(examples[i].s, rand)), y: examples[i].y }), {
      epochs: fx.python.epochs,
      lr: fx.python.lr,
    });
    const after = accuracy(tuned);
    console.log(JSON.stringify({ examples: examples.length, before, after, python: fx.python }));
    expect(after - before).toBeGreaterThan(0.6 * (fx.python.finetuned - fx.python.base));
  }, 120_000);
});
