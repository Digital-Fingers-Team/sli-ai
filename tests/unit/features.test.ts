// Parity with the Python pipeline the model was validated with (training/export_parity.py).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-node';
import { frameFeatures, tsnSample, FRAME_SIZE, SEQ_LEN, type RawFrame } from '../../src/recognition/features';
import { softmaxTopK } from '../../src/recognition/topk';

interface Case {
  name: string;
  raws: RawFrame[];
  input_sum: number;
  input_head: number[];
  top5: number[];
}

const cases: Case[] = JSON.parse(readFileSync(new URL('../fixtures/parity.json', import.meta.url), 'utf8'));

describe('features match the Python pipeline', () => {
  it.each(cases.map((c) => [c.name, c] as const))('%s', async (_name, c) => {
    const seq = tsnSample(c.raws.map(frameFeatures));
    expect(seq.length).toBe(SEQ_LEN * FRAME_SIZE);

    let sum = 0;
    for (const v of seq) sum += Math.abs(v);
    expect(sum).toBeCloseTo(c.input_sum, 0);
    c.input_head.forEach((v, i) => expect(seq[i]).toBeCloseTo(v, 4));

    const session = await ort.InferenceSession.create(
      new URL('../../public/models/karsl502.onnx', import.meta.url).pathname,
    );
    const out = await session.run({ input: new ort.Tensor('float32', seq, [1, SEQ_LEN, FRAME_SIZE]) });
    const top = softmaxTopK(out.output.data as Float32Array, 5).map((g) => g.id);
    expect(top).toEqual(c.top5);
  });
});
