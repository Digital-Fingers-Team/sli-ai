import { describe, expect, it } from 'vitest';
import { Decoder, DEFAULT_OPTIONS, type Commit } from '../../src/recognition/decoder';
import { FRAME_SIZE } from '../../src/recognition/features';
import type { Guess } from '../../src/recognition/topk';

// Fake classifier: the "sign" is encoded in feature[0]; the window's majority wins, and mixed
// windows are less confident, like the real model on a transition. Frames without a hand carry
// NONE and, like the real model on an empty picture, give no confident answer.
const NONE = 400;
function fakeClassify(seq: Float32Array): Promise<Guess[]> {
  const counts = new Map<number, number>();
  const frames = seq.length / FRAME_SIZE;
  for (let t = 0; t < frames; t++) {
    const id = Math.round(seq[t * FRAME_SIZE]);
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  const [id, n] = [...counts].sort((a, b) => b[1] - a[1])[0];
  return Promise.resolve([
    { id, p: id === NONE ? 0.1 : n / frames },
    { id: id + 1, p: 0.01 },
  ]);
}

const frame = (id: number) => {
  const f = new Float32Array(FRAME_SIZE);
  f[0] = id;
  return f;
};

async function run(script: [number | null, number][]) {
  const commits: Commit[] = [];
  const d = new Decoder(fakeClassify, { onCommit: (c) => commits.push(c) });
  let t = 0;
  for (const [id, ms] of script) {
    for (let end = t + ms; t < end; t += 33) await d.push(t, frame(id ?? NONE), id !== null);
  }
  return commits.map((c) => c.id);
}

describe('Decoder', () => {
  it('commits a held sign once, not repeatedly', async () => {
    expect(await run([[7, 3000], [null, 1000]])).toEqual([7]);
  });

  it('commits consecutive different signs without dropping the hands', async () => {
    expect(await run([[3, 2500], [9, 2500], [null, 1000]])).toEqual([3, 9]);
  });

  it('does not add a word for the end of a sign it already committed', async () => {
    expect(await run([[3, 2600], [null, 1000]])).toEqual([3]);
  });

  it('allows the same sign again after the hands drop', async () => {
    expect(await run([[4, 1200], [null, 800], [4, 1200], [null, 800]])).toEqual([4, 4]);
  });

  it('ignores flickers shorter than the minimum window', async () => {
    const tooShort = (DEFAULT_OPTIONS.minFrames - 2) * 33;
    expect(await run([[5, tooShort], [null, 1000]])).toEqual([]);
  });

  it('does not commit from the lead-in alone', async () => {
    expect(await run([[null, 2000], [6, 60], [null, 1000]])).toEqual([]);
  });

  it('keeps frames where the hand was briefly lost', async () => {
    const seen: Float32Array[] = [];
    const d = new Decoder(async (seq) => {
      seen.push(seq);
      return [{ id: 1, p: 0.3 }];
    });
    let t = 0;
    for (; t < 400; t += 33) await d.push(t, frame(1), true);
    for (; t < 600; t += 33) await d.push(t, frame(NONE), false);
    for (; t < 1200; t += 33) await d.push(t, frame(1), true);
    const last = seen[seen.length - 1];
    const values = Array.from({ length: last.length / FRAME_SIZE }, (_, i) => Math.round(last[i * FRAME_SIZE]));
    expect(values).toContain(NONE);
    expect(values).toContain(1);
  });

  it('commits a short sign at the end of the segment', async () => {
    const commits: Commit[] = [];
    const d = new Decoder(
      async () => [{ id: 12, p: 0.45 }],
      { onCommit: (c) => commits.push(c) },
    );
    let t = 0;
    for (; t < 600; t += 33) await d.push(t, frame(12), true);
    for (; t < 1500; t += 33) await d.push(t, frame(NONE), false);
    expect(commits.map((c) => c.id)).toEqual([12]);
  });
});
