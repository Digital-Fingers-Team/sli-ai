import { describe, expect, it } from 'vitest';
import { DEFAULT_SPELLER, Speller } from '../../src/recognition/speller';

const CLASSES = [100, 101, 102];
const probs = (k: number, p = 0.9) => Float32Array.from(CLASSES, (_, i) => (i === k ? p : (1 - p) / 2));

/** Script of [letter index | null (no hand) | -1 (hand between shapes), duration ms] at 15 fps. */
function run(script: [number | null, number][]) {
  const typed: number[] = [];
  const words: number[] = [];
  const s = new Speller(CLASSES, {
    onCommit: (c) => typed.push(c.id),
    onSegmentEnd: () => words.push(typed.length),
  });
  let t = 0;
  for (const [k, ms] of script) {
    for (const end = t + ms; t < end; t += 66) s.push(t, k === null ? null : k < 0 ? Float32Array.of(0.25, 0.25, 0.5) : probs(k));
  }
  return { typed, words };
}

describe('Speller', () => {
  it('types a held letter once', () => {
    expect(run([[0, 2000], [null, 1000]]).typed).toEqual([100]);
  });

  it('types letters one after another without lowering the hand', () => {
    expect(run([[0, 600], [1, 600], [2, 600], [null, 1000]]).typed).toEqual([100, 101, 102]);
  });

  it('does not type a shape that only passes by', () => {
    const passing = DEFAULT_SPELLER.holdMs - 150;
    expect(run([[0, 600], [2, passing], [1, 600]]).typed).toEqual([100, 101]);
  });

  it('types a double letter when the hand relaxes in between', () => {
    expect(run([[0, 600], [-1, 300], [0, 600]]).typed).toEqual([100, 100]);
  });

  it('survives a brief tracking dropout without doubling the letter', () => {
    expect(run([[0, 600], [null, 130], [0, 600]]).typed).toEqual([100]);
  });

  it('ends the word when the hand is lowered', () => {
    const { typed, words } = run([[0, 600], [1, 600], [null, 1000], [2, 600], [null, 1000]]);
    expect(typed).toEqual([100, 101, 102]);
    expect(words).toEqual([2, 3]);
  });
});
