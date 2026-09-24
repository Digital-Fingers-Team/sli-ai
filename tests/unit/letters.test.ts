// The browser letter model must see exactly what training/train_letters.py trained it on.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { LetterClassifier, handFeatures, type LetterModelFile } from '../../src/recognition/letters';
import type { Point } from '../../src/recognition/features';

const model: LetterModelFile = JSON.parse(readFileSync(new URL('../../public/models/letters.json', import.meta.url), 'utf8'));
const fixture: { frames: { label: string; lms: Point[] }[]; features: number[][]; probs: number[][] } = JSON.parse(
  readFileSync(new URL('../fixtures/letters-parity.json', import.meta.url), 'utf8'),
);

describe('letter model parity with Python', () => {
  // LetterClassifier.load fetches over HTTP; construct it from the file instead.
  const classifier = new (LetterClassifier as unknown as new (m: LetterModelFile) => LetterClassifier)(model);

  it('computes the same features', () => {
    fixture.frames.forEach((f, i) => {
      const got = handFeatures(f.lms, model.mirror !== '' && f.label === model.mirror, model.extra);
      expect(got.length).toBe(fixture.features[i].length);
      got.forEach((v, k) => expect(v).toBeCloseTo(fixture.features[i][k], 3));
    });
  });

  it('gives the same probabilities', () => {
    fixture.frames.forEach((f, i) => {
      const got = classifier.predict(f);
      got.forEach((v, k) => expect(v).toBeCloseTo(fixture.probs[i][k], 3));
    });
  });
});
