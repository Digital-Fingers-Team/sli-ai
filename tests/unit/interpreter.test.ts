import { describe, expect, it } from 'vitest';
import { signingHand } from '../../src/recognition/interpreter';
import type { Point, RawFrame } from '../../src/recognition/features';

const hand = (label: string, wristY: number) => ({
  label,
  lms: Array.from({ length: 21 }, (_, i): Point => [0.5, wristY - i * 0.005, 0]),
});

describe('signingHand', () => {
  it('uses the only hand in view', () => {
    const raw: RawFrame = { pose: null, face: null, hands: [hand('Left', 0.8)] };
    expect(signingHand(raw)?.label).toBe('Left');
  });

  it('picks the raised hand when both are in view', () => {
    const raw: RawFrame = { pose: null, face: null, hands: [hand('Left', 0.9), hand('Right', 0.4)] };
    expect(signingHand(raw)?.label).toBe('Right');
  });

  it('has no signing hand when none is in view', () => {
    expect(signingHand({ pose: null, face: null, hands: [] })).toBeUndefined();
  });
});
