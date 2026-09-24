import { describe, expect, it } from 'vitest';
import { FramingCoach, frameIssues } from '../../src/recognition/framing';
import type { PosePoint, RawFrame } from '../../src/recognition/features';

/** A pose with shoulders `width` apart around x = 0.5 and the given wrist positions. */
function pose(width: number, left: [number, number], right: [number, number]): RawFrame {
  const p: PosePoint[] = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1]);
  p[11] = [0.5 + width / 2, 0.4, 0, 1];
  p[12] = [0.5 - width / 2, 0.4, 0, 1];
  p[13] = [0.5 + width / 2 + 0.05, 0.6, 0, 1];
  p[14] = [0.5 - width / 2 - 0.05, 0.6, 0, 1];
  p[15] = [...left, 0, 0.3];
  p[16] = [...right, 0, 0.3];
  return { pose: p, face: null, hands: [] };
}

describe('frameIssues', () => {
  it('accepts KArSL-like framing', () => {
    expect([...frameIssues(pose(0.32, [0.7, 0.5], [0.3, 0.5]), 0.48)]).toEqual([]);
  });

  it('flags a signer too close to the camera', () => {
    expect(frameIssues(pose(0.6, [0.8, 0.5], [0.2, 0.5]), 0.48).has('tooClose')).toBe(true);
  });

  it('flags a raised hand past the edge while its elbow is in view', () => {
    expect(frameIssues(pose(0.4, [1.1, 0.45], [0.3, 0.5]), 0.48).has('handOut')).toBe(true);
  });

  it('does not flag a hand resting below the picture', () => {
    expect(frameIssues(pose(0.4, [0.75, 1.2], [0.25, 1.2]), 0.48).has('handOut')).toBe(false);
  });
});

describe('FramingCoach', () => {
  it('reports an issue only once it has held for a while', () => {
    const coach = new FramingCoach();
    const out = pose(0.4, [1.1, 0.45], [0.3, 0.5]);
    const ok = pose(0.32, [0.7, 0.5], [0.3, 0.5]);
    expect(coach.push(0, out)).toBe(null); // one frame is not enough
    let t = 0;
    for (; t < 600; t += 66) coach.push(t, out);
    expect(coach.push(t, out)).toBe('handOut');
    for (; t < 3000; t += 66) coach.push(t, ok);
    expect(coach.push(t, ok)).toBe(null);
  });
});
