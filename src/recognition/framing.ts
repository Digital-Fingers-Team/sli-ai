// Is the signer framed so both hands can stay in the picture? The word model was trained on
// KArSL, where the shoulders take about a third of the image width (0.32 median) and signs made
// far from the body stay in view. A phone held upright and close cuts hands off while the
// elbow is still visible, and two-handed signs such as أهلا وسهلا then fail
// (training/eval_framing.py measures how much). This watches the pose and names the problem.

import type { RawFrame } from './features';

export type FramingIssue = 'tooClose' | 'handOut';

export interface FramingOptions {
  maxShoulders: number; // shoulder width as a share of image width above which we are too close
  windowMs: number; // judged over this much recent time
  share: number; // an issue is reported when it holds for this share of the window
}

// Some signs reach out of the picture for a moment even as KArSL films them (تفضل, ولادة,
// مريض) and are recognised fine, so only a hand that stays clearly outside counts: 3 of the
// 1,506 test videos get a warning, against 14 when any touch of the edge for 1.5 s counted
// (tests/replay/framing.test.ts).
export const DEFAULT_FRAMING: FramingOptions = {
  maxShoulders: 0.48,
  windowMs: 2000,
  share: 0.5,
};

// How far past the edge (in image widths/heights) a wrist must be to count as outside.
const MARGIN = 0.03;
const VISIBLE = 0.5;

/** The framing problems seen in one frame. */
export function frameIssues(raw: RawFrame, maxShoulders: number): Set<FramingIssue> {
  const out = new Set<FramingIssue>();
  const p = raw.pose;
  if (!p) return out;
  if (Math.abs(p[11][0] - p[12][0]) > maxShoulders) out.add('tooClose');
  // Pose estimates the wrist even outside the picture; an arm whose elbow is in view but whose
  // wrist is past the edge means that hand is cut off.
  for (const [elbow, wrist] of [
    [13, 15],
    [14, 16],
  ]) {
    const e = p[elbow];
    const w = p[wrist];
    const elbowIn = e[3] > VISIBLE && e[0] > 0 && e[0] < 1 && e[1] > 0 && e[1] < 1;
    const wristOut = w[0] < -MARGIN || w[0] > 1 + MARGIN || w[1] < -MARGIN;
    // A wrist below the picture is a hand at rest, not a sign cut off.
    const raised = w[1] < e[1] + 0.05;
    if (elbowIn && wristOut && raised) out.add('handOut');
  }
  return out;
}

export class FramingCoach {
  private seen: { t: number; issues: Set<FramingIssue> }[] = [];

  constructor(private opts: FramingOptions = DEFAULT_FRAMING) {}

  /** The issue to show now, if one has held for a while (hand out of frame first). */
  push(t: number, raw: RawFrame): FramingIssue | null {
    this.seen.push({ t, issues: frameIssues(raw, this.opts.maxShoulders) });
    while (this.seen.length && t - this.seen[0].t > this.opts.windowMs) this.seen.shift();
    for (const issue of ['handOut', 'tooClose'] as FramingIssue[]) {
      const n = this.seen.filter((s) => s.issues.has(issue)).length;
      if (this.seen.length >= 3 && n / this.seen.length >= this.opts.share) return issue;
    }
    return null;
  }

  reset() {
    this.seen = [];
  }
}
