// Does the framing coach stay quiet on well-framed signing? Runs it over every cached KArSL
// test video (as recorded, the framing the model was trained on), including signs where a hand
// is hidden or drops below the picture. Run: SLI_REPLAY=1 npx vitest run tests/replay/framing.test.ts
import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { FramingCoach } from '../../src/recognition/framing';
import type { RawFrame } from '../../src/recognition/features';

const CACHE = new URL('../../training/cache/', import.meta.url).pathname;

describe.runIf(process.env.SLI_REPLAY)('framing coach on KArSL framing', () => {
  it('does not warn on signs filmed as the model expects', () => {
    const files = readdirSync(CACHE).filter((f) => f.startsWith('h1_'));
    const warned: Record<string, string[]> = { handOut: [], tooClose: [] };
    let handMissing = 0; // videos where a hand is out of sight for part of the sign
    for (const f of files) {
      const coach = new FramingCoach();
      const frames: RawFrame[] = JSON.parse(readFileSync(CACHE + f, 'utf8'));
      const seen = new Set<string>();
      frames.forEach((raw, i) => {
        const issue = coach.push((i * 1000) / 30, raw);
        if (issue) seen.add(issue);
      });
      if (frames.some((r) => r.pose && r.hands.length === 0 && r.pose[15][1] < 1 && r.pose[16][1] < 1)) handMissing++;
      for (const issue of seen) warned[issue].push(f);
    }
    const summary = {
      videos: files.length,
      withAHandOutOfSight: handMissing,
      warnedHandOut: warned.handOut.length,
      warnedTooClose: warned.tooClose.length,
      examples: warned.handOut.slice(0, 5),
    };
    console.log(JSON.stringify(summary));
    expect((warned.handOut.length + warned.tooClose.length) / files.length).toBeLessThan(0.005);
  }, 300_000);
});
