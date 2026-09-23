// Sentence-level check of the whole recognition path (features -> decoder -> real model) on
// landmarks extracted from held-out KArSL videos, at the frame rates phones actually reach.
// Needs training/cache (made by training/eval_pretrained.py). Run: SLI_REPLAY=1 npx vitest run
import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-node';
import { frameFeatures, FRAME_SIZE, SEQ_LEN, type RawFrame } from '../../src/recognition/features';
import { softmaxTopK } from '../../src/recognition/topk';
import { Decoder, DEFAULT_OPTIONS } from '../../src/recognition/decoder';

const CACHE = new URL('../../training/cache/', import.meta.url).pathname;
const TRIALS = Number(process.env.SLI_REPLAY_TRIALS ?? 40);
const WORDS = 3;

// Deterministic pseudo-random so runs are comparable while tuning.
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);

const files = readdirSync(CACHE).filter((f) => f.startsWith('h1_'));
const signOf = (f: string) => Number(f.split('_')[3]) - 1;

function handsDown(raw: RawFrame): RawFrame {
  return { ...raw, hands: [] };
}

async function trial(session: ort.InferenceSession, fps: number) {
  const picks = Array.from({ length: WORDS }, () => files[Math.floor(rand() * files.length)]);
  const videos = picks.map((f) => JSON.parse(readFileSync(CACHE + f, 'utf8')) as RawFrame[]);
  const rest = handsDown(videos[0].find((r) => r.pose) ?? videos[0][0]);
  const stream: RawFrame[] = [];
  const gap = () => stream.push(...Array.from({ length: 30 }, () => rest));
  gap();
  for (const v of videos) {
    stream.push(...v);
    gap();
  }
  const commits: number[] = [];
  const decoder = new Decoder(
    async (seq) => {
      const out = await session.run({ input: new ort.Tensor('float32', seq, [1, SEQ_LEN, FRAME_SIZE]) });
      return softmaxTopK(out.output.data as Float32Array, 5);
    },
    { onCommit: (c) => commits.push(c.id) },
    { ...DEFAULT_OPTIONS, ...JSON.parse(process.env.SLI_OPTS ?? '{}') },
  );
  // Source is 30 fps; a device at `fps` sees every (30/fps)-th frame.
  let last = -1;
  for (let i = 0; i < stream.length; i++) {
    const tick = Math.floor((i * fps) / 30);
    if (tick === last) continue;
    last = tick;
    await decoder.push((i * 1000) / 30, frameFeatures(stream[i]), stream[i].hands.length > 0);
  }
  return { truth: picks.map(signOf), commits };
}

describe.runIf(process.env.SLI_REPLAY)('sentence replay', () => {
  for (const fps of (process.env.SLI_FPS ?? '10,15,30').split(',').map(Number)) {
    it(`${fps} fps`, async () => {
      const session = await ort.InferenceSession.create(new URL('../../public/models/karsl502.onnx', import.meta.url).pathname);
      let exact = 0;
      let wordsRight = 0;
      let extra = 0;
      for (let k = 0; k < TRIALS; k++) {
        const { truth, commits } = await trial(session, fps);
        if (commits.join() === truth.join()) exact++;
        wordsRight += truth.filter((id, i) => commits[i] === id).length;
        extra += Math.max(0, commits.length - truth.length);
        if (process.env.SLI_REPLAY_VERBOSE && commits.join() !== truth.join()) console.log(fps, 'truth', truth, 'got', commits);
      }
      const summary = { fps, sentences: TRIALS, exact: exact / TRIALS, words: wordsRight / (TRIALS * WORDS), extraPerSentence: extra / TRIALS };
      console.log(JSON.stringify(summary));
      expect(summary.words).toBeGreaterThan(0.85);
    }, 600_000);
  }
});
