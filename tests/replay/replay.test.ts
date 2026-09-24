// Sentence-level check of the whole recognition path (Interpreter: features -> decoder -> real
// word model, and the letter model + speller) on landmarks extracted from held-out KArSL
// videos, at the frame rates phones actually reach.
// Needs training/cache (made by training/eval_pretrained.py). Run: SLI_REPLAY=1 npx vitest run
//   SLI_MODE=auto|words|letters   interpreter mode (default words)
//   SLI_MIX=any|mixed|letters     sentences of 3 random signs / word, letter, letter, word / 3 letters
//   SLI_SIGNER=01                 only that signer's videos (pair with a letter model trained
//   SLI_LETTERS_MODEL=path        without them: train_letters.py --exclude 01)
import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-node';
import { FRAME_SIZE, SEQ_LEN, type RawFrame } from '../../src/recognition/features';
import { softmaxTopK } from '../../src/recognition/topk';
import { DEFAULT_OPTIONS } from '../../src/recognition/decoder';
import { DEFAULT_SPELLER } from '../../src/recognition/speller';
import { DEFAULT_INTERPRETER, Interpreter, type SignMode } from '../../src/recognition/interpreter';
import { LetterClassifier, type LetterModelFile } from '../../src/recognition/letters';

const CACHE = new URL('../../training/cache/', import.meta.url).pathname;
const TRIALS = Number(process.env.SLI_REPLAY_TRIALS ?? 40);
const MODE = (process.env.SLI_MODE ?? 'words') as SignMode;
const MIX = process.env.SLI_MIX ?? 'any';
const lettersModel: LetterModelFile = JSON.parse(
  readFileSync(process.env.SLI_LETTERS_MODEL ?? new URL('../../public/models/letters.json', import.meta.url).pathname, 'utf8'),
);
const letters = new (LetterClassifier as unknown as new (m: LetterModelFile) => LetterClassifier)(lettersModel);
// SLI_CONTINUOUS=1: signs follow each other without lowering the hands (each clip's hand-less
// start and end are cut). SLI_BODY_EVERY=N: pose and face refreshed every N-th frame, as the app does.
const CONTINUOUS = !!process.env.SLI_CONTINUOUS;
const BODY_EVERY = Number(process.env.SLI_BODY_EVERY ?? 2);

// Deterministic pseudo-random so runs are comparable while tuning.
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);

const signOf = (f: string) => Number(f.split('_')[3]) - 1;
const files = readdirSync(CACHE).filter((f) => f.startsWith('h1_') && (!process.env.SLI_SIGNER || f.split('_')[2] === process.env.SLI_SIGNER));
const isLetter = (f: string) => signOf(f) >= 31 && signOf(f) <= 69;
const wordFiles = files.filter((f) => !isLetter(f));
const letterFiles = files.filter(isLetter);
const pick = (from: string[]) => from[Math.floor(rand() * from.length)];

function sentence(): string[] {
  if (MIX === 'mixed') return [pick(wordFiles), pick(letterFiles), pick(letterFiles), pick(wordFiles)];
  if (MIX === 'letters') return [pick(letterFiles), pick(letterFiles), pick(letterFiles)];
  return [pick(files), pick(files), pick(files)];
}

function handsDown(raw: RawFrame): RawFrame {
  return { ...raw, hands: [] };
}

function trimHandless(v: RawFrame[]): RawFrame[] {
  const first = v.findIndex((r) => r.hands.length);
  let last = v.length - 1;
  while (last > 0 && !v[last].hands.length) last--;
  return first < 0 ? v : v.slice(first, last + 1);
}

async function trial(session: ort.InferenceSession, fps: number) {
  const picks = sentence();
  const videos = picks.map((f) => JSON.parse(readFileSync(CACHE + f, 'utf8')) as RawFrame[]);
  const rest = handsDown(videos[0].find((r) => r.pose) ?? videos[0][0]);
  const stream: RawFrame[] = [];
  const ends: number[] = []; // frame index where each sign ends
  const gap = () => stream.push(...Array.from({ length: 30 }, () => rest));
  gap();
  for (const v of videos) {
    stream.push(...(CONTINUOUS ? trimHandless(v) : v));
    ends.push(stream.length);
    if (!CONTINUOUS) gap();
  }
  if (CONTINUOUS) gap();
  const commits: number[] = [];
  const commitAt: number[] = [];
  const interpreter = new Interpreter(
    async (seq) => {
      const out = await session.run({ input: new ort.Tensor('float32', seq, [1, SEQ_LEN, FRAME_SIZE]) });
      return softmaxTopK(out.output.data as Float32Array, 5);
    },
    letters,
    {
      onCommit: (c) => {
        commits.push(c.id);
        commitAt.push(c.at);
        if (process.env.SLI_TRACE) console.log('COMMIT', Math.round(c.at), c.id, c.p.toFixed(2));
      },
      onLive: (g, at) => {
        if (process.env.SLI_TRACE) console.log('live', Math.round(at), g.slice(0, 3).map((x) => `${x.id}:${x.p.toFixed(2)}`).join(' '));
      },
    },
    MODE,
    {
      ...DEFAULT_INTERPRETER,
      ...JSON.parse(process.env.SLI_IOPTS ?? '{}'),
      decoder: { ...DEFAULT_OPTIONS, ...JSON.parse(process.env.SLI_OPTS ?? '{}') },
      speller: { ...DEFAULT_SPELLER, ...JSON.parse(process.env.SLI_SOPTS ?? '{}') },
    },
  );
  // Source is 30 fps; a device at `fps` sees every (30/fps)-th frame.
  let last = -1;
  let seen = 0;
  let body: Pick<RawFrame, 'pose' | 'face'> = stream[0];
  for (let i = 0; i < stream.length; i++) {
    const tick = Math.floor((i * fps) / 30);
    if (tick === last) continue;
    last = tick;
    if (seen++ % BODY_EVERY === 0) body = stream[i];
    const raw = { ...stream[i], pose: body.pose, face: body.face };
    await interpreter.push((i * 1000) / 30, raw);
  }
  if (process.env.SLI_TRACE) console.log('TRUTH', picks.map(signOf).join(','), 'ends(ms)', ends.map((e) => Math.round((e * 1000) / 30)).join(','));
  // Latency: how long after a sign's last frame its word appeared (right words only).
  const truth = picks.map(signOf);
  const delays = truth.flatMap((id, k) => (commits[k] === id ? [commitAt[k] - (ends[k] * 1000) / 30] : []));
  return { truth, commits, delays };
}

/** Longest common subsequence: words recognised in the right order, even if one was missed or added. */
function lcs(a: number[], b: number[]): number {
  const dp = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++) dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
  return dp[a.length][b.length];
}

describe.runIf(process.env.SLI_REPLAY)('sentence replay', () => {
  for (const fps of (process.env.SLI_FPS ?? '10,15,30').split(',').map(Number)) {
    it(`${fps} fps`, async () => {
      const session = await ort.InferenceSession.create(new URL('../../public/models/karsl502.onnx', import.meta.url).pathname);
      let exact = 0;
      let wordsRight = 0;
      let extra = 0;
      const delays: number[] = [];
      for (let k = 0; k < TRIALS; k++) {
        const { truth, commits, delays: d } = await trial(session, fps);
        delays.push(...d);
        if (commits.join() === truth.join()) exact++;
        wordsRight += lcs(truth, commits);
        extra += Math.max(0, commits.length - truth.length);
        if (process.env.SLI_REPLAY_VERBOSE && commits.join() !== truth.join()) console.log(fps, 'truth', truth, 'got', commits);
      }
      delays.sort((a, b) => a - b);
      const summary = {
        fps,
        continuous: CONTINUOUS,
        sentences: TRIALS,
        exact: exact / TRIALS,
        mode: MODE,
        mix: MIX,
        words: wordsRight / (TRIALS * (MIX === 'mixed' ? 4 : 3)),
        extraPerSentence: extra / TRIALS,
        medianDelayMs: Math.round(delays[Math.floor(delays.length / 2)] ?? NaN),
      };
      console.log(JSON.stringify(summary));
      if (MODE === 'words' && MIX === 'any' && !CONTINUOUS) expect(summary.words).toBeGreaterThan(0.85);
    }, 600_000);
  }
});
