// Live fingerspelling. The letter model (training/train_letters_seq.py) reads the hand shape
// in the newest frame plus the wrist's path over the last few frames, so letters show up while
// the hand is still up. It can be fine-tuned on the device from one recording of each letter
// ("teach the app your hand"): on a signer it never saw, that took letters from 77% to 86%.
// normalise/windowFrom/featurise must match normalise/windows_of/featurise in the Python.

import type { Point } from './features';
import { LetterNet, type LayerFile, type Rand } from './letternet';

export interface LetterModelFile {
  classes: number[]; // sign ids, in output order
  mirror: string; // handedness label mirrored onto the other hand
  window: number; // frames per window (at ~15 fps)
  shapeLast: boolean;
  layers: LayerFile[];
}

export interface HandFrame {
  label: string;
  lms: Point[];
}

/** Model input before features: the newest hand shape and the wrist's path. */
export interface WindowSample {
  q: Float32Array; // 21 × 3, newest frame, wrist-relative, in palm lengths, mirrored
  motion: Float32Array; // window × 2, wrist relative to the newest wrist, in palm lengths
}

const TIPS = [0, 4, 8, 12, 16, 20];
const PAIRS = TIPS.flatMap((a, i) => TIPS.slice(i + 1).map((b) => [a, b] as const));

function normalise(f: HandFrame, mirror: string) {
  const [w, m] = [f.lms[0], f.lms[9]];
  const palm = Math.max(Math.hypot(m[0] - w[0], m[1] - w[1]), 1e-6);
  const flip = f.label === mirror;
  const q = new Float32Array(63);
  f.lms.forEach((p, i) => {
    q[i * 3] = ((p[0] - w[0]) / palm) * (flip ? -1 : 1);
    q[i * 3 + 1] = (p[1] - w[1]) / palm;
    q[i * 3 + 2] = (p[2] - w[2]) / palm;
  });
  return { q, wx: w[0], wy: w[1], palm, flip };
}

/**
 * The window ending at the newest of `frames` (oldest first, null = no hand), or null when the
 * newest frame has no hand. A missed detection repeats the previous frame; a window shorter than
 * `size` (the stream just started) is padded with its first frame.
 */
export function windowFrom(frames: (HandFrame | null)[], size: number, mirror: string): WindowSample | null {
  const end = frames[frames.length - 1];
  if (!end) return null;
  const win = frames.slice(-size);
  const first = win.find((f) => f) as HandFrame;
  const filled: HandFrame[] = [];
  let last: HandFrame | null = null;
  for (const f of win) filled.push((last = f ?? last ?? first));
  while (filled.length < size) filled.unshift(filled[0]);
  const e = normalise(end, mirror);
  const motion = new Float32Array(size * 2);
  filled.forEach((f, i) => {
    motion[i * 2] = ((f.lms[0][0] - e.wx) / e.palm) * (e.flip ? -1 : 1);
    motion[i * 2 + 1] = (f.lms[0][1] - e.wy) / e.palm;
  });
  return { q: e.q, motion };
}

/** Hand shape (63), fingertip distances (15), wrist path (window × 2). */
export function featurise(s: WindowSample): Float32Array {
  const out = new Float32Array(63 + PAIRS.length + s.motion.length);
  out.set(s.q);
  const q = s.q;
  PAIRS.forEach(([a, b], k) => {
    out[63 + k] = Math.hypot(q[a * 3] - q[b * 3], q[a * 3 + 1] - q[b * 3 + 1], q[a * 3 + 2] - q[b * 3 + 2]);
  });
  out.set(s.motion, 63 + PAIRS.length);
  return out;
}

/** Same augmentation as the Python: posture rotation/stretch, jitter, motion size. */
export function augment(s: WindowSample, rand: Rand): WindowSample {
  const th = rand.uniform(-0.3, 0.3);
  const [c, sn] = [Math.cos(th), Math.sin(th)];
  const scale = [rand.uniform(0.85, 1.15), rand.uniform(0.85, 1.15), rand.uniform(0.85, 1.15)];
  const q = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    const [x, y, z] = [s.q[i * 3], s.q[i * 3 + 1], s.q[i * 3 + 2]];
    q[i * 3] = (c * x - sn * y) * scale[0] + rand.normal(0.03);
    q[i * 3 + 1] = (sn * x + c * y) * scale[1] + rand.normal(0.03);
    q[i * 3 + 2] = z * scale[2] + rand.normal(0.03);
  }
  const k = rand.uniform(0.7, 1.3);
  const motion = new Float32Array(s.motion.length);
  for (let i = 0; i < motion.length; i += 2) {
    const [x, y] = [s.motion[i], s.motion[i + 1]];
    motion[i] = (c * x - sn * y) * k + rand.normal(0.05);
    motion[i + 1] = (sn * x + c * y) * k + rand.normal(0.05);
  }
  return { q, motion };
}

/**
 * Training windows from one recording, like windows_of() in the Python: windows ending in the
 * `span` part of the hand frames, at a frame stride, each with at least size - 2 hand frames.
 */
export function recordingWindows(
  frames: (HandFrame | null)[],
  size: number,
  mirror: string,
  stride = 1,
  offset = 0,
  span: [number, number] = [0.25, 1],
): WindowSample[] {
  const fr = frames.filter((_, i) => i >= offset && (i - offset) % stride === 0);
  const hands = fr.flatMap((f, i) => (f ? [i] : []));
  const n = hands.length;
  const ends = hands.slice(Math.floor(n * span[0]), Math.max(Math.floor(n * span[1]), Math.floor(n * span[0]) + 1));
  const out: WindowSample[] = [];
  for (const end of ends) {
    if (end - size + 1 < 0) continue;
    const win = fr.slice(end - size + 1, end + 1);
    if (win.filter((f) => f).length < Math.max(1, size - 2)) continue;
    out.push(windowFrom(win, size, mirror)!);
  }
  return out;
}

const PERSONAL_KEY = 'sli-letters-personal-v1';

export class LetterClassifier {
  private net: LetterNet;
  readonly base: LetterNet;

  constructor(readonly model: LetterModelFile) {
    this.base = LetterNet.fromFile(model.layers);
    this.net = this.base;
    const saved = loadPersonal(model);
    if (saved) this.net = saved;
  }

  static async load(base: string): Promise<LetterClassifier> {
    const res = await fetch(`${base}models/letters-seq.json`);
    if (!res.ok) throw new Error(`letters model: HTTP ${res.status}`);
    return new LetterClassifier(await res.json());
  }

  get classes() {
    return this.model.classes;
  }

  get personal() {
    return this.net !== this.base;
  }

  predict(s: WindowSample): Float32Array {
    return this.net.predict(featurise(s));
  }

  /** A per-camera stream: push each frame's signing hand (or null), get probabilities. */
  stream(): LetterStream {
    return new LetterStream(this);
  }

  /** Use (and remember on this device) a model fine-tuned to the user's hand. */
  setPersonal(net: LetterNet) {
    this.net = net;
    try {
      localStorage.setItem(PERSONAL_KEY, JSON.stringify({ id: modelId(this.model), layers: net.toFile() }));
    } catch {
      /* storage full or blocked: it still applies until the page closes */
    }
  }

  resetPersonal() {
    this.net = this.base;
    try {
      localStorage.removeItem(PERSONAL_KEY);
    } catch {
      /* ignore */
    }
  }
}

export class LetterStream {
  private frames: (HandFrame | null)[] = [];

  constructor(private clf: LetterClassifier) {}

  push(hand: HandFrame | null): Float32Array | null {
    this.frames.push(hand);
    if (this.frames.length > this.clf.model.window) this.frames.shift();
    const s = windowFrom(this.frames, this.clf.model.window, this.clf.model.mirror);
    return s && this.clf.predict(s);
  }

  reset() {
    this.frames = [];
  }
}

/** Personal weights are only valid for the base model they were tuned from. */
function modelId(m: LetterModelFile) {
  const l = m.layers[m.layers.length - 1];
  return `${m.classes.join(',')}|${m.layers.map((x) => x.rows).join('x')}|${l.b.slice(0, 4).join(',')}`;
}

function loadPersonal(model: LetterModelFile): LetterNet | null {
  try {
    const raw = localStorage.getItem(PERSONAL_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw) as { id: string; layers: LayerFile[] };
    return saved.id === modelId(model) ? LetterNet.fromFile(saved.layers) : null;
  } catch {
    return null;
  }
}
