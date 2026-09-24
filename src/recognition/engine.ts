// Camera -> landmarks (worker) -> Interpreter (words, letters or both). One engine per page
// that signs; the models behind it are loaded once and shared.
//
// The main thread only grabs frames and draws. MediaPipe runs in one worker and the word model
// in another, so neither the camera view nor the landmark rate waits on a classification.

import type { RawFrame } from './features';
import { DEFAULT_OPTIONS, type DecoderEvents, type DecoderOptions } from './decoder';
import { DEFAULT_SPELLER, type SpellerOptions } from './speller';
import { DEFAULT_INTERPRETER, Interpreter, type InterpreterOptions, type SignMode } from './interpreter';
import { LetterClassifier } from './letters';
import type { Guess } from './topk';
import type { LandmarksReply, LandmarksRequest } from './landmarks.worker';
import type { ClassifierReply, ClassifierRequest } from './classifier.worker';

export interface EngineEvents extends DecoderEvents {
  onFrame?(raw: RawFrame, fps: number): void;
  onStatus?(status: EngineStatus, detail?: string): void;
}

export type EngineStatus = 'loading' | 'ready' | 'running' | 'stopped' | 'error';
export type { SignMode } from './interpreter';

const params = new URLSearchParams(location.search);

// ?debug keeps every raw frame on window.__sliFrames (used by the end-to-end probes).
const debugFrames: { t: number; raw: RawFrame; ms: number }[] | null = params.has('debug') ? [] : null;
if (debugFrames) Object.assign(window, { __sliFrames: debugFrames });

// ?timescale=N stretches every decoder timing N times. Tests use it with a camera video slowed
// down N times to emulate a fast device on a slow test machine.
const timescale = Number(params.get('timescale')) || 1;

function decoderOptions(): DecoderOptions {
  const o = DEFAULT_OPTIONS;
  const k = timescale;
  return {
    ...o,
    stepMs: o.stepMs * k,
    windowsMs: o.windowsMs.map((w) => w * k),
    gapMs: o.gapMs * k,
    maxBufferMs: o.maxBufferMs * k,
    tailMinMs: o.tailMinMs * k,
    minSignMs: o.minSignMs * k,
    preRollMs: o.preRollMs * k,
    postRollMs: o.postRollMs * k,
  };
}

function spellerOptions(): SpellerOptions {
  const o = DEFAULT_SPELLER;
  return { ...o, holdMs: o.holdMs * timescale, gapMs: o.gapMs * timescale };
}

function interpreterOptions(): InterpreterOptions {
  const o = DEFAULT_INTERPRETER;
  return { ...o, decoder: decoderOptions(), speller: spellerOptions(), stillSpeed: o.stillSpeed / timescale };
}

// Pose and face move much less than the hands. On a device that cannot keep up (below
// SLOW_FPS) they are refreshed every other frame and reused in between: in the sentence replay
// that costs 2-3 points of word accuracy, less than losing frames does. Skipping them raises
// the frame rate, so going back to every frame needs a clear margin (FAST_FPS).
// ?bodyEvery=N forces a fixed rate.
const SLOW_FPS = 18;
const FAST_FPS = 26;
const BODY_EVERY = Number(params.get('bodyEvery')) || 0;

interface Models {
  landmarks: Worker;
  classify: (seq: Float32Array) => Promise<Guess[]>;
  letters: LetterClassifier;
}

let shared: Promise<Models> | null = null;
let progress: ((step: string) => void) | undefined;

function startLandmarks(base: string): Promise<Worker> {
  const worker = new Worker(new URL('./landmarks.worker.ts', import.meta.url), { type: 'module' });
  const delegate = params.get('delegate')?.toUpperCase();
  const request: LandmarksRequest = {
    type: 'init',
    options: {
      base,
      delegate: delegate === 'CPU' || delegate === 'GPU' ? delegate : undefined,
      // ?handsMode=video / ?body=image override the per-part modes, ?hands=1 tracks one hand.
      handsMode: params.get('handsMode') === 'video' ? 'VIDEO' : 'IMAGE',
      bodyMode: params.get('body') === 'image' ? 'IMAGE' : 'VIDEO',
      numHands: params.get('hands') === '1' ? 1 : 2,
    },
  };
  worker.postMessage(request);
  return new Promise((resolve, reject) => {
    const onMessage = (e: MessageEvent<LandmarksReply>) => {
      const m = e.data;
      if (m.type === 'progress') progress?.(m.step);
      else if (m.type === 'ready') {
        worker.removeEventListener('message', onMessage);
        resolve(worker);
      } else if (m.type === 'error') {
        worker.removeEventListener('message', onMessage);
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.addEventListener('message', onMessage);
  });
}

function startClassifier(base: string): Promise<(seq: Float32Array) => Promise<Guess[]>> {
  const worker = new Worker(new URL('./classifier.worker.ts', import.meta.url), { type: 'module' });
  const pending = new Map<number, { resolve: (g: Guess[]) => void; reject: (e: Error) => void }>();
  let nextId = 0;
  const classify = (seq: Float32Array) =>
    new Promise<Guess[]>((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      const request: ClassifierRequest = { type: 'classify', id, seq };
      worker.postMessage(request, [seq.buffer]);
    });
  return new Promise((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<ClassifierReply>) => {
      const m = e.data;
      if (m.type === 'ready') resolve(classify);
      else if (m.type === 'guesses') {
        pending.get(m.id)?.resolve(m.guesses);
        pending.delete(m.id);
      } else if (m.id !== undefined) {
        pending.get(m.id)?.reject(new Error(m.message));
        pending.delete(m.id);
      } else reject(new Error(m.message));
    };
    const request: ClassifierRequest = { type: 'init', base };
    worker.postMessage(request);
  });
}

/** Loads models once per page load; later callers reuse them. */
export function loadModels(onProgress?: (step: string) => void) {
  progress = onProgress;
  // Workers resolve relative URLs against their own script, so hand them an absolute base.
  const base = new URL(import.meta.env.BASE_URL, location.href).href;
  shared ??= Promise.all([startLandmarks(base), startClassifier(base), LetterClassifier.load(base)]).then(
    ([landmarks, classify, letters]) => ({ landmarks, classify, letters }),
  );
  shared.catch(() => (shared = null));
  return shared;
}

export class Engine {
  private stream: MediaStream | null = null;
  private running = false;
  private models: Models | null = null;
  private interpreter: Interpreter | null = null;
  private inflight = false;
  private frameWaiting = false;
  private frameNo = 0;
  private bodyEvery = 1;
  private lastResultAt = -1;
  private fps = 0;
  private mode: SignMode = 'auto';

  constructor(
    private video: HTMLVideoElement,
    private events: EngineEvents,
  ) {}

  setMode(mode: SignMode) {
    this.mode = mode;
    this.frameNo = 0;
    this.interpreter?.setMode(mode);
  }

  async start() {
    this.events.onStatus?.('loading');
    try {
      const [models, stream] = await Promise.all([
        loadModels((s) => this.events.onStatus?.('loading', s)),
        navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        }),
      ]);
      this.models = models;
      this.stream = stream;
      this.video.srcObject = stream;
      this.video.muted = true;
      this.video.playsInline = true;
      await this.video.play();
      this.interpreter = new Interpreter(models.classify, models.letters, this.events, this.mode, interpreterOptions());
      models.landmarks.addEventListener('message', this.onResult);
      this.running = true;
      this.inflight = false;
      this.events.onStatus?.('running');
      this.watchFrames();
    } catch (err) {
      this.stop();
      this.events.onStatus?.('error', err instanceof Error ? err.name || err.message : String(err));
    }
  }

  /** Marks each new camera frame; it is sent as soon as the worker is free. */
  private watchFrames() {
    const onFrame = () => {
      if (!this.running) return;
      this.frameWaiting = true;
      this.send();
      if ('requestVideoFrameCallback' in this.video) this.video.requestVideoFrameCallback(onFrame);
      else requestAnimationFrame(onFrame);
    };
    onFrame();
  }

  private send() {
    if (!this.running || this.inflight || !this.frameWaiting || this.video.readyState < 2 || !this.models) return;
    this.inflight = true;
    this.frameWaiting = false;
    const t = performance.now();
    const worker = this.models.landmarks;
    createImageBitmap(this.video).then(
      (bitmap) => {
        if (!this.running) {
          bitmap.close();
          this.inflight = false;
          return;
        }
        // Letters are hand shapes only; words need the body and face too.
        if (this.fps > 0 && this.fps < SLOW_FPS) this.bodyEvery = 2;
        else if (this.fps > FAST_FPS) this.bodyEvery = 1;
        const needsBody = this.interpreter?.needsBody ?? this.mode !== 'letters';
        const body = needsBody && this.frameNo++ % (BODY_EVERY || this.bodyEvery) === 0;
        const request: LandmarksRequest = { type: 'frame', bitmap, t, body, face: body };
        worker.postMessage(request, [bitmap]);
      },
      () => (this.inflight = false),
    );
  }

  private onResult = (e: MessageEvent<LandmarksReply>) => {
    const m = e.data;
    if (m.type === 'error') {
      console.error('landmarks:', m.message);
      this.inflight = false;
      return;
    }
    if (m.type !== 'result' || !this.running) return;
    this.inflight = false;
    const now = performance.now();
    if (this.lastResultAt > 0) this.fps = this.fps * 0.85 + (1000 / (now - this.lastResultAt)) * 0.15;
    this.lastResultAt = now;
    this.send();

    const { raw, t } = m;
    if (debugFrames) debugFrames.push({ t, raw, ms: m.ms });
    this.events.onFrame?.(raw, this.fps);
    // Not awaited: the decoder skips steps while a classification is still running.
    void this.interpreter?.push(t, raw);
  };

  resetSentence() {
    this.interpreter?.reset();
  }

  stop() {
    this.running = false;
    this.models?.landmarks.removeEventListener('message', this.onResult);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.lastResultAt = -1;
    this.fps = 0;
    this.events.onStatus?.('stopped');
  }
}
