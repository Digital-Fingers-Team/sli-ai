// Camera -> landmarks -> features -> decoder. One engine is shared by the pages that sign.

import { frameFeatures, type RawFrame } from './features';
import { Landmarks } from './landmarks';
import { SignClassifier } from './classifier';
import { Decoder, DEFAULT_OPTIONS, type DecoderEvents, type DecoderOptions } from './decoder';

export interface EngineEvents extends DecoderEvents {
  onFrame?(raw: RawFrame, fps: number): void;
  onStatus?(status: EngineStatus, detail?: string): void;
}

export type EngineStatus = 'loading' | 'ready' | 'running' | 'stopped' | 'error';

// ?debug keeps every raw frame on window.__sliFrames (used by the end-to-end probes).
const debugFrames: { t: number; raw: RawFrame }[] | null = new URLSearchParams(location.search).has('debug') ? [] : null;
if (debugFrames) Object.assign(window, { __sliFrames: debugFrames, __sliLoadModels: () => loadModels() });

// ?timescale=N stretches every decoder timing N times. Tests use it with a camera video slowed
// down N times to emulate a fast device on a slow test machine.
function decoderOptions(): DecoderOptions {
  const k = Number(new URLSearchParams(location.search).get('timescale')) || 1;
  const o = DEFAULT_OPTIONS;
  return {
    ...o,
    stepMs: o.stepMs * k,
    windowsMs: o.windowsMs.map((w) => w * k),
    gapMs: o.gapMs * k,
    maxBufferMs: o.maxBufferMs * k,
    tailMinMs: o.tailMinMs * k,
    preRollMs: o.preRollMs * k,
    postRollMs: o.postRollMs * k,
  };
}

let shared: Promise<{ landmarks: Landmarks; classifier: SignClassifier }> | null = null;

/** Loads models once per page load; later callers reuse them. */
export function loadModels(onProgress?: (step: string) => void) {
  const base = import.meta.env.BASE_URL;
  shared ??= Promise.all([Landmarks.load(base, onProgress), SignClassifier.load(base)]).then(
    ([landmarks, classifier]) => ({ landmarks, classifier }),
  );
  shared.catch(() => (shared = null));
  return shared;
}

export class Engine {
  private stream: MediaStream | null = null;
  private running = false;
  private decoder: Decoder | null = null;
  private lastTs = -1;
  private fps = 0;

  constructor(
    private video: HTMLVideoElement,
    private events: EngineEvents,
  ) {}

  async start() {
    this.events.onStatus?.('loading');
    try {
      const [{ landmarks, classifier }, stream] = await Promise.all([
        loadModels((s) => this.events.onStatus?.('loading', s)),
        navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        }),
      ]);
      this.stream = stream;
      this.video.srcObject = stream;
      this.video.muted = true;
      this.video.playsInline = true;
      await this.video.play();
      this.decoder = new Decoder((seq) => classifier.classify(seq), this.events, decoderOptions());
      this.running = true;
      this.events.onStatus?.('running');
      this.loop(landmarks);
    } catch (err) {
      this.stop();
      this.events.onStatus?.('error', err instanceof Error ? err.name || err.message : String(err));
    }
  }

  private loop(landmarks: Landmarks) {
    const tick = async () => {
      if (!this.running) return;
      const now = performance.now();
      if (this.video.readyState >= 2 && now > this.lastTs) {
        const dt = this.lastTs > 0 ? now - this.lastTs : 0;
        this.lastTs = now;
        if (dt > 0) this.fps = this.fps * 0.9 + (1000 / dt) * 0.1;
        const raw = landmarks.detect(this.video, now);
        if (debugFrames) debugFrames.push({ t: now, raw });
        this.events.onFrame?.(raw, this.fps);
        // Frames are not awaited so the camera keeps flowing while the model runs.
        void this.decoder?.push(now, frameFeatures(raw), raw.hands.length > 0);
      }
      if ('requestVideoFrameCallback' in this.video) this.video.requestVideoFrameCallback(() => tick());
      else requestAnimationFrame(() => tick());
    };
    tick();
  }

  resetSentence() {
    this.decoder?.reset();
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.events.onStatus?.('stopped');
  }
}
