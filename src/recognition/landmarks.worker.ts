/// <reference lib="webworker" />
// Runs MediaPipe off the main thread, so the camera view and the page stay smooth whatever the
// device's speed. The main thread sends the next frame only after the previous result is back,
// so results never queue up behind stale frames.

import { Landmarks, type LandmarkOptions } from './landmarks';
import type { RawFrame } from './features';

export type LandmarksRequest =
  | { type: 'init'; options: LandmarkOptions }
  | { type: 'frame'; bitmap: ImageBitmap; t: number; body: boolean; face: boolean };

export type LandmarksReply =
  | { type: 'progress'; step: string }
  | { type: 'ready'; delegate: string }
  | { type: 'error'; message: string }
  | { type: 'result'; t: number; raw: RawFrame; ms: number };

// tasks-vision loads its wasm glue with importScripts, which module workers do not have, and
// then falls back to self.import. The glue is a classic script, so evaluate it as one.
Object.assign(self, {
  import: async (url: string) => {
    const code = await (await fetch(url)).text();
    (0, eval)(`${code}\n;self.ModuleFactory = ModuleFactory;`);
  },
});

let landmarks: Landmarks | null = null;
const post = (m: LandmarksReply) => self.postMessage(m);

self.onmessage = async (e: MessageEvent<LandmarksRequest>) => {
  const m = e.data;
  if (m.type === 'init') {
    try {
      landmarks = await Landmarks.load(m.options, (step) => post({ type: 'progress', step }));
      post({ type: 'ready', delegate: landmarks.delegate });
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  if (!landmarks) return m.bitmap.close();
  const start = performance.now();
  try {
    const raw = landmarks.detect(m.bitmap, m.t, m.body, m.face);
    post({ type: 'result', t: m.t, raw, ms: performance.now() - start });
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  } finally {
    m.bitmap.close();
  }
};
