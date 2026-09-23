// Copies the MediaPipe and ONNX Runtime wasm files into public/runtime so the app works offline
// and never depends on a CDN.
import { cpSync, mkdirSync, readdirSync } from 'node:fs';

const copy = (from, to, keep) => {
  mkdirSync(to, { recursive: true });
  for (const f of readdirSync(from)) if (keep(f)) cpSync(`${from}/${f}`, `${to}/${f}`);
};

copy('node_modules/@mediapipe/tasks-vision/wasm', 'public/runtime/mediapipe', () => true);
console.log('runtime assets copied');
