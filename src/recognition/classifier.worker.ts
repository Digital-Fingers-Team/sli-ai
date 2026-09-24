/// <reference lib="webworker" />
// The word model runs in its own worker so a classification never delays the next camera frame.

import { SignClassifier } from './classifier';
import type { Guess } from './topk';

export type ClassifierRequest = { type: 'init'; base: string } | { type: 'classify'; id: number; seq: Float32Array };

export type ClassifierReply =
  | { type: 'ready' }
  | { type: 'error'; message: string; id?: number }
  | { type: 'guesses'; id: number; guesses: Guess[]; ms: number };

let classifier: SignClassifier | null = null;
const post = (m: ClassifierReply) => self.postMessage(m);

self.onmessage = async (e: MessageEvent<ClassifierRequest>) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      classifier = await SignClassifier.load(m.base);
      post({ type: 'ready' });
    } else {
      const start = performance.now();
      const guesses = await classifier!.classify(m.seq);
      post({ type: 'guesses', id: m.id, guesses, ms: performance.now() - start });
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err), id: m.type === 'classify' ? m.id : undefined });
  }
};
