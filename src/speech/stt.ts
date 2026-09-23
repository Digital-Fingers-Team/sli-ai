// Speech to text via the Web Speech API (Chrome, Edge, Safari, Android). Firefox has none.

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: any) => void) | null;
  onerror: ((e: any) => void) | null;
  onend: (() => void) | null;
}

const Ctor: (new () => SpeechRecognitionLike) | undefined =
  (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;

export const sttSupported = () => !!Ctor;

export interface ListenHandlers {
  onInterim?(text: string): void;
  onFinal(text: string): void;
  onError?(code: string): void;
  onEnd?(): void;
}

/** Starts listening; returns a stop function. */
export function listen(lang: 'ar' | 'en', h: ListenHandlers): () => void {
  if (!Ctor) {
    h.onError?.('unsupported');
    h.onEnd?.();
    return () => {};
  }
  const rec = new Ctor();
  rec.lang = lang === 'ar' ? 'ar-EG' : 'en-US';
  rec.continuous = true;
  rec.interimResults = true;
  rec.onresult = (e) => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) h.onFinal(r[0].transcript.trim());
      else interim += r[0].transcript;
    }
    if (interim) h.onInterim?.(interim.trim());
  };
  rec.onerror = (e) => h.onError?.(e.error ?? 'error');
  rec.onend = () => h.onEnd?.();
  rec.start();
  return () => rec.stop();
}
