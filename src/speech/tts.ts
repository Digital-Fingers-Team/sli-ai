// Text to speech with the browser's voices. Arabic voices differ a lot between devices,
// so we pick the best one available and report what we found.

export type Lang = 'ar' | 'en';

let voices: SpeechSynthesisVoice[] = [];
const ready = new Promise<void>((resolve) => {
  if (!('speechSynthesis' in window)) return resolve();
  const load = () => {
    voices = speechSynthesis.getVoices();
    if (voices.length) resolve();
  };
  load();
  speechSynthesis.addEventListener('voiceschanged', load);
  setTimeout(resolve, 1500);
});

export const ttsSupported = () => 'speechSynthesis' in window;

export async function pickVoice(lang: Lang): Promise<SpeechSynthesisVoice | null> {
  await ready;
  const matches = voices.filter((v) => v.lang.toLowerCase().startsWith(lang));
  const prefer = lang === 'ar' ? ['ar-eg', 'ar-sa', 'ar'] : ['en-us', 'en-gb', 'en'];
  for (const p of prefer) {
    const v = matches.find((m) => m.lang.toLowerCase().replace('_', '-').startsWith(p));
    if (v) return v;
  }
  return null;
}

export async function speak(text: string, lang: Lang = 'ar', rate = 1): Promise<void> {
  if (!ttsSupported() || !text.trim()) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang === 'ar' ? 'ar-EG' : 'en-US';
  const voice = await pickVoice(lang);
  if (voice) u.voice = voice;
  u.rate = rate;
  await new Promise<void>((resolve) => {
    u.onend = () => resolve();
    u.onerror = () => resolve();
    speechSynthesis.speak(u);
  });
}

export function stopSpeaking() {
  if (ttsSupported()) speechSynthesis.cancel();
}
