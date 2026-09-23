// Text -> a sequence of signs to play. Known words and phrases become word signs;
// numbers are composed from KArSL number signs; anything else is fingerspelled.

import { LETTER_IDS, NUMBER_IDS, SIGNS } from '../data/signs';
import { normalize, tokenize } from './normalize';

export type PlanItem =
  | { kind: 'word'; text: string; id: number }
  | { kind: 'number'; text: string; ids: number[] }
  | { kind: 'spell'; text: string; ids: number[]; missing: string[] };

const MAX_PHRASE = 5;
const PREFIXES = ['وبال', 'وال', 'بال', 'فال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ل'];
const SUFFIXES = ['هما', 'كما', 'كم', 'هم', 'هن', 'نا', 'ها', 'ات', 'ون', 'ين', 'ان', 'ه', 'ك', 'ي'];

// Normalised phrase -> sign id. Letters and numbers are not word entries.
const index = new Map<string, number>();
for (const s of SIGNS) {
  if (s.cat === 'letters' || s.cat === 'numbers') continue;
  for (const form of [s.ar, s.label, ...s.aliases, s.en]) {
    const key = normalize(form);
    if (key && !index.has(key)) index.set(key, s.id);
  }
}

function lookupWord(word: string): number | undefined {
  const key = normalize(word);
  const direct = index.get(key);
  if (direct !== undefined) return direct;
  // Light stemming: try removing one prefix and/or one suffix, keeping at least 2 letters.
  const stems = new Set<string>();
  for (const p of ['', ...PREFIXES]) {
    if (p && !key.startsWith(p)) continue;
    const a = key.slice(p.length);
    for (const s of ['', ...SUFFIXES]) {
      if (s && !a.endsWith(s)) continue;
      const stem = a.slice(0, a.length - s.length);
      if (stem.length >= 2 && (p || s)) stems.add(stem);
    }
  }
  for (const stem of [...stems].sort((x, y) => y.length - x.length)) {
    const id = index.get(stem) ?? index.get(stem + 'ه');
    if (id !== undefined) return id;
  }
  return undefined;
}

/** Signs for a whole number, or null if it is too large to compose. */
export function numberSigns(n: number): number[] | null {
  if (!Number.isInteger(n) || n < 0) return null;
  const direct = NUMBER_IDS.get(n);
  if (direct !== undefined) return [direct];
  if (n >= 100000) return null;
  const out: number[] = [];
  const thousands = Math.floor(n / 1000);
  const hundreds = Math.floor((n % 1000) / 100) * 100;
  const rest = n % 100;
  if (thousands) {
    if (thousands > 1) {
      const t = numberSigns(thousands);
      if (!t) return null;
      out.push(...t);
    }
    out.push(NUMBER_IDS.get(1000)!);
  }
  if (hundreds) out.push(NUMBER_IDS.get(hundreds)!);
  if (rest) {
    if (NUMBER_IDS.has(rest)) out.push(NUMBER_IDS.get(rest)!);
    else out.push(NUMBER_IDS.get(rest - (rest % 10))!, NUMBER_IDS.get(rest % 10)!);
  }
  return out;
}

function digitSigns(digits: string): number[] {
  return [...digits].map((d) => NUMBER_IDS.get(Number(d))!);
}

/** Fingerspelling with the KArSL letter signs, including the ال and لا signs. */
export function spell(word: string): { ids: number[]; missing: string[] } {
  const ids: number[] = [];
  const missing: string[] = [];
  let i = 0;
  if (word.startsWith('ال') && word.length > 2) {
    ids.push(LETTER_IDS.get('ال')!);
    i = 2;
  }
  while (i < word.length) {
    const two = word.slice(i, i + 2);
    if (/^ل[اأإآ]$/.test(two)) {
      ids.push(LETTER_IDS.get('لا')!);
      i += 2;
      continue;
    }
    const ch = word[i];
    const id = LETTER_IDS.get(ch) ?? (ch === 'ٱ' ? LETTER_IDS.get('ا') : undefined);
    if (id !== undefined) ids.push(id);
    else missing.push(ch);
    i++;
  }
  return { ids, missing };
}

export function plan(text: string): PlanItem[] {
  const words = tokenize(text);
  const out: PlanItem[] = [];
  let i = 0;
  while (i < words.length) {
    // Longest phrase first, e.g. "السلام عليكم", "صلاة الفجر".
    let matched = false;
    for (let len = Math.min(MAX_PHRASE, words.length - i); len >= 2; len--) {
      const phrase = words.slice(i, i + len).join(' ');
      const id = index.get(normalize(phrase));
      if (id !== undefined) {
        out.push({ kind: 'word', text: phrase, id });
        i += len;
        matched = true;
        break;
      }
    }
    if (matched) continue;

    const word = words[i++];
    if (/^\d+$/.test(word)) {
      const ids = numberSigns(Number(word)) ?? digitSigns(word);
      out.push({ kind: 'number', text: word, ids });
      continue;
    }
    const id = lookupWord(word);
    if (id !== undefined) {
      out.push({ kind: 'word', text: word, id });
      continue;
    }
    out.push({ kind: 'spell', text: word, ...spell(word) });
  }
  return out;
}

/** Flattened list of sign ids with the item each one belongs to. */
export function sequence(items: PlanItem[]): { id: number; item: number }[] {
  return items.flatMap((it, item) =>
    it.kind === 'word' ? [{ id: it.id, item }] : it.ids.map((id) => ({ id, item })),
  );
}
