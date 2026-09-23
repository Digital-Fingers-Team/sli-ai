import raw from './signs.json';

export type Category =
  | 'numbers' | 'letters' | 'health' | 'verbs' | 'family' | 'traits'
  | 'places' | 'social' | 'home' | 'religion' | 'jobs';

export interface Sign {
  id: number; // model class index
  ar: string; // display word
  label: string; // full KArSL label with variants
  en: string;
  cat: Category;
  aliases: string[];
}

export const SIGNS = raw as Sign[];
export const CATEGORIES: Category[] = [
  'social', 'family', 'verbs', 'traits', 'places', 'home', 'health', 'jobs', 'religion', 'numbers', 'letters',
];

export const signById = (id: number): Sign => SIGNS[id];

/** Letter sign ids keyed by the exact written form (أ and ا are different signs). */
export const LETTER_IDS = new Map(SIGNS.filter((s) => s.cat === 'letters').map((s) => [s.ar, s.id]));
/** Number sign ids keyed by value. */
export const NUMBER_IDS = new Map(SIGNS.filter((s) => s.cat === 'numbers').map((s) => [Number(s.ar), s.id]));

export function clipUrl(id: number): string {
  return `${import.meta.env.BASE_URL}clips/${String(id + 1).padStart(4, '0')}.mp4`;
}

export function posterUrl(id: number): string {
  return `${import.meta.env.BASE_URL}clips/${String(id + 1).padStart(4, '0')}.jpg`;
}
