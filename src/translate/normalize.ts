const DIACRITICS = /[ؐ-ًؚ-ٰٟۖ-ۭـ]/g; // tashkeel + tatweel
const ARABIC_INDIC = '٠١٢٣٤٥٦٧٨٩';
const PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹';

/** Western digits for Arabic-Indic / Persian digits. */
export function asciiDigits(s: string): string {
  return s.replace(/[٠-٩۰-۹]/g, (d) => {
    const i = ARABIC_INDIC.indexOf(d);
    return String(i >= 0 ? i : PERSIAN_DIGITS.indexOf(d));
  });
}

/** Strip marks only; keeps letter identity (used for fingerspelling). */
export function stripMarks(s: string): string {
  return asciiDigits(s).replace(DIACRITICS, '');
}

/** Loose form used for dictionary lookup: hamza/alef/taa-marbuta/yaa variants folded. */
export function normalize(s: string): string {
  return stripMarks(s)
    .toLowerCase()
    .replace(/[أإآٱ]/g, 'ا')
    .replace(/ة/g, 'ه')
    .replace(/ى/g, 'ي')
    .replace(/ؤ/g, 'و')
    .replace(/ئ/g, 'ي')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Words of the input, keeping their original spelling (marks removed). */
export function tokenize(s: string): string[] {
  return stripMarks(s)
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);
}
