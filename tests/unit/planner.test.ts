import { describe, expect, it } from 'vitest';
import { numberSigns, plan, sequence, spell } from '../../src/translate/planner';
import { SIGNS } from '../../src/data/signs';

const idOf = (ar: string) => SIGNS.find((s) => s.ar === ar)!.id;
const words = (text: string) => plan(text).map((p) => (p.kind === 'word' ? SIGNS[p.id].ar : `${p.kind}:${p.text}`));

describe('plan', () => {
  it('matches multi-word phrases before single words', () => {
    expect(words('السلام عليكم')).toEqual(['السلام عليكم']);
    expect(words('صلاة الفجر')).toEqual(['صلاة الفجر']);
  });

  it('ignores diacritics and hamza/taa-marbuta spelling differences', () => {
    expect(words('شُكْرًا')).toEqual(['شكراً']);
    expect(words('مستشفي')).toEqual(['مستشفى']);
    expect(words('اب')).toEqual(['أب']);
  });

  it('matches aliases packed into KArSL labels', () => {
    expect(words('ضمادة')).toEqual(['شاش']);
    expect(words('كوب')).toEqual(['كأس']);
  });

  it('strips common prefixes and suffixes', () => {
    expect(words('والطبيب')).toEqual(['طبيب']);
    expect(words('بيتي')).toEqual(['بيت']);
    expect(words('للمسجد')).toEqual(['مسجد']);
  });

  it('builds a sentence', () => {
    expect(words('أهلا يا صديق، أمي في المستشفى')).toEqual([
      'أهلا وسهلاً', 'spell:يا', 'صديق', 'أم', 'spell:في', 'مستشفى',
    ]);
  });

  it('fingerspells unknown words with exact letter signs', () => {
    const [item] = plan('أحمد');
    expect(item.kind).toBe('spell');
    if (item.kind !== 'spell') return;
    expect(item.ids).toEqual(['أ', 'ح', 'م', 'د'].map(idOf));
    expect(item.missing).toEqual([]);
  });

  it('uses the ال and لا signs', () => {
    expect(spell('الاسلام').ids).toEqual(['ال', 'ا', 'س', 'لا', 'م'].map(idOf));
  });

  it('accepts English sign names', () => {
    expect(words('doctor')).toEqual(['طبيب']);
  });
});

describe('numbers', () => {
  const vals = (n: number) => numberSigns(n)!.map((id) => SIGNS[id].ar);
  it('uses direct signs where they exist', () => {
    expect(vals(7)).toEqual(['7']);
    expect(vals(40)).toEqual(['40']);
    expect(vals(1000000)).toEqual(['1000000']);
  });
  it('composes other numbers', () => {
    expect(vals(25)).toEqual(['20', '5']);
    expect(vals(305)).toEqual(['300', '5']);
    expect(vals(2026)).toEqual(['2', '1000', '20', '6']);
    expect(vals(1000)).toEqual(['1000']);
  });
  it('reads Arabic-Indic digits', () => {
    const [item] = plan('١٥');
    expect(item.kind).toBe('number');
    expect(sequence([item]).map((s) => SIGNS[s.id].ar)).toEqual(['10', '5']);
  });
});
