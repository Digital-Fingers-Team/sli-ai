import { expect, test } from '@playwright/test';

// The letters camera fingerspells SLI_LETTERS (default ب then ل), each letter framed by empty
// background, slowed down SLI_TIMESCALE times like the words camera (see app.spec.ts).
const expected = (process.env.SLI_LETTERS ?? 'ب,ل').split(',');
const timescale = process.env.SLI_TIMESCALE ?? '4';

for (const mode of ['حروف', 'تلقائي']) test(`${mode}: fingerspelled letters are typed live`, async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !m.text().startsWith('INFO:') && errors.push(m.text()));

  await page.goto(`/?delegate=CPU&timescale=${timescale}#/sign`);
  await page.getByRole('button', { name: mode }).click();
  await expect(page.getByRole('button', { name: mode })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'تشغيل الكاميرا' }).click();
  await expect(page.getByRole('button', { name: 'إيقاف الكاميرا' })).toBeVisible({ timeout: 120_000 });

  // A letter must show up while the hand is still up (the pending preview or a typed letter).
  await expect(page.locator('.sentence-words .word.letter').first()).toBeVisible({ timeout: 120_000 });
  const words = page.locator('.sentence-words .word.letter:not(.pending)');
  await expect(words).toHaveCount(expected.length, { timeout: 180_000 });
  const got = await words.allTextContents();
  const alts = await words.evaluateAll((els) => els.map((e) => (e as HTMLElement).dataset.alts?.split('|') ?? []));
  console.log('typed:', got.join(' | '), '— alternatives:', JSON.stringify(alts));
  const rotations = expected.map((_, i) => [...expected.slice(i), ...expected.slice(0, i)]);
  expect(rotations.some((r) => r.every((w, k) => got[k] === w || alts[k].includes(w)))).toBe(true);
  expect(errors).toEqual([]);
});
