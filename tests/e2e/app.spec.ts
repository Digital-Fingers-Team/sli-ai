import { expect, test, type Page } from '@playwright/test';

// The fake camera plays KArSL signs given by SLI_EXPECT (comma-separated Arabic words),
// separated by empty background, slowed down SLI_TIMESCALE times (make_fake_camera.py --slow):
// the test machine runs MediaPipe far slower than a phone, so the video and the decoder's
// timings are stretched by the same factor.
const expected = (process.env.SLI_EXPECT ?? 'السلام عليكم,طبيب').split(',');
const timescale = process.env.SLI_TIMESCALE ?? '4';
// SLI_URL_EXTRA adds query parameters, e.g. hands=1, to compare pipeline settings.
const extra = process.env.SLI_URL_EXTRA ? `&${process.env.SLI_URL_EXTRA}` : '';

function collectErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  // MediaPipe logs routine INFO lines through console.error.
  page.on('console', (m) => m.type() === 'error' && !m.text().startsWith('INFO:') && errors.push(m.text()));
  return errors;
}

test('sign → text: recognises the signs from the camera', async ({ page }) => {
  const errors = collectErrors(page);

  // Test browsers have no GPU; emulated WebGL is far slower than the CPU path.
  await page.goto(`/?delegate=CPU&debug&timescale=${timescale}${extra}#/sign`);
  // The default mode, auto, must not turn still moments of words into letters.
  await expect(page.getByRole('button', { name: 'تلقائي' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'تشغيل الكاميرا' }).click();
  await expect(page.getByRole('button', { name: 'إيقاف الكاميرا' })).toBeVisible({ timeout: 120_000 });

  // The faint live preview (.pending) is not a recognised word yet.
  // The word model needs pose and face as well as hands; losing them once made every sign
  // come out as the same word.
  await expect
    .poll(() => page.evaluate(() => (window as unknown as { __sliFrames: { raw: { pose: unknown; face: unknown } }[] }).__sliFrames.filter((f) => f.raw.pose && f.raw.face).length), { timeout: 120_000 })
    .toBeGreaterThan(10);
  const words = page.locator('.sentence-words .word:not(.pending)');
  await expect(words).toHaveCount(expected.length, { timeout: 180_000 });
  const got = await words.allTextContents();
  const alts = await words.evaluateAll((els) => els.map((e) => (e as HTMLElement).dataset.alts?.split('|') ?? []));
  console.log('recognised:', got.join(' | '), '— alternatives:', JSON.stringify(alts));
  // KArSL framing is what the model expects: no framing warning.
  await expect(page.locator('.framing-tip')).toBeEmpty();
  // The camera video loops and may start mid-loop, so any rotation of the sequence counts.
  // Each word must be right or one tap away (among the alternatives offered for it): on this
  // slow test machine near-identical signs such as طبيب/صيدلي can swap places between runs.
  const rotations = expected.map((_, i) => [...expected.slice(i), ...expected.slice(0, i)]);
  const ok = rotations.some((r) => r.every((w, k) => got[k] === w || alts[k].includes(w)));
  expect(ok, `expected ${expected.join(' | ')}`).toBe(true);
  expect(errors).toEqual([]);
});

test('text → sign: plays the right clips in order', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/#/text');
  await page.getByRole('textbox').fill('السلام عليكم أنا طبيب');
  await page.getByRole('button', { name: 'ترجم إلى إشارة' }).click();

  const chips = page.locator('.plan .chip');
  await expect(chips).toHaveText(['السلام عليكم', /أنا/, 'طبيب']);
  await expect(page.locator('.player-word')).toHaveText('السلام عليكم');
  // A clip actually decodes and plays.
  await expect
    .poll(() => page.locator('video.clip.front').evaluate((v: HTMLVideoElement) => v.currentTime), { timeout: 20_000 })
    .toBeGreaterThan(0);
  await expect(page.locator('.player-word')).toHaveText('طبيب', { timeout: 30_000 });
  expect(errors).toEqual([]);
});

test('dictionary: search and open a sign', async ({ page }) => {
  await page.goto('/#/dict');
  await page.getByRole('searchbox').fill('مستشفي');
  await expect(page.locator('.dict-item')).toHaveCount(1);
  await page.locator('.dict-item').click();
  await expect(page.locator('dialog h2')).toHaveText('مستشفى');
});
