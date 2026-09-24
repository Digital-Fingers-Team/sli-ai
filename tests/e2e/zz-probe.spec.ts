// Diagnostic, not part of the suite: records what the browser sees. Run with -g probe.
import { test } from '@playwright/test';
import { writeFileSync } from 'node:fs';

test('probe', async ({ page }) => {
  test.skip(!process.env.SLI_PROBE, 'diagnostic only');
  await page.goto(`/?delegate=CPU&debug&timescale=4${process.env.SLI_URL_EXTRA ? '&' + process.env.SLI_URL_EXTRA : ''}#/sign`);
  await page.getByRole('button', { name: 'تشغيل الكاميرا' }).click();
  await page.waitForFunction(() => document.querySelector('.fps')?.textContent, null, { timeout: 120_000 });
  await page.waitForTimeout(Number(process.env.SLI_PROBE_MS ?? 20000));
  const frames = await page.evaluate(() => (window as any).__sliFrames);
  const words = await page.locator('.sentence-words .word').allTextContents();
  writeFileSync('test-results/probe.json', JSON.stringify({ frames, words }));
  console.log('frames', frames.length, 'words', words.join(' | '));
});
