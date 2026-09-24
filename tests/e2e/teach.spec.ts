import { expect, test } from '@playwright/test';

// "Teach the app your hand": with the letters camera (a signer holding letters), the page must
// see the hand, record the first letter and move on to the next.
test('teach: records a letter and moves on', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`${e}\n${e.stack}`));
  page.on('console', (m) => m.type() === 'error' && !m.text().startsWith('INFO:') && errors.push(m.text()));

  await page.goto('/?delegate=CPU#/sign');
  await page.getByRole('link', { name: /علّم التطبيق يدك/ }).click();
  await expect(page.getByRole('heading', { name: 'علّم التطبيق يدك' })).toBeVisible();
  await page.getByRole('button', { name: 'ابدأ' }).click();
  await expect(page.locator('.teach-glyph')).toHaveText('ا', { timeout: 120_000 });
  await expect(page.locator('.teach-card .hint')).toHaveText('1 من 39');
  await expect(page.locator('.teach-card .hint')).toHaveText('2 من 39', { timeout: 180_000 });
  await expect(page.locator('.teach-glyph')).toHaveText('ب');
  expect(errors).toEqual([]);
});
