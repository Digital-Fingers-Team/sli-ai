// Diagnostic, not part of the suite: phone-width screenshots of every page. SLI_SHOTS=1 to run.
import { test } from '@playwright/test';

test.use({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });

test('shots', async ({ page }) => {
  test.skip(!process.env.SLI_SHOTS, 'diagnostic only');
  for (const r of ['sign', 'text', 'talk', 'dict', 'about']) {
    await page.goto(`/?delegate=CPU#/${r}`);
    if (r === 'text') {
      await page.getByRole('textbox').fill('السلام عليكم أنا طبيب اسمي أحمد');
      await page.getByRole('button', { name: 'ترجم إلى إشارة' }).click();
      await page.waitForTimeout(2500);
    }
    if (r === 'dict') await page.waitForTimeout(1500);
    await page.screenshot({ path: `test-results/shot-${r}.png`, fullPage: false });
  }
  await page.goto('/?delegate=CPU#/sign');
  await page.getByRole('button', { name: 'English' }).click();
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.screenshot({ path: 'test-results/shot-sign-en-dark.png', fullPage: true });
});
