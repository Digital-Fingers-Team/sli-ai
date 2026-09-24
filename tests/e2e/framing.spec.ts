import { expect, test } from '@playwright/test';

// The framing camera shows أهلا وسهلا cropped as if the signer stood too close to an upright
// phone (make_fake_camera.py --crop 0.5,0.67): a hand leaves the picture while the elbow is
// still in view. The app must say so.
test('framing: warns when a hand is outside the picture', async ({ page }) => {
  await page.goto('/?delegate=CPU&timescale=4#/sign');
  await page.getByRole('button', { name: 'تشغيل الكاميرا' }).click();
  await expect(page.getByRole('button', { name: 'إيقاف الكاميرا' })).toBeVisible({ timeout: 120_000 });
  await expect(page.locator('.framing-tip')).toContainText('ابتعد قليلًا', { timeout: 180_000 });
  await expect(page.locator('.camera-stage')).toHaveClass(/framing-bad/);
});
