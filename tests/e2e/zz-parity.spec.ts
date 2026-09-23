// Diagnostic, not part of the suite: runs the browser landmarkers on KArSL JPEG frames copied to
// dist/debug-frames/ and saves the raw landmarks for comparison with Python. SLI_PARITY=1 to run.
import { test } from '@playwright/test';
import { readdirSync, writeFileSync } from 'node:fs';

test('parity', async ({ page }) => {
  test.skip(!process.env.SLI_PARITY, 'diagnostic only');
  const files = readdirSync('dist/debug-frames').sort();
  await page.goto('/?delegate=CPU&debug#/about');
  const raws = await page.evaluate(async (files) => {
    const { landmarks } = await (window as any).__sliLoadModels();
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 256;
    const ctx = canvas.getContext('2d')!;
    const out = [];
    let ts = 1;
    for (const f of files) {
      const img = new Image();
      img.src = `debug-frames/${f}`;
      await img.decode();
      ctx.drawImage(img, 0, 0, 256, 256);
      out.push(landmarks.detect(canvas, (ts += 33)));
    }
    return out;
  }, files);
  writeFileSync('test-results/parity-browser.json', JSON.stringify({ files, raws }));
  console.log('frames', raws.length);
});
