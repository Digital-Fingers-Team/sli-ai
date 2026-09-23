import { defineConfig } from '@playwright/test';

// The fake camera is a KArSL test video (see training/make_fake_camera.py and tests/e2e/README).
const fakeCamera = process.env.SLI_FAKE_CAMERA ?? 'tests/fixtures/camera.y4m';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 240_000,
  expect: { timeout: 20_000 },
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:4173',
    locale: 'ar-EG',
    permissions: ['camera'],
    launchOptions: {
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-video-capture=${fakeCamera}`,
        '--enable-unsafe-swiftshader',
      ],
    },
  },
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: true,
  },
});
