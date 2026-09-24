import { defineConfig } from '@playwright/test';

// Fake cameras are KArSL test videos (training/make_fake_camera.py): one of words for the main
// suite, one of fingerspelled letters for letters.spec.ts.
const camera = (file: string) => ({
  launchOptions: {
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-video-capture=${file}`,
      '--enable-unsafe-swiftshader',
    ],
  },
});

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 240_000,
  expect: { timeout: 20_000 },
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:4173',
    locale: 'ar-EG',
    permissions: ['camera'],
  },
  projects: [
    {
      name: 'words',
      testIgnore: /letters\.spec/,
      use: camera(process.env.SLI_FAKE_CAMERA ?? 'tests/fixtures/camera.y4m'),
    },
    {
      name: 'letters',
      testMatch: /letters\.spec/,
      use: camera(process.env.SLI_LETTERS_CAMERA ?? 'tests/fixtures/camera-letters.y4m'),
    },
  ],
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: true,
  },
});
