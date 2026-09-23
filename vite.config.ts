import { defineConfig } from 'vite';

export default defineConfig({
  // Served from the root of a custom domain on GitHub Pages; relative paths also work in subfolders.
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: { include: process.env.SLI_REPLAY ? ['tests/replay/**/*.test.ts'] : ['tests/unit/**/*.test.ts'], testTimeout: 30000 },
});
