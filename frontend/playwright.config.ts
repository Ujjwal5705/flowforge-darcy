import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  // Hosted cold-page transfers measured ~25 seconds on this connection.
  timeout: process.env.FLOWFORGE_URL?.startsWith('https://') ? 90_000 : 30_000,
  expect: { timeout: 10_000 },
  workers: 1,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: process.env.FLOWFORGE_URL ?? 'http://127.0.0.1:7860',
    browserName: 'chromium',
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROME_PATH,
      timeout: 30_000,
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
    screenshot: 'only-on-failure',
  },
});
