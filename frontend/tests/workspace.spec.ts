import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFile } from 'node:fs/promises';

async function captureFromTop(page: import('@playwright/test').Page, path: string) {
  await page.evaluate(() => {
    document.documentElement.style.scrollBehavior = 'auto';
    (document.activeElement as HTMLElement | null)?.blur();
    window.scrollTo(0, 0);
  });
  await expect.poll(() => page.evaluate(() => scrollY)).toBe(0);
  await page.screenshot({ path, fullPage: true });
}

test('held-out prediction, comparison, real JSON/CSV exports, and console health', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1050 });
  await page.goto('/');
  await expect(page.getByTestId('input-canvas')).toBeVisible();
  await page.getByRole('button', { name: 'Run prediction' }).click();
  await expect(page.getByTestId('prediction-canvas')).toBeVisible();
  await expect(page.getByTestId('reference-canvas')).toBeVisible();
  await expect(page.getByTestId('error-canvas')).toBeVisible();
  const jsonDownload = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'JSON', exact: true }).click(),
  ]).then(([download]) => download);
  const result = JSON.parse(await readFile((await jsonDownload.path())!, 'utf8'));
  expect(result.prediction).toHaveLength(32);
  expect(result.prediction.flat().every(Number.isFinite)).toBeTruthy();
  expect(result.comparison.relative_l2).toBeGreaterThanOrEqual(0);
  expect(result.checkpoint_sha256).toHaveLength(64);
  const csvDownload = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'CSV', exact: true }).click(),
  ]).then(([download]) => download);
  const csv = (await readFile((await csvDownload.path())!, 'utf8')).trim().split('\n');
  expect(csv).toHaveLength(1025);
  const selected = csv[1 + 2 * 32 + 17].split(',').map(Number);
  expect(selected[0]).toBe(2);
  expect(selected[1]).toBe(17);
  expect(selected[4]).toBe(result.input[2][17]);
  expect(selected[5]).toBe(result.prediction[2][17]);
  await page.getByRole('button', { name: 'Hide reference comparison' }).click();
  await expect(page.getByTestId('reference-canvas')).toHaveCount(0);
  await page.getByRole('button', { name: 'Show reference comparison' }).click();
  await expect(page.getByTestId('reference-canvas')).toBeVisible();
  expect(errors).toEqual([]);
  await page.getByRole('button', { name: 'Relative L₂', exact: true }).click();
  await expect(page.getByRole('img', { name: /Validation relative L2 over/ })).toBeVisible();
  await page.getByRole('button', { name: 'MSE', exact: true }).click();
  await captureFromTop(page, '../artifacts/browser/desktop.png');
});

test('keyboard edits preserve x/y orientation and clear the reference; reset restores it', async ({
  page,
  request,
}) => {
  const sample = await (await request.get('/api/examples/test-01')).json();
  await page.goto('/');
  const canvas = page.getByTestId('input-canvas');
  await expect(canvas).toBeVisible();
  const paintLow = sample.coefficient[17][16] > 0.5;
  await page
    .getByRole('button', { name: paintLow ? 'Low · 0.1' : 'High · 1.0', exact: true })
    .click();
  await canvas.focus();
  await canvas.press('ArrowRight');
  await canvas.press('Space');
  await expect(page.getByText('CUSTOM INPUT', { exact: true })).toBeVisible();
  const pixel = await canvas.evaluate((element) => {
    const context = (element as HTMLCanvasElement).getContext('2d')!;
    return [...context.getImageData(17 * 16 + 8, (31 - 16) * 16 + 8, 1, 1).data].slice(0, 3);
  });
  expect(pixel).toEqual(paintLow ? [237, 240, 255] : [23, 56, 135]);
  const requestPromise = page.waitForRequest('**/api/predict');
  await page.getByRole('button', { name: 'Run prediction' }).click();
  const body = (await requestPromise).postDataJSON();
  expect(body.example_id).toBeNull();
  expect(body.coefficient[17][16]).toBe(paintLow ? 0.1 : 1);
  await expect(page.getByTestId('prediction-canvas')).toBeVisible();
  await expect(page.getByTestId('reference-canvas')).toHaveCount(0);
  await expect(page.getByText('No reference', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Reset field' }).click();
  await expect(page.locator('#sample-select')).toHaveValue('test-01');
  await expect(page.getByTestId('prediction-canvas')).toHaveCount(0);
});

test('presets, pointer drawing, sample selection and true loading/error recovery', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.getByTestId('input-canvas')).toBeVisible();
  await page.getByRole('button', { name: 'Channel', exact: true }).click();
  const box = (await page.getByTestId('input-canvas').boundingBox())!;
  await page.mouse.move(box.x + box.width / 3, box.y + box.height / 3);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 10 });
  await page.mouse.up();
  await expect(page.getByText('Custom field', { exact: true })).toBeVisible();
  await page.locator('#sample-select').selectOption('test-02');
  await expect(page.getByText('Held-out sample 02', { exact: true }).last()).toBeVisible();
  await page.route('**/api/predict', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'The model is busy. Please retry shortly.' }),
    }),
  );
  await page.getByRole('button', { name: 'Run prediction' }).click();
  await expect(page.getByRole('alert')).toContainText('The model is busy');
  await expect(page.getByRole('button', { name: 'Run prediction' })).toBeEnabled();
  await page.unroute('**/api/predict');
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route('**/api/predict', async (route) => {
    await gate;
    await route.continue();
  });
  await page.getByRole('button', { name: 'Run prediction' }).click();
  await expect(page.getByRole('status')).toContainText('Evaluating the operator');
  release();
  await expect(page.getByTestId('prediction-canvas')).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('mobile workspace has no horizontal overflow and runs a real prediction', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByTestId('input-canvas')).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
  ).toBeTruthy();
  await page.getByRole('button', { name: 'Run prediction' }).click();
  await expect(page.getByTestId('prediction-canvas')).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
  ).toBeTruthy();
  await captureFromTop(page, '../artifacts/browser/mobile.png');
});

test('initial workspace meets automated WCAG A/AA accessibility checks', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('input-canvas')).toBeVisible();
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(
    results.violations.map((item) => ({
      id: item.id,
      impact: item.impact,
      nodes: item.nodes.map((node) => node.target),
    })),
  ).toEqual([]);
});
