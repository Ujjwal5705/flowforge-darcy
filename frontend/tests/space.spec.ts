import { test, expect } from '@playwright/test';

test('the hosting adapter renders and runs its actual Gradio callback', async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/benchmark/');
  await expect(page.getByRole('heading', { name: 'FLOWFORGE · benchmark check' })).toBeVisible();
  await page.getByRole('button', { name: 'Run benchmark check' }).click();
  await expect(page.getByText('darcy-fno-f711c406ba83', { exact: false })).toBeVisible({
    timeout: 90_000,
  });
  expect(errors).toEqual([]);
});
