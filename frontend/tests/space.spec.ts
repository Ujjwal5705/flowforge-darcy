import { test, expect } from '@playwright/test';

test('the hosting adapter renders and runs its actual Gradio callback', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/benchmark/');
  await expect(page.getByRole('heading', { name: 'FLOWFORGE · benchmark check' })).toBeVisible();
  await page.getByRole('button', { name: 'Run benchmark check' }).click();
  await expect(page.getByText('darcy-fno-f711c406ba83', { exact: false })).toBeVisible();
  expect(errors).toEqual([]);
});
