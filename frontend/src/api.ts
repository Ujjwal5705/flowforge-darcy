export async function api<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    });
  } catch {
    throw new Error('The solver could not be reached. Check your connection and try again.');
  }
  if (!response.ok) {
    let detail = 'The request could not be completed. Please try again.';
    try {
      const payload = await response.json();
      if (typeof payload.detail === 'string') detail = payload.detail;
    } catch {
      /* An upstream proxy may return a non-JSON error page. */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
