# Verification record

The main checkpoint completed its full train/reload/evaluate path. The measured
quantities are recorded in `artifacts/darcy-fno/` and `docs/experiments.md`.

## Completed checks

- Python ML/API/hosting suite: **25 passed** on the compatible FastAPI 0.162.0 stack.
  Includes independent metric formulas, normalization, deterministic/disjoint
  partitions, dataset and checkpoint corruption, tensor-only checkpoint reload,
  API readiness, strict shape/value/type validation, non-finite JSON, malformed
  bodies, size limits with/without Content-Length, actual trained predictions,
  example retrieval, reference identity enforcement, the shared Gradio lifespan,
  and denial of unused file-upload/file-path routes.
- TypeScript production build: passed.
- Ruff checks and formatting: passed for the implemented Python modules.
- Real headless Chromium: **6 passed** on the final hosting adapter, covering
  the complete five-test main workspace suite and the real Gradio callback UI.
  The main suite covers held-out prediction and comparison,
  JSON/CSV downloads with numeric orientation checks, keyboard and pointer edits,
  resets/presets/sample loading, loading/error recovery and mobile overflow.
- Automated axe checks: no WCAG A/AA violations in the initial workspace.
- Desktop and 390 px mobile screenshots inspected; actual held-out figure inspected.
- CLI checkpoint reload/inference passed with finite `[32,32]` output and the
  expected checkpoint hash. Verified local subset reuse passed its stored SHA-256 checks.
- The Gradio callback was also called over actual HTTP using gradio_client, returning
  the same model version and finite `[32,32]` prediction as the main API.
- CPU Docker image: built successfully; non-root runtime became model-ready in
  3.57 seconds of polling. Image size 286,636,421 bytes; observed memory about
  235 MiB. Independent HTTP prediction/metric/checkpoint checks passed. All five
  main browser tests passed against the container, with no application errors
  found in its startup/runtime logs. See `docs/deployment.md` for measurement scope.

The first integration-suite invocation preceded completion of the evaluation
artifact and correctly reported three setup errors. It was rerun after the
artifact existed; all core tests passed, followed by the final 25-test suite.
System Chrome 156 could not initialize
EGL in this environment. Matching Playwright Chromium 136 with software rendering
ran the browser tests successfully. Screenshots are real browser captures.

Matplotlib reported upstream pyparsing deprecations and a system font-cache
permission warning during figure generation; the figures were generated and
visually checked. The Makefile directs caches to the project for subsequent runs.
No warnings or failing checks were suppressed.

## Still being verified

The private hosted application is undergoing its deployment checks. It is not
yet reported as deployed or verified here.
This record is updated from actual results before the final commit.
