# Execution plan

Budget starts 2026-11-03T16:55:35.582Z; final commit must precede
2026-11-06T16:55:35.582Z. Work stays in this session using GPT-6.1 Sol.

1. Verify official loader/FNO compatibility with a tiny forward/backward test.
   Audit source arrays and obtain a bounded standard dataset subset.
2. Use PDEBench Darcy beta=1 if the inspected raw data checks pass. The tiny
   NeuralOperator example is not preferred because its physical-unit transform
   is absent. Freeze deterministic train/validation/test indices before fitting.
3. Train a modest CPU FNO from random initialization; select on validation only.
   Save normalization, architecture, history, best weights and dataset hashes.
   Evaluate the final held-out split and a training-mean-field baseline once.
4. Build FastAPI inference with one startup model load, strict bounded JSON
   validation, held-out reference comparisons and same-origin static serving.
5. Build a React/TypeScript/Vite/Tailwind client: editable coefficient canvas,
   honest presets, prediction/reference/error fields, downloads and methodology.
   Vite is the permitted fallback: this interactive client needs no SSR, and its
   static output avoids a second Node runtime in the small FastAPI container.
6. Run focused ML/API tests, build, browser interaction and mobile checks, then
   Docker build and inference. Prepare a free Hugging Face Docker Space with
   an explicit audience decision before public publication.
7. Verify remote endpoints, prediction and visualization; finish factual docs,
   scan tracked files, inspect history, and commit the complete source.

## Compute budget and risks

Eight-core ARM64 CPU, 8 GiB RAM. Use two PyTorch threads initially to avoid
oversubscription, 32×32 fields, batches of 32 and a modest four-layer FNO. Measure
epoch time before choosing the final number of epochs; target minutes to tens of
minutes rather than full benchmark-scale training. No GPU or paid resource is
required. MPS is visible outside the sandbox but CPU provides the conservative,
reproducible FFT path and matches the deployment target.

Risks: source encoding ambiguity (audit before selection); network throughput
(bounded HTTP ranges rather than whole shard); CPU overfitting (validation-based
checkpoint selection); mismatched preprocessing (shared model wrapper); public
deployment permissions (prepare and test before requesting audience approval).
