---
title: FLOWFORGE — Darcy Flow
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# FLOWFORGE

A React/TypeScript scientific workspace and FastAPI inference API for a Fourier
neural operator trained from scratch on a documented PDEBench Darcy beta=1 subset.
The full application is served at `/`; `/benchmark` is a Gradio benchmark check
using the same trained checkpoint and preprocessing. The main workspace/API use
CPU. The auxiliary benchmark genuinely runs on an on-demand GPU, consuming free
daily ZeroGPU quota; it falls back to CPU when run locally. No paid upgrade or
external model inference service is used.

Training: 800 examples; validation: 112; final test: 112; 32×32 spatial subsampling.
Test mean relative L2: 0.061768; MSE: 0.000435679; MAE: 0.0107552 in dataset units.
The checkpoint and measured reports are included in this repository.

See [the full technical README](TECHNICAL_README.md) for setup, training commands,
API contracts, architecture, metrics and limitations.

Data: [PDEBench, DOI 10.18419/darus-2986](https://doi.org/10.18419/darus-2986),
Takamoto et al. (2022), CC BY 4.0. This Space redistributes only six attributed
held-out examples, not the full dataset. See `docs/data-audit.md`,
`docs/methodology.md` and `docs/experiments.md` for source conventions, preprocessing
and evaluation. NeuralOperator's maintained FNO implementation is used.

This is a research surrogate for one fixed PDE, forcing, domain and input
resolution. It does not enforce exact boundary behavior or guarantee correctness
on custom fields. Predictions use dataset units without SI calibration.

Author: Ujjwal Sharma · QUEST Lab application · 2026.
