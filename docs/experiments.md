# Experiments

All code, data access, experiments and commits were initiated in this single Codex
session using GPT-6.1 Sol. No pretrained weights or other agents were used.

## Compatibility smoke test

The official `load_darcy_flow_small` API from neuraloperator 2.0.0 loaded four
training and two test examples at 16×16. A two-layer, eight-channel FNO with
`n_modes=(4,4)` ran four optimizer updates. Batch input/output shape was
`[2,1,16,16]`; all outputs were finite. MSE decreased from 0.353720 to 0.227477.
This was an engineering smoke test, not a physical benchmark result. See
[smoke.json](smoke.json).

The compact archive has Boolean source inputs and processed target values whose
inverse physical transform is not included. Rather than invent a mapping, the
main experiment used the raw PDEBench Darcy beta=1 shard. Its documented binary
coefficients, source coordinates, forcing attribute and generator were inspected.
The dataset audit records the boundary helper's misleading mode name explicitly.

## Bounded source preparation

The official DaRUS file ID 133219 supports HTTP byte ranges. We fetched source
samples 0–1023 from its contiguous `nu` and `tensor` arrays, plus metadata and
coordinates, transferring 134,287,752 bytes in 1,254.20 seconds. Each range was
checked for HTTP 206, exact Content-Range and exact byte count. The full-shard
MD5 was not checked because the complete 1.31 GB object was not downloaded.
The saved 32×32 arrays are SHA-256 verified before training/evaluation.

Sample each spatial axis at indices 0,4,…,124. No synthetic benchmark samples,
interpolation, phase remapping or label clipping were used. Source indices 0–1023
are a bounded cohort, not the official full-benchmark split. NumPy default_rng(42)
defines a saved 800/112/112 partition. Coefficient fingerprints were checked to
prevent exact duplicate fields across those partitions.

## PDEBench pipeline smoke run

A separate ignored `artifacts/pde-smoke` run used 64 training and 16 validation
examples, an eight-channel, two-layer FNO and three epochs. Training normalized
MSE decreased from 1.001706 to 0.995187; validation relative L2 decreased from
0.604374 to 0.599915. Training/validation loop wall time was 0.469 seconds.
The tensor-only checkpoint reloaded and produced finite `[1,1,32,32]` output.
Its unoptimized test partition was not evaluated or used to select the final
experiment. These weights were not the full experiment's initialization.

## Final training run

| Item | Recorded value |
| --- | --- |
| Initialization | Random, seed 42 |
| Implementation | neuraloperator 2.0.0, PyTorch 2.8.0 |
| Tensor input/output | `[batch,1,32,32]` |
| FNO | 4 blocks, 32 hidden channels, n_modes=(12,12) |
| Coordinates | Built-in grid embedding; physical source coordinates saved separately |
| Parameters | 357,217 tensor elements / 701,281 independent real scalars |
| Objective | MSE in standardized target units |
| Optimizer | AdamW, lr 0.003, weight decay 0.0001 |
| Schedule | Cosine annealing over 80 epochs |
| Gradient clipping | Norm 1.0 |
| Batch size | 32 |
| Checkpoint criterion | Lowest validation mean per-sample relative L2 |
| Selected epoch | 73; validation relative L2 0.0560952 |
| Duration | 1,146.09 seconds for training/validation loop and checkpoint writes |
| Hardware | ARM64 macOS 15.0.1, eight logical CPUs, 8 GiB RAM, two torch threads |

The final run enabled deterministic PyTorch algorithms. CPU was chosen despite
MPS visibility outside the sandbox, avoiding an unverified complex-FFT accelerator
path and matching the intended small inference workload. No CUDA device was available.

Training-only scalar normalization:

| Variable | Mean | Population std |
| --- | ---: | ---: |
| Coefficient | 0.5416702032 | 0.4499228895 |
| Solution | 0.1602898836 | 0.1387024671 |

## Final held-out evaluation

The best checkpoint was reloaded before test evaluation. The test split did not
enter normalization fitting, optimization or checkpoint selection. No subsequent
hyperparameter tuning used its reported metrics.

| Metric, 112 held-out samples | FNO | Training-mean-field baseline |
| --- | ---: | ---: |
| MSE | 0.000435678737 | 0.013103230539 |
| MAE | 0.010755230519 | 0.077857602526 |
| Mean per-sample relative L2 | 0.061768142852 | 0.575351194163 |

The baseline is the pixelwise training mean solution. Metrics average decoded
dataset units, with per-sample relative-L2 denominator floored at 1e-12.
No numerical-solver runtime comparison or state-of-the-art comparison was made.

Batch-one wrapper latency: median 3.675583 ms, p95 4.292804 ms, 100 calls after
10 warm-ups, two CPU threads on the same ARM64 machine. This excludes transport,
serialization, queuing and rendering. Actual API timings are separately measured
and can differ because of host load, thread initialization and deployment hardware.

The qualitative figure uses the first frozen test entry rather than a selected
best-looking example. Prediction and reference share limits; the absolute error
has its own nonnegative scale. Axes are original array indices with explicit
Cartesian rendering. See [comparison.png](results/comparison.png) and
[training.png](results/training.png).

Checkpoint SHA-256:
`f711c406ba83cee285f97adf2cd067c803e4d915753ee613e5f2639c727265e5`.
The full per-epoch history, source hashes, split indices, exact constructor config,
checkpoint and reports are versioned under `artifacts/darcy-fno/`.

## Interpretation and limits

The FNO improves on this simple baseline within the tested cohort. That evidence
does not establish generalization to the full shard, unseen physics, resolutions,
coarse geometries, or arbitrary edited fields. Subsampling omits fine structure.
The surrogate is unconstrained and can produce negative edge values. Dataset
units have no supplied SI calibration, and no residual/convergence guarantee is
provided. Uncertainty calibration and physics constraints remain future work.
