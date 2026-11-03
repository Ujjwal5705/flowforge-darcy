# Methodology contract

FLOWFORGE approximates the coefficient-to-solution map for the selected PDEBench
Darcy beta=1 subset. It does not solve arbitrary PDEs or change forcing, geometry,
boundary behavior, or resolution on request. The dataset audit records the precise
source conventions and the important distinction between physical boundaries and
interior cell-center samples.

## Preprocessing

Source tensors are sampled at indices `0,4,...,124` in both spatial dimensions.
For both coefficient and solution, fit one scalar mean and population standard
deviation over all pixels of training examples only. Normalize with
`z = (value - mean) / std`; decode the model output with `u = z * std_u + mean_u`.
The FNO receives one coefficient channel and adds its built-in two-dimensional
grid embedding internally. No externally added coordinate channels, test-fitted
statistics, output clipping, or boundary mask are permitted.

## Loss and selection

Optimize mean squared error in standardized target units. Record per-epoch training
loss and validation MSE and relative L2. Save a checkpoint whenever validation mean
relative L2 improves. Reload that checkpoint before final evaluation. The test
split is reserved for the final report and qualitative examples.

## Metric definitions

For N test examples and P spatial values per example, prediction p and reference u:

- `MSE = sum((p-u)^2) / (N*P)` in squared dataset solution units.
- `MAE = sum(abs(p-u)) / (N*P)` in dataset solution units.
- `relative_L2 = mean_n(||p_n-u_n||_2 / max(||u_n||_2, 1e-12))`.

Relative L2 averages individual sample ratios; it is not a ratio of aggregate
batch norms. Report the same definitions for the pixelwise training-mean solution
baseline on exactly the same test examples. Report parameter counts as both
PyTorch tensor elements and independent real scalars (complex weights count twice
in the latter).

Latency measurements use a reloaded model in inference mode, batch size one, a
stated number of CPU threads, warm-up calls, and repeated timed calls. Report the
median and 95th percentile. They exclude network transit, server queuing, JSON
serialization and browser rendering. The app can separately measure round-trip
latency. No comparison to a numerical solver's speed is inferred from these values.

## Scope of evidence

Random held-out examples from a bounded source cohort establish performance on
that cohort. They do not establish generalization to unseen physics, fine-scale
coefficient structure, different resolutions or arbitrary hand-drawn geometries.
Even a range-valid, binary custom field may lie outside the training distribution.
Homogeneous, inclusion and channel presets are exploratory inputs without reference
solutions. Predictions are unconstrained approximations; neither the PDE residual,
positivity, nor the original boundary behavior is guaranteed.
