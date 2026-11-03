# Dataset audit

## Investigated alternative: NeuralOperator tiny example

Record: <https://doi.org/10.5281/zenodo.12784353>.
The record's API declares **CC BY 4.0**. This is the NeuralOperator team's Darcy
benchmark, **not PDEBench**. It is a standard dataset explicitly allowed by the
application brief's small-loader alternative.

The original FNO paper, Appendix A.3.2, specifies the unit square, fixed forcing
`f = 1`, homogeneous Dirichlet conditions, and a thresholded Gaussian coefficient
field taking values 3 and 12. Its solutions were generated on 421×421 grids with
a second-order finite-difference method. These are the original benchmark's
physical conventions, not automatic evidence of the compact archive's encoding.
Source: <https://arxiv.org/html/2010.08895v3>.

## Inspected tiny archive

- `darcy_16.tgz`, 1,010,949 bytes.
- Published and locally verified MD5: `091575565d8c4ab17f8c3bf2bbb566cb`.
- Training: 1,000 pairs of 16×16 arrays; test: 50 pairs.
- Inputs are Boolean, with approximately equal fractions of each phase.
- Training targets: range approximately −0.42788 to 2.05719.
- Test targets: range approximately −0.11912 to 1.90151.
- Targets at array edges are not identically zero.

This is evidence of a processed example representation. The archive has only
`x` and `y` tensors, without an inverse physical-unit transform. We must not invent
such a transform, apply 3/12 to Boolean inputs without evidence, call its target
values physical pressure, or impose a boundary mask that changes its labels.
The larger archive was inspected partially and also begins with 0/1 encoded
coefficients. Its transfer timed out at 4,631,800 of 93,721,772 bytes; its checksum
failed as expected and it is **not used**. No incomplete archive is treated as data.

## Selected source: PDEBench raw Darcy shard

- DOI: <https://doi.org/10.18419/darus-2986>.
- Permanent public file URL: <https://darus.uni-stuttgart.de/api/access/datafile/133219>.
- Filename: `2D_DarcyFlow_beta1.0_Train.hdf5`.
- Record license: **CC BY 4.0**.
- Full file: 1,310,724,488 bytes; published MD5
  `81694ed31306ff2e5f6b76349b0b4389`.
- Full-shard checksum is **not verified**, because we download bounded byte ranges,
  not the entire shard. Individual HTTP ranges are checked for response status,
  exact Content-Range and exact byte length. Saved subset arrays receive SHA-256 hashes.
- Verified HDF5 fields: `nu` = `[10000,128,128]`, `tensor` = `[10000,1,128,128]`;
  both float32, contiguous, uncompressed. Attribute `beta = 1.0`.
- Sample coefficient values: 0.1 and 1.0. First solution range: approximately
  0.0017174 to 0.4812452, in the stored benchmark's units.
- Coordinates are cell centers from 0.00390625 to 0.99609375 on `[0,1]²`.
- Intended subset: source sample indices 0–1023, spatial slice `[:, ::4, ::4]`,
  without interpolation, with the same coordinate slice. This is a custom bounded
  subset, not a canonical full-benchmark result.

### Physical and numerical conventions

The generator evolves diffusion with a constant source toward a steady Darcy
field: `-div(a grad u) = beta`, with beta=1 for the selected shard. The coefficient
is a thresholded sum of five Gaussian inclusions and takes values 0.1 or 1.0.
It uses a finite-volume-style flux update with arithmetic averages of adjacent
coefficients and cell-centered solution values. The configuration uses a unit
square and final evolution time 2.0. We describe the stored solution as the dataset
reference; we do not assert a newly verified exact continuum solve or convergence
certificate. No physical SI unit calibration is supplied, so the UI uses dataset units.

The generator calls `bc_2D(..., mode="Neumann")`, but the helper implementation
actually fills exterior ghost cells with **zero**, rather than enforcing a
zero normal derivative. Documentation and UI must describe this implemented
zero-exterior boundary convention. Since cell centers are interior to the domain,
their edge rows need not be zero. The learned model does not hard-enforce this
boundary behavior.

Array axes follow the generator: first spatial dimension is x, second is y.
Canvas coordinates and plots map these axes explicitly with increasing y upward;
API and downloaded arrays keep the original x-major order. Coarse input loses
fine-scale information; outputs are sampled fine-grid dataset references, not a
fresh numerical solve of the coarse-grid coefficient.

Primary source files inspected in this session:

- [Download instructions](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/README.md)
- [Generator](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/ReactionDiffusionEq/reaction_diffusion_2D_multi_solution_Hydra.py)
- [Boundary helper](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/utils.py)
- [Configuration](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/ReactionDiffusionEq/config/multi/config_2D.yaml)
