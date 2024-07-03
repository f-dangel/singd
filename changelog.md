# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Fixed

## [0.0.4] - 2024-07-03

This release adds a new interface function to `SINGD`'s structured matrix
sub-module. This is a useful addition for our
[`SIRFShampoo`](https://github.com/f-dangel/sirfshampoo) optimizer. The SINGD
optimizer is not affected by this feature.

### Added

- Method to extract a structured matrix from `X @ X.T`
  ([PR](https://github.com/f-dangel/singd/pull/80))

## [0.0.3] - 2024-06-24

This release requires `torch>=2.2.0` and ships internal refactorings to update
newly supported operations in half precision.

### Added

- Functionality to compute the Frobenius norm of a structured matrix
  (interface [PR](https://github.com/f-dangel/singd/pull/70),
  efficient implementation [PR](https://github.com/f-dangel/singd/pull/71))

### Changed

- Update `.bib` entry to ICML 2024
  ([PR](https://github.com/f-dangel/singd/pull/74))

### Deprecated

- Use `torch>=2.2.0` and update its newly added support for `matmul`, `conv1d`,
  and `einsum` in half precision on CPU
  ([PR](https://github.com/f-dangel/singd/pull/72))

### Fixed

No bug fixes

## [0.0.2] - 2023-12-11

This release adds support for neural networks with in-place activations and also
comes with performance improvements for convolutions, as well as improvements
regarding numerical stability in half precision.

### Added

New features:

- Support `Conv2d` layers with `dilation != 1`
  ([PR](https://github.com/f-dangel/singd/pull/51))
- Support neural networks with inplace activation functions
  ([PR](https://github.com/f-dangel/singd/pull/63))

Performance improvements:

- Speed up input processing for `Conv2d` with `groups != 1`
  ([PR](https://github.com/f-dangel/singd/pull/59))
- Speed up computation of averaged patches for KFAC-reduce
  (`kfac_approx='reduce'`) in `Conv2d` using the tensor network approach of
  Dangel, 2023 ([PR](https://github.com/f-dangel/singd/pull/61))

### Changed

- Move un-scaling of `H_C` into the update step to improve numerical stability
  when using half precision + gradient scaling
  ([PR](https://github.com/f-dangel/singd/pull/67))

### Deprecated

No deprecations

### Fixed

No bug fixes

## [0.0.1] - 2023-10-31

Initial release

[unreleased]: https://github.com/f-dangel/singd/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/f-dangel/singd/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/f-dangel/singd/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/f-dangel/singd/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/f-dangel/singd/compare/5c68fa548ee947c841b3f7e6c31f65eeee079384...v0.0.1
