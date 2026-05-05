# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0.x`, breaking API changes may land in any release;
treat the package as alpha-stage until `1.0`.

## [0.1.0] — 2026-05-05

Initial public release. CaMAP was previously developed internally as `placecell`;
this is the first version published to PyPI under the new name.

### Added

- **CLI** (`camap`): `analysis` runs the full pipeline (load → preprocess →
  deconvolve → match → analyze → save bundle); `define-zones` and `detect-zones`
  build and apply maze zone graphs.
- **Datasets**: `ArenaDataset` for 2D open-field analysis, `MazeDataset` for
  1D arm/maze analysis, sharing a common abstract base (`BaseCaMAPDataset`).
- **Analysis primitives** (`camap.analysis`): rate maps, spatial information
  (Skaggs 1993), split-half stability with multi-scale block configurations,
  shuffle-based significance, place-field detection (Guo et al. 2023), coverage
  maps and curves, arm-by-arm population vector overlap (`compute_dataset_arm_pvo`).
- **Bundle format** (`.camap`): self-contained directory with config,
  per-neural-frame canonical table, occupancy/footprint arrays, per-unit
  results, summary figures, and a versioned `metadata.json`. Round-trips via
  `BaseCaMAPDataset.save_bundle()` / `load_bundle()`.
- **Notebook viewers** for arena and maze bundles (`notebook/view_results_*.ipynb`).
- **Docs**: installation, quickstart, pipeline reference, CLI reference,
  workflows, notebooks (built with Sphinx + sphinx-book-theme).

### Notes

- **Alpha-stage.** APIs may change without notice in `0.x`; the
  `Development Status :: 3 - Alpha` classifier is set on PyPI and the alpha
  signal is carried by the leading `0.x` per common scientific-Python practice
  (rather than a strict PEP 440 `aN` suffix that would force `pip install --pre`).
- **Companion paper.** A `CITATION.cff` will be added when the associated
  publication is finalized.
- **Internal history.** `placecell` git tags `v0.1.0`–`v0.7.0` (private dev
  before the rename) were deleted from `origin`; this `0.1.0` is the first
  publicly-tagged release.
- **No backward compatibility for legacy `.pcellbundle` directories** —
  loading raises with explicit rename instructions. Re-run the pipeline to
  produce a `.camap` directory.
