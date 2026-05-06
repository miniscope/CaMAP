# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0.x`, breaking API changes may land in any release;
treat the package as alpha-stage until `1.0`.

## [0.1.3] — 2026-05-06

### Fixed

- `clip_to_arena` now logs the number of clamped frames and the maximum
  out-of-bounds deviation at INFO. Previously silent — closes the
  "no silent data repair" policy gap.

## [0.1.2] — 2026-05-06

### Fixed

- Stability shuffle no longer injects `0.0` for degenerate iterations;
  the p-value denominator now uses the count of valid shuffles
  (Phipson & Smyth 2010). Previously anti-conservative for stable units
  on sparse or short sessions.

### Changed

- Arena `preprocess_behavior` step order: `perspective → Hampel → clip
  → mm` (was Hampel first). Bundle outputs unchanged for existing data.
- `oasis-deconv` `ImportError` reformatted with install commands on
  separate lines.

## [0.1.1] — 2026-05-06

### Fixed

- Project URLs in `pyproject.toml`, `README.md`, and docs now use the
  canonical PascalCase GitHub repository name `miniscope/CaMAP`. The
  previously-listed lowercase paths returned 404 on GitHub Pages (which
  is case-sensitive on case-sensitive filesystems).

### Added

- Python 3.13 support (`requires-python = ">=3.11,<3.14"`). PyPI classifier
  and black `target-version` updated; the matrix in `test.yml` already
  exercised 3.13.

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
