# Demo

A functionality demo of the full `camap analysis` pipeline on the small
dataset bundled with the repository under `tests/assets/regression_2d/`.
The data is a 2D arena recording cut down to 10 units and 5000 neural
frames, with the shuffle count lowered to 100. It is a minimal
functionality check for the install, not a full experiment: with this
little data, no unit is expected to pass the place-cell criteria.

## Setup

The demo data ships in the repository, so clone it and install from the
checkout (see [Installation](installation.md) for `oasis-deconv`):

```bash
git clone https://github.com/miniscope/CaMAP.git
cd CaMAP
pip install .
pip install --no-binary oasis-deconv oasis-deconv
```

## Run

From the repository root:

```bash
camap analysis \
  -c tests/assets/regression_2d/analysis_config.yaml \
  -d tests/assets/regression_2d/data_paths.yaml \
  -o output/demo_arena.camap -y
```

`-y` skips the interactive "Proceed with analyze_units?" prompt.

## Expected output

The console shows the pipeline steps, a 10-unit progress bar, and ends with:

```text
Done. 10 units analyzed.
Total: 10 | SI pass: 0 (0%) | Stability pass: 0 (0%) | Both: 0 (0%) | Neither: 10
Full analysis saved to output/demo_arena.camap
```

Zero place cells is the expected result for this subset. NumPy
`RuntimeWarning: invalid value encountered in divide` lines may appear
during the stability shuffles; they come from constant-valued map halves
in the tiny dataset and can be ignored here.

`output/demo_arena.camap/` contains:

- `unit_results/scalars.csv`, `unit_results/stability_scalars.csv` (csv, per-unit spatial information, p-values, stability)
- `unit_results/events.parquet` (parquet, per-unit events), `unit_results/*.npz` (npz, rate maps and shuffle arrays)
- `figures/*.pdf` (pdf, nine summary figures: `arena_calibration`, `behavior_preview`, `diagnostics`, `footprints`, `occupancy`, `preprocess_steps`, `stability_splits`, `summary_scatter`, `timestamp_diagnostics`)
- `canonical.parquet`, `trajectory*.parquet` (parquet, per-frame position, speed, deconvolved activity)
- `deconv.npz`, `spatial.npz` (npz, deconvolution and occupancy arrays), `traces.nc`, `footprints.nc` (netCDF, raw traces and footprints)
- `config.yaml`, `data_config.yaml` (yaml, copies of the input configs), `metadata.json` (json, bundle schema and `camap` version)

The same fixture is checked against a saved reference bundle by
`tests/test_regression_2d.py`, so a successful run plus a passing test
suite confirms the outputs match.

## Run time

Below a few minutes on a normal desktop (about 20 s measured on a laptop,
single worker).

## Running the tests

The test suite runs the same pipeline on the bundled 2D arena and 1D maze
fixtures and compares against saved references:

```bash
pip install ".[tests]"
pytest tests/
```

Expect all tests to pass in below a few minutes (about 15 s measured on a
laptop).
