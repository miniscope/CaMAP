# CaMAP

[![Tests](https://github.com/miniscope/CaMAP/actions/workflows/test.yml/badge.svg)](https://github.com/miniscope/CaMAP/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/camap)](https://pypi.org/project/camap/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Calcium imaging mapping and analysis pipeline. Extracts neural events from
calcium traces, aligns them with animal behavior, and computes spatial
tuning metrics including rate maps, spatial information, stability, and
place-cell classification.

> **Alpha-stage** in `0.x` — APIs may change between minor releases.

## Install

```bash
pip install camap
```

The default deconvolution engine (`fista`) needs no extra dependencies. The
optional `oasis` engine additionally requires `oasis-deconv`; see
[Installation](https://miniscope.github.io/CaMAP/installation.html) for the
recommended install path.

## Minimum example

```python
from camap.dataset import BaseCaMAPDataset

ds = BaseCaMAPDataset.from_yaml("config.yaml", "data_paths.yaml")
ds.load()
ds.preprocess_behavior()
ds.deconvolve()
ds.match_events()
ds.compute_occupancy()
ds.analyze_units()
ds.save_bundle("output/session_name")
```

## Choosing a deconvolution engine

`ds.deconvolve()` turns calcium traces into event ("spike") trains. CaMAP ships
two engines, selected in the config under `neural.deconv.engine`:

```yaml
neural:
  deconv:
    engine: fista   # default; or 'oasis'
    tau_rise: 0.06   # seconds
    tau_decay: 0.9   # seconds
    lam: 0.8         # sparsity (higher = fewer events)
    baseline: p10
    s_min: 0
```

**What's the same.** Both engines assume the *same blip shape* for a single
event: a double-exponential set by a rise time and a decay time
(`tau_rise`, `tau_decay`). OASIS writes this as AR(2) coefficients `(g1, g2)`
and FISTA writes it as `exp(-t/tau_decay) − exp(-t/tau_rise)`, but these are two
spellings of the same curve — CaMAP converts between them exactly, so you give
rise/decay times either way.

**What's different.** A deconvolution *method* is more than the blip shape. It
also includes how the baseline is estimated, how sparsity is enforced, the
solver, and small extras. The two engines share the blip but differ on the
rest:

| | OASIS (`oasis`) | FISTA (`fista`) |
|---|---|---|
| Event blip | double-exponential | double-exponential (same) |
| Solver | fast exact AR(2) recursion | iterative non-negative L1 (FFT-based) |
| Sparsity | `lam` + hard `s_min` floor | `lam` (L1 weight); `s_min` applied as a post-floor |
| Baseline | subtracted up front (`baseline`) | subtracted up front, plus a joint DC refinement |
| Dependency | needs `oasis-deconv` | none (pure NumPy/SciPy) |

## Documentation

- [Installation](https://miniscope.github.io/CaMAP/installation.html)
- [Quickstart](https://miniscope.github.io/CaMAP/quickstart.html)
- [Pipeline Details](https://miniscope.github.io/CaMAP/pipeline.html)

## License

AGPL-3.0. See [LICENSE](LICENSE).
