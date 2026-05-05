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

`oasis-deconv` is required for the deconvolution step; see
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

## Documentation

- [Installation](https://miniscope.github.io/CaMAP/installation.html)
- [Quickstart](https://miniscope.github.io/CaMAP/quickstart.html)
- [Pipeline Details](https://miniscope.github.io/CaMAP/pipeline.html)

## License

AGPL-3.0. See [LICENSE](LICENSE).
