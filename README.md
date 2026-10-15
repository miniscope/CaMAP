# CaMAP

[![Tests](https://github.com/miniscope/CaMAP/actions/workflows/test.yml/badge.svg)](https://github.com/miniscope/CaMAP/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/camap)](https://pypi.org/project/camap/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/camap)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Calcium imaging mapping and analysis pipeline. Extracts neural events from
calcium traces, aligns them with animal behavior, and computes spatial
tuning metrics including rate maps, spatial information, stability, and
place-cell classification.

> **Alpha-stage** in `0.x` — APIs may change between minor releases.

## System requirements

Supported operating systems and Python versions are shown on
[PyPI](https://pypi.org/project/camap/), and dependencies are listed in
`pyproject.toml`. No non-standard hardware is required.

## Install

```bash
pip install camap
```

Or clone the repository for the demo data and development:

```bash
git clone https://github.com/miniscope/CaMAP.git
cd CaMAP
pip install .
```

`oasis-deconv` is required for the deconvolution step; see
[Installation](https://miniscope.github.io/CaMAP/installation.html) for the
recommended install path. A typical install takes below a few minutes on a
normal desktop computer.

## Demo

See the [Demo](https://miniscope.github.io/CaMAP/demo.html) for a
functionality demo on the small sample data included in `tests/assets`.

## Documentation

- [Installation](https://miniscope.github.io/CaMAP/installation.html)
- [Quickstart](https://miniscope.github.io/CaMAP/quickstart.html)
- [Demo](https://miniscope.github.io/CaMAP/demo.html)
- [Pipeline Details](https://miniscope.github.io/CaMAP/pipeline.html)

To run on your own data, see
[Quickstart](https://miniscope.github.io/CaMAP/quickstart.html).

## License

AGPL-3.0. See [LICENSE](LICENSE).
