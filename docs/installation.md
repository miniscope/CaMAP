# Installation

## From PyPI (stable)

```bash
pip install camap
```

This pulls the latest released version. CaMAP is alpha-stage in `0.x`, so
APIs may change between minor releases — pin the version (`pip install
camap==0.1.0`) if you need reproducibility.

## From GitHub (unreleased / development snapshots)

To install the in-development version directly from `main`:

```bash
pip install git+https://github.com/miniscope/camap.git
```

Or a specific branch / tag / commit:

```bash
pip install git+https://github.com/miniscope/camap.git@<branch-or-tag-or-sha>
```

Useful when you want a fix or feature that has been merged but not yet
cut into a release.

## Install oasis-deconv (required for deconvolution)

CaMAP uses [`oasis-deconv`](https://github.com/j-friedrich/OASIS) for
its deconvolution step. It is **required** for the full pipeline but
**not bundled** with CaMAP — PyPI wheel coverage is patchy (arm64
macOS only), so we leave the install path to you:

```bash
# recommended: force a source build for consistency across platforms (needs a C compiler)
pip install --no-binary oasis-deconv oasis-deconv

# alternative: prebuilt binaries via conda-forge
conda install -c conda-forge oasis-deconv
```

If the source build fails, that is an upstream issue — see the
[oasis-deconv repository](https://github.com/j-friedrich/OASIS).

> **uv users.** `pyproject.toml` carries
> `[tool.uv] no-binary-package = ["oasis-deconv"]`, which forces a source
> build automatically when `uv sync` is used. The directive is uv-specific
> and does not apply to plain `pip install`, which is why the explicit
> `--no-binary` flag is recommended above.

## Development

```bash
git clone https://github.com/miniscope/camap.git
cd camap
uv sync --extra all
```

`--extra all` pulls in development, test, docs, and notebook dependencies.
For a leaner setup install only the extras you need (e.g. `--extra tests`).
