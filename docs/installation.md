# Installation

```bash
pip install camap
```

CaMAP is alpha-stage in `0.x` — APIs may change between minor releases.

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

## Development

```bash
git clone https://github.com/miniscope/camap.git
cd camap
uv sync --extra all
```
