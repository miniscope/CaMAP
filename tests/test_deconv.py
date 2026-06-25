"""Tests for the FISTA calcium deconvolver in ``camap.deconv``."""

import numpy as np
import pytest
import xarray as xr

from camap.deconv import build_kernel, fista_deconvolve, g_to_tau, tau_to_g
from camap.neural import run_deconvolution


def test_build_kernel_peak_and_length() -> None:
    """Kernel is peak-normalised, starts near zero, and has a sensible length."""
    fps = 20.0
    tau_rise, tau_decay = 0.06, 0.9
    kernel = build_kernel(tau_rise, tau_decay, fps)

    assert kernel.ndim == 1
    assert np.isclose(kernel.max(), 1.0)
    assert kernel[0] == 0.0  # h(0) = 1 - 1 = 0
    # Length ~ -ln(1e-6) * tau_decay * fps.
    expected_len = int(np.ceil(-np.log(1e-6) * tau_decay * fps))
    assert kernel.size == expected_len


def test_clamp_tau_rise_avoids_degenerate_kernel() -> None:
    """Equal rise/decay must not produce an all-zero kernel."""
    kernel = build_kernel(0.5, 0.5, 20.0)
    assert kernel.max() > 0.0


def test_g_to_tau_round_trip() -> None:
    """g_to_tau recovers the time constants used to build AR(2) coefficients."""
    fps = 20.0
    tau_rise, tau_decay = 0.06, 0.9
    dt = 1.0 / fps
    d = np.exp(-dt / tau_decay)
    r = np.exp(-dt / tau_rise)
    g1 = d + r
    g2 = -d * r

    rise, decay = g_to_tau(g1, g2, fps)
    assert np.isclose(rise, tau_rise, rtol=1e-6)
    assert np.isclose(decay, tau_decay, rtol=1e-6)


def test_tau_to_g_inverts_g_to_tau() -> None:
    """tau_to_g and g_to_tau are inverses."""
    fps = 20.0
    g1, g2 = 1.3805576774138437, -0.4111122905071874
    tau_rise, tau_decay = g_to_tau(g1, g2, fps)
    g1_rt, g2_rt = tau_to_g(tau_rise, tau_decay, fps)
    assert np.isclose(g1_rt, g1, rtol=1e-9)
    assert np.isclose(g2_rt, g2, rtol=1e-9)


def test_g_to_tau_known_session_values() -> None:
    """The placecell test-data AR(2) values map to ~60 ms / ~900 ms."""
    rise, decay = g_to_tau(1.3805576774138437, -0.4111122905071874, 20.0)
    assert np.isclose(rise, 0.06, atol=2e-3)
    assert np.isclose(decay, 0.9, atol=2e-3)


def test_fista_recovers_sparse_spikes() -> None:
    """FISTA recovers planted spikes from a synthetic kernel*spikes signal."""
    rng = np.random.default_rng(0)
    fps = 20.0
    n = 2000
    kernel = build_kernel(0.06, 0.9, fps)

    spikes = np.zeros(n)
    spike_idx = rng.choice(n, size=20, replace=False)
    spikes[spike_idx] = rng.uniform(2.0, 5.0, size=20)

    full = np.convolve(spikes, kernel)[:n]
    y = full + rng.normal(0.0, 0.02, size=n)

    s, c, _ = fista_deconvolve(y, kernel, lam=0.1, max_iters=2000)

    # Denoised reconstruction tracks the observed trace.
    assert np.corrcoef(c, y)[0, 1] > 0.95

    # Recovered events cluster near the planted spikes.
    recovered = np.where(s > 0.5)[0]
    assert recovered.size > 0
    for idx in spike_idx:
        assert np.min(np.abs(recovered - idx)) <= 2


def _synthetic_traces(n_units: int = 3, n: int = 2000):
    """Build a small (unit_id, frame) DataArray of synthetic calcium traces."""
    rng = np.random.default_rng(1)
    fps = 20.0
    kernel = build_kernel(0.06, 0.9, fps)
    traces = np.zeros((n_units, n))
    for u in range(n_units):
        spikes = np.zeros(n)
        spikes[rng.choice(n, size=15, replace=False)] = rng.uniform(2.0, 5.0, size=15)
        traces[u] = np.convolve(spikes, kernel)[:n] + rng.normal(0.0, 0.02, size=n)
    da = xr.DataArray(
        traces,
        dims=("unit_id", "frame"),
        coords={"unit_id": list(range(n_units)), "frame": list(range(n))},
    )
    return da, fps


def test_run_deconvolution_fista_engine() -> None:
    """FISTA engine returns one spike train per unit at the right length."""
    da, fps = _synthetic_traces()
    ids, S = run_deconvolution(
        da,
        [0, 1, 2],
        fps=fps,
        tau_rise=0.06,
        tau_decay=0.9,
        lam=0.0,
        baseline="p10",
        s_min=0.0,
        engine="fista",
    )
    assert ids == [0, 1, 2]
    assert all(s.shape[0] == da.sizes["frame"] for s in S)
    assert all((s >= 0).all() for s in S)


def test_run_deconvolution_unknown_engine_raises() -> None:
    """An unrecognised engine name is rejected."""
    da, fps = _synthetic_traces(n_units=1)
    with pytest.raises(ValueError):
        run_deconvolution(
            da,
            [0],
            fps=fps,
            tau_rise=0.06,
            tau_decay=0.9,
            lam=0.0,
            baseline="p10",
            s_min=0.0,
            engine="bogus",
        )


def test_fista_and_oasis_engines_agree() -> None:
    """FISTA and OASIS place events in the same regions (oasis optional).

    Raw single-frame spike trains are too sparse for a meaningful Pearson
    correlation (a one-frame offset zeroes it), so we compare Gaussian-smoothed
    event-rate envelopes — the quantity downstream rate maps actually integrate.
    """
    oasis = pytest.importorskip("oasis.oasis_methods")
    assert oasis  # silence unused-import lint
    from scipy.ndimage import gaussian_filter1d

    da, fps = _synthetic_traces(n_units=3)
    kwargs = dict(fps=fps, tau_rise=0.06, tau_decay=0.9, lam=0.0, baseline="p10", s_min=0.0)
    _, S_f = run_deconvolution(da, [0, 1, 2], engine="fista", **kwargs)
    _, S_o = run_deconvolution(da, [0, 1, 2], engine="oasis", **kwargs)
    for sf, so in zip(S_f, S_o):
        n = min(len(sf), len(so))
        ef = gaussian_filter1d(sf[:n], sigma=5)
        eo = gaussian_filter1d(so[:n], sigma=5)
        r = np.corrcoef(ef, eo)[0, 1]
        assert r > 0.7
