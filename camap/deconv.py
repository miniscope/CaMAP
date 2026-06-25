"""Non-negative FISTA deconvolution of calcium traces.

A pure NumPy/SciPy port of the deconvolution used in CaLab / CaTune
(https://github.com/miniscope/CaLab). A double-exponential kernel is built from
the indicator's rise/decay time constants, and the spike train is recovered by
minimising

    1/2 * ||K @ s + b - y||^2 + lam_eff * ||s||_1     subject to   s >= 0

with FISTA (Beck & Teboulle 2009) plus adaptive restart (O'Donoghue & Candes
2015). This replaces the external ``oasis-deconv`` dependency.
"""

import numpy as np
from scipy.fft import irfft, rfft

# Kernel decays to this fraction of its peak before being truncated.
_KERNEL_TAIL = 1e-6


def g_to_tau(g1: float, g2: float, fps: float) -> tuple[float, float]:
    """Convert AR(2) coefficients ``(g1, g2)`` to ``(tau_rise, tau_decay)``.

    Migration helper for configs that historically specified OASIS AR(2)
    coefficients. The AR(2) poles are the roots of ``x^2 - g1 * x - g2 = 0``;
    each pole ``p`` maps to a time constant ``tau = -dt / ln(p)`` where
    ``dt = 1 / fps``. The larger pole is the decay, the smaller the rise.

    Parameters
    ----------
    g1, g2:
        AR(2) coefficients (``g1 > 0``, ``g2 < 0``).
    fps:
        Sampling rate (frames per second).

    Returns
    -------
    tuple[float, float]
        ``(tau_rise, tau_decay)`` in seconds.
    """
    disc = g1 * g1 + 4.0 * g2
    if disc < 0:
        raise ValueError(
            f"AR(2) coefficients g1={g1}, g2={g2} have complex poles "
            "(g1**2 + 4*g2 < 0); cannot convert to real time constants."
        )
    sqrt_disc = np.sqrt(disc)
    p_decay = (g1 + sqrt_disc) / 2.0
    p_rise = (g1 - sqrt_disc) / 2.0
    if not (0.0 < p_rise < 1.0) or not (0.0 < p_decay < 1.0):
        raise ValueError(
            f"AR(2) poles ({p_rise:.4f}, {p_decay:.4f}) are outside (0, 1); "
            "coefficients do not describe a stable bi-exponential."
        )
    dt = 1.0 / fps
    tau_decay = -dt / np.log(p_decay)
    tau_rise = -dt / np.log(p_rise)
    return float(tau_rise), float(tau_decay)


def tau_to_g(tau_rise: float, tau_decay: float, fps: float) -> tuple[float, float]:
    """Convert ``(tau_rise, tau_decay)`` to AR(2) coefficients ``(g1, g2)``.

    Inverse of :func:`g_to_tau`. The AR(2) poles are
    ``d = exp(-dt / tau_decay)`` and ``r = exp(-dt / tau_rise)`` with
    ``dt = 1 / fps``; then ``g1 = d + r`` and ``g2 = -d * r``. Used to drive the
    OASIS engine from rise/decay time constants.

    Parameters
    ----------
    tau_rise, tau_decay:
        Rise and decay time constants in seconds.
    fps:
        Sampling rate (frames per second).

    Returns
    -------
    tuple[float, float]
        ``(g1, g2)`` AR(2) coefficients.
    """
    dt = 1.0 / fps
    d = np.exp(-dt / tau_decay)
    r = np.exp(-dt / tau_rise)
    g1 = d + r
    g2 = -d * r
    return float(g1), float(g2)


def _clamp_tau_rise(tau_rise: float, tau_decay: float) -> float:
    """Avoid a degenerate (zero) kernel when ``tau_rise`` ~= ``tau_decay``."""
    if abs(tau_rise - tau_decay) < 1e-6 * max(tau_rise, tau_decay, 1e-12):
        return tau_decay * 0.5
    return tau_rise


def build_kernel(tau_rise: float, tau_decay: float, fps: float) -> np.ndarray:
    """Build a peak-normalised double-exponential calcium kernel.

    ``h(t) = exp(-t / tau_decay) - exp(-t / tau_rise)``, sampled at ``1 / fps``
    and truncated where the decay term falls below ``_KERNEL_TAIL`` of its peak.

    Parameters
    ----------
    tau_rise, tau_decay:
        Rise and decay time constants in seconds.
    fps:
        Sampling rate (frames per second).

    Returns
    -------
    np.ndarray
        1-D kernel normalised so its peak equals 1.0.
    """
    tau_rise = _clamp_tau_rise(tau_rise, tau_decay)
    dt = 1.0 / fps
    kernel_len = int(np.ceil(-np.log(_KERNEL_TAIL) * tau_decay / dt))
    kernel_len = max(kernel_len, 2)

    t = np.arange(kernel_len, dtype=np.float64) * dt
    kernel = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)

    peak = kernel.max()
    if peak > 0:
        kernel = kernel / peak
    return kernel


def fista_deconvolve(
    y: np.ndarray,
    kernel: np.ndarray,
    lam: float,
    max_iters: int = 2000,
    tol: float = 1e-5,
    estimate_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover a non-negative spike train from a calcium trace via FISTA.

    Solves ``min 1/2 ||K s + b - y||^2 + lam_eff ||s||_1`` over ``s >= 0`` where
    ``K`` is convolution with ``kernel`` and ``lam_eff = lam * sum(kernel)`` (DC
    gain), so the sparsity penalty is invariant to kernel scaling.

    Parameters
    ----------
    y:
        1-D calcium trace (baseline correction may already be applied).
    kernel:
        Convolution kernel from :func:`build_kernel`.
    lam:
        Sparsity weight. ``0`` disables the L1 penalty.
    max_iters:
        Maximum FISTA iterations.
    tol:
        Relative convergence tolerance on the solution update.
    estimate_baseline:
        If True, refine a scalar DC offset each iteration as
        ``b = mean(y - K s)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(s, c, baseline)`` — spike train, denoised trace ``K s + b``, and the
        estimated baseline.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = y.size
    k = kernel.size

    # FFT length for linear convolution; cache kernel transform and its adjoint.
    fft_len = n + k - 1
    k_fft = rfft(kernel, n=fft_len)
    k_fft_conj = np.conj(k_fft)

    def conv(s: np.ndarray) -> np.ndarray:
        """Forward convolution K @ s, truncated to the trace length."""
        full = irfft(rfft(s, n=fft_len) * k_fft, n=fft_len)
        return full[:n]

    def conv_adjoint(r: np.ndarray) -> np.ndarray:
        """Adjoint (correlation) K^T @ r, truncated to the trace length."""
        full = irfft(rfft(r, n=fft_len) * k_fft_conj, n=fft_len)
        return full[:n]

    # Lipschitz constant of the gradient of the quadratic term = max |H(w)|^2.
    lipschitz = float(np.max(np.abs(k_fft) ** 2))
    if lipschitz <= 0:
        lipschitz = 1.0
    step = 1.0 / lipschitz

    g_dc = float(kernel.sum())
    threshold = step * lam * g_dc

    x = np.zeros(n, dtype=np.float64)
    y_k = x.copy()
    t_k = 1.0
    baseline = 0.0
    tol_sq = tol * tol

    for it in range(max_iters):
        recon = conv(y_k)
        if estimate_baseline:
            baseline = float(np.mean(y - recon))
        residual = recon + baseline - y
        gradient = conv_adjoint(residual)

        z = y_k - step * gradient
        # Proximal step: soft-threshold with non-negativity projection.
        x_new = np.maximum(0.0, z - threshold)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        momentum = (t_k - 1.0) / t_new
        y_new = np.maximum(0.0, x_new + momentum * (x_new - x))

        # Adaptive restart: reset momentum when it pushes uphill.
        if float(np.dot(y_k - x_new, x_new - x)) > 0.0:
            t_new = 1.0
            y_new = x_new.copy()

        # Relative convergence check on the solution update.
        if it >= 5:
            dx = x_new - x
            if float(dx @ dx) < tol_sq * (float(x @ x) + 1e-20):
                x = x_new
                break

        x = x_new
        y_k = y_new
        t_k = t_new

    c = conv(x) + baseline
    return x, c, baseline
