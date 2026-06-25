"""Neural data loading and deconvolution."""

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from camap.deconv import build_kernel, fista_deconvolve, tau_to_g
from camap.log import init_logger

logger = init_logger(__name__)

_OASIS_INSTALL_HINT = (
    "The 'oasis' deconvolution engine requires the oasis-deconv package, which "
    "is not bundled with CaMAP. Either set neural.deconv.engine to 'fista' "
    "(no extra dependency) or install oasis-deconv with one of:\n"
    "\n"
    "  # source build (recommended; needs a C compiler)\n"
    "  pip install --no-binary oasis-deconv oasis-deconv\n"
    "\n"
    "  # prebuilt binaries via conda-forge\n"
    "  conda install -c conda-forge oasis-deconv\n"
    "\n"
    "See https://github.com/j-friedrich/OASIS for more."
)


def load_calcium_traces(
    neural_path: Path,
    trace_name: str = "C",
) -> xr.DataArray:
    """Load traces from a Minian-style zarr store as a DataArray.

    Parameters
    ----------
    neural_path:
        Directory containing ``<trace_name>.zarr``.
    trace_name:
        Base name of the zarr group (e.g. ``"C"`` or ``"C_lp"``).
        Also used as the variable name if the zarr contains a Dataset.

    Returns
    -------
    xr.DataArray
        DataArray with dimensions ('unit_id', 'frame').
    """
    zarr_path = neural_path / f"{trace_name}.zarr"
    ds_or_da = xr.open_zarr(zarr_path, consolidated=False)

    if isinstance(ds_or_da, xr.Dataset):
        if trace_name not in ds_or_da:
            raise KeyError(
                f"Variable {trace_name!r} not found in dataset; "
                f"available: {list(ds_or_da.data_vars)}"
            )
        C = ds_or_da[trace_name]
    else:
        C = ds_or_da

    if "unit_id" not in C.dims or "frame" not in C.dims:
        raise ValueError(f"Expected dims ('unit_id','frame'), got {C.dims}")

    unit_ids = C.coords["unit_id"].values
    if len(unit_ids) != len(np.unique(unit_ids)):
        raise ValueError(
            f"unit_id coordinates must be unique, but found {len(np.unique(unit_ids))} "
            f"unique values for {len(unit_ids)} units. "
            f"The zarr file has corrupted coordinates."
        )

    return C


def run_deconvolution(
    C_da: Any,
    unit_ids: list[int],
    fps: float,
    tau_rise: float,
    tau_decay: float,
    lam: float,
    baseline: float | str,
    s_min: float,
    engine: str = "fista",
    max_iters: int = 2000,
    tol: float = 1e-5,
    progress_bar: Any = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Deconvolve calcium traces with the selected engine (see :mod:`camap.deconv`).

    Both engines are driven by the indicator rise/decay time constants:
    ``"fista"`` builds a double-exponential kernel and solves a non-negative L1
    problem (no extra dependency); ``"oasis"`` converts the time constants to
    AR(2) coefficients and runs ``oasisAR2`` (requires ``oasis-deconv``).

    Parameters
    ----------
    C_da : xarray.DataArray
        Calcium traces with dimensions (unit_id, frame).
    unit_ids : list[int]
        List of unit IDs to process.
    fps : float
        Sampling rate (frames per second), used to build the kernel / AR(2) poles.
    tau_rise, tau_decay : float
        Indicator rise/decay time constants in seconds.
    lam : float
        Sparsity weight (L1). 0 disables the penalty.
    baseline : float or str
        Baseline correction applied before deconvolution. Use 'pXX' for a
        percentile (e.g. 'p10') or a numeric value (0 = none).
    s_min : float
        Minimum event size; recovered events below this are zeroed.
    engine : str
        ``"fista"`` (default) or ``"oasis"``.
    max_iters : int
        Maximum FISTA iterations per unit (FISTA engine only).
    tol : float
        Relative convergence tolerance (FISTA engine only).
    progress_bar : optional
        tqdm progress bar wrapper (e.g., tqdm.notebook.tqdm).

    Returns
    -------
    good_unit_ids : list[int]
        Unit IDs that were successfully deconvolved.
    S_list : list[np.ndarray]
        Spike trains.
    """
    if engine == "fista":
        kernel = build_kernel(tau_rise, tau_decay, fps)

        def deconv_one(y_corrected: np.ndarray) -> np.ndarray:
            s, _c, _b = fista_deconvolve(y_corrected, kernel, lam=lam, max_iters=max_iters, tol=tol)
            return s
    elif engine == "oasis":
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                from oasis.oasis_methods import oasisAR2
            except ImportError as exc:
                raise ImportError(_OASIS_INSTALL_HINT) from exc
        for w in caught:
            logger.warning(str(w.message))

        g1, g2 = tau_to_g(tau_rise, tau_decay, fps)

        def deconv_one(y_corrected: np.ndarray) -> np.ndarray:
            _c, s = oasisAR2(y_corrected, g1=g1, g2=g2, lam=lam, s_min=s_min)
            return np.asarray(s, dtype=float)
    else:
        raise ValueError(f"Unknown deconvolution engine {engine!r}; use 'fista' or 'oasis'.")

    good_unit_ids: list[int] = []
    S_list: list[np.ndarray] = []

    iterator = progress_bar(unit_ids) if progress_bar else unit_ids

    for uid in iterator:
        y = np.ascontiguousarray(C_da.sel(unit_id=uid).values, dtype=np.float64)

        # Baseline correction
        if isinstance(baseline, str) and baseline.startswith("p"):
            p = float(baseline[1:])
            b = float(np.percentile(y, p))
        else:
            b = float(baseline)

        y_corrected = y - b

        try:
            s = deconv_one(y_corrected)
            # OASIS applies s_min internally; apply the floor for FISTA too.
            if engine == "fista" and s_min > 0:
                s = np.where(s < s_min, 0.0, s)
            good_unit_ids.append(int(uid))
            S_list.append(np.asarray(s, dtype=float))
        except Exception:
            continue

    return good_unit_ids, S_list
