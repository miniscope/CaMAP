"""Behavior data spatial corrections and DLC loading."""

from pathlib import Path

import numpy as np
import pandas as pd

from camap.dataset_validation import hampel_mask
from camap.log import init_logger

logger = init_logger(__name__)


def _load_behavior_xy(
    csv_path: Path,
    bodypart: str,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Load DeepLabCut-style behavior CSV and return x/y coordinates per frame.

    Parameters
    ----------
    csv_path:
        Path to DeepLabCut CSV file with multi-index header.
    bodypart:
        Body part name to extract (e.g. 'LED').
    x_col:
        Coordinate column name for the x-axis (default 'x').
    y_col:
        Coordinate column name for the y-axis (default 'y').
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    scorer = None
    for col in df.columns[1:]:
        if col[1] == bodypart and col[2] == x_col:
            scorer = col[0]
            break

    if scorer is None:
        available_bodyparts = {col[1] for col in df.columns[1:]}
        raise ValueError(
            f"Bodypart '{bodypart}' not found in CSV. "
            f"Available bodyparts: {sorted(available_bodyparts)}"
        )

    x = df[(scorer, bodypart, x_col)]
    y = df[(scorer, bodypart, y_col)]
    frame_index = df.iloc[:, 0]

    return pd.DataFrame({"frame_index": frame_index, "x": x, "y": y})


def remove_position_jumps(
    positions: pd.DataFrame,
    window_frames: int = 7,
    n_sigmas: float = 3.0,
) -> tuple[pd.DataFrame, int]:
    """Replace implausible position jumps with linear interpolation (Hampel filter).

    For each frame the local centroid is the (median x, median y) over a
    centered window of ``window_frames`` frames. The deviation is the
    Euclidean distance from the frame to its centroid; the local scale is
    the rolling median of those deviations. A frame is flagged when its
    deviation exceeds ``n_sigmas * 1.4826 * scale`` — the standard Hampel
    rule (Hampel 1974) generalized to 2D via the spatial median.

    Flagged frames have their x/y replaced by linear interpolation from the
    surrounding good frames.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y`` (and any others, preserved).
    window_frames:
        Window size for the rolling median centroid and MAD.  Should be odd
        and large enough to span typical glitch durations.
    n_sigmas:
        Number of (MAD-based) standard deviations beyond which a frame is
        treated as an outlier.  3.0 corresponds to a ~99.7% Gaussian band.

    Returns
    -------
    tuple of (DataFrame with jumps interpolated, number of frames fixed).
    """
    if window_frames < 3:
        raise ValueError("window_frames must be >= 3.")

    df = positions.copy()
    x = df["x"].astype(float)
    y = df["y"].astype(float)

    min_periods = window_frames // 2 + 1
    x_med = x.rolling(window_frames, center=True, min_periods=min_periods).median()
    y_med = y.rolling(window_frames, center=True, min_periods=min_periods).median()
    deviation = pd.Series(np.hypot(x - x_med, y - y_med))
    bad = hampel_mask(deviation, window=window_frames, n_sigmas=n_sigmas)

    n_bad = int(bad.sum())
    if n_bad > 0:
        x_clean = x.to_numpy(copy=True)
        y_clean = y.to_numpy(copy=True)
        x_clean[bad] = np.nan
        y_clean[bad] = np.nan
        df["x"] = pd.Series(x_clean).interpolate(limit_direction="both").to_numpy()
        df["y"] = pd.Series(y_clean).interpolate(limit_direction="both").to_numpy()

    return df, n_bad


def correct_perspective(
    positions: pd.DataFrame,
    arena_bounds: tuple[float, float, float, float],
    camera_height_mm: float,
    tracking_height_mm: float,
) -> pd.DataFrame:
    """Correct perspective distortion from overhead camera parallax.

    An LED at height *h* above the floor appears shifted radially outward
    from the optical axis.  The corrected position is::

        x_corrected = cx + (x - cx) * (H - h) / H

    where *cx, cy* is the arena center (midpoint of *arena_bounds*),
    *H* is the camera height, and *h* is the tracking height.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y``.
    arena_bounds:
        (x_min, x_max, y_min, y_max) in pixels.
    camera_height_mm:
        Camera height above floor in mm.
    tracking_height_mm:
        Tracked point height above floor in mm.

    Returns
    -------
    DataFrame with corrected ``x``, ``y``.
    """
    x_min, x_max, y_min, y_max = arena_bounds
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    factor = (camera_height_mm - tracking_height_mm) / camera_height_mm

    df = positions.copy()
    df["x"] = cx + (df["x"] - cx) * factor
    df["y"] = cy + (df["y"] - cy) * factor
    return df


def clip_to_arena(
    positions: pd.DataFrame,
    arena_bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Clip positions to arena boundaries.

    Points outside the arena (from detection errors) are clamped to the
    nearest boundary edge. The number of clamped frames and the maximum
    out-of-bounds deviation are logged at INFO so the silent-data-repair
    warning of the review's "no silent data repair" policy is satisfied.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y``.
    arena_bounds:
        (x_min, x_max, y_min, y_max) in pixels.

    Returns
    -------
    DataFrame with ``x``, ``y`` clipped to arena bounds.
    """
    x_min, x_max, y_min, y_max = arena_bounds
    x = positions["x"].to_numpy(dtype=float)
    y = positions["y"].to_numpy(dtype=float)

    # Per-axis out-of-bounds amount (zero when in-bounds, NaN propagates).
    dx = np.maximum(0.0, np.maximum(x_min - x, x - x_max))
    dy = np.maximum(0.0, np.maximum(y_min - y, y - y_max))
    out_of_bounds = (dx > 0) | (dy > 0)
    n_clamped = int(np.sum(out_of_bounds))

    if n_clamped > 0:
        max_dev = float(np.nanmax(np.where(out_of_bounds, np.hypot(dx, dy), 0.0)))
        n_total = len(positions)
        logger.info(
            "clip_to_arena: %d/%d frames clamped (max deviation %.2f px from boundary).",
            n_clamped,
            n_total,
            max_dev,
        )

    df = positions.copy()
    df["x"] = np.clip(x, x_min, x_max)
    df["y"] = np.clip(y, y_min, y_max)
    return df
