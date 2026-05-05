"""1D spatial analysis functions for place cells in linear tracks / arms."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from camap.config import SpatialMap1DConfig


def gaussian_filter_normalized_1d(
    data: np.ndarray,
    sigma: float,
    segment_bins: list[int] | None = None,
) -> np.ndarray:
    """Apply 1D Gaussian smoothing with boundary normalization.

    Uses zero-padding and normalizes by the kernel weight sum so that
    edge bins are not penalized.

    Parameters
    ----------
    data:
        1D array to smooth.
    sigma:
        Gaussian smoothing sigma in bins.
    segment_bins:
        Bin boundary indices for independent segments (e.g.
        ``[0, 50, 100, 200, 300]`` for 4 segments of varying length).
        Each segment ``[segment_bins[i]:segment_bins[i+1]]`` is smoothed
        independently.  When *None* the whole array is smoothed as one.
    """
    if sigma <= 0:
        return data.copy()
    if segment_bins is None or len(segment_bins) <= 2:
        smoothed = gaussian_filter1d(data, sigma=sigma, mode="constant", cval=0)
        norm = gaussian_filter1d(np.ones_like(data), sigma=sigma, mode="constant", cval=0)
        norm[norm == 0] = 1
        return smoothed / norm

    # Smooth each segment independently
    result = np.empty_like(data)
    for i in range(len(segment_bins) - 1):
        s = segment_bins[i]
        e = segment_bins[i + 1]
        seg = data[s:e]
        smoothed = gaussian_filter1d(seg, sigma=sigma, mode="constant", cval=0)
        norm = gaussian_filter1d(np.ones_like(seg), sigma=sigma, mode="constant", cval=0)
        norm[norm == 0] = 1
        result[s:e] = smoothed / norm
    return result


def _draw_shift(rng: np.random.RandomState, n_frames: int, min_shift_frames: int) -> int:
    """Draw a circular-shift offset for a shuffle iteration."""
    if min_shift_frames > 0:
        return int(rng.randint(min_shift_frames, n_frames - min_shift_frames))
    return int(rng.randint(1, n_frames))


def _shuffled_rate_map_1d(
    traj_pos: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    occ_for_division: np.ndarray,
    valid_mask: np.ndarray,
    spatial_sigma: float,
    segment_bins: list[int] | None,
) -> np.ndarray:
    """One 1D shuffle iteration: histogram weighted events, optionally
    smooth the numerator, divide by ``occ_for_division`` on ``valid_mask``.

    ``occ_for_division`` is the loop-invariant denominator the caller has
    already prepared — typically ``gaussian_filter_normalized_1d(occ, sigma)``
    when smoothing, or raw ``occupancy_time`` when ``spatial_sigma == 0``.
    """
    event_weights, _ = np.histogram(traj_pos, bins=edges, weights=weights)
    event_weights = event_weights.astype(float)
    if spatial_sigma > 0:
        event_weights = gaussian_filter_normalized_1d(
            event_weights,
            sigma=spatial_sigma,
            segment_bins=segment_bins,
        )
    rate = np.zeros_like(occ_for_division)
    rate[valid_mask] = event_weights[valid_mask] / occ_for_division[valid_mask]
    return rate


def compute_occupancy_map_1d(
    trajectory_df: pd.DataFrame,
    n_bins: int,
    pos_range: tuple[float, float],
    behavior_fps: float,
    spatial_sigma: float = 1.0,
    min_occupancy: float = 0.1,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1D occupancy histogram (raw).

    Parameters
    ----------
    trajectory_df:
        Speed-filtered trajectory with pos_column.
    n_bins:
        Total number of spatial bins across all arms.
    pos_range:
        (min, max) of the 1D position axis.
    behavior_fps:
        Behavior sampling rate.
    spatial_sigma:
        Gaussian smoothing sigma in bins. Used only to compute
        ``valid_mask``; ``occupancy_time`` is returned unsmoothed so
        downstream smoothing in the rate-map/SI path doesn't compound
        to sigma·sqrt(2).
    min_occupancy:
        Minimum occupancy in seconds (checked against the smoothed
        occupancy for the mask).
    pos_column:
        Column name for position values.
    segment_bins:
        Bin boundary indices for per-segment smoothing.

    Returns
    -------
    tuple
        (occupancy_time, valid_mask, edges) — ``occupancy_time`` is raw
        (seconds per bin); ``valid_mask`` comes from the smoothed copy.
    """
    edges = np.linspace(pos_range[0], pos_range[1], n_bins + 1)
    time_per_frame = 1.0 / behavior_fps

    counts, _ = np.histogram(trajectory_df[pos_column], bins=edges)
    occupancy_time = counts.astype(float) * time_per_frame

    if spatial_sigma > 0:
        occ_for_mask = gaussian_filter_normalized_1d(
            occupancy_time, sigma=spatial_sigma, segment_bins=segment_bins
        )
    else:
        occ_for_mask = occupancy_time
    valid_mask = occ_for_mask >= min_occupancy

    return occupancy_time, valid_mask, edges


def compute_rate_map_1d(
    unit_events: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    spatial_sigma: float = 1.0,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute smoothed 1D rate map.

    When ``normalize`` is ``True`` (default) the map is divided by its
    peak so values span 0-1 (for display). Set ``normalize=False`` to
    keep firing-rate units (events·s⁻¹ per bin) for quantitative
    analyses such as population-vector overlap.
    """
    if unit_events.empty:
        return np.full_like(occupancy_time, np.nan)

    event_weights, _ = np.histogram(
        unit_events[pos_column],
        bins=edges,
        weights=unit_events["s"],
    )
    event_weights = event_weights.astype(float)
    # Smooth numerator and denominator independently (Skaggs 1996).
    event_smooth = gaussian_filter_normalized_1d(
        event_weights, sigma=spatial_sigma, segment_bins=segment_bins
    )
    occ_smooth = gaussian_filter_normalized_1d(
        occupancy_time, sigma=spatial_sigma, segment_bins=segment_bins
    )
    rate_map_smoothed = np.zeros_like(occupancy_time)
    rate_map_smoothed[valid_mask] = event_smooth[valid_mask] / occ_smooth[valid_mask]

    if normalize:
        valid_rate_values = rate_map_smoothed[valid_mask]
        if len(valid_rate_values) > 0 and np.nanmax(valid_rate_values) > 0:
            rate_map_smoothed[valid_mask] = rate_map_smoothed[valid_mask] / np.nanmax(
                valid_rate_values
            )

    rate_map_smoothed[~valid_mask] = np.nan
    return rate_map_smoothed


def compute_raw_rate_map_1d(
    unit_events: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    pos_column: str = "pos_1d",
) -> np.ndarray:
    """Compute unsmoothed 1D rate map."""
    if unit_events.empty:
        return np.full_like(occupancy_time, np.nan)
    event_weights, _ = np.histogram(
        unit_events[pos_column],
        bins=edges,
        weights=unit_events["s"],
    )
    event_weights = event_weights.astype(float)
    rate_map = np.full_like(occupancy_time, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]
    return rate_map


def compute_spatial_information_1d(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    n_shuffles: int = 100,
    random_seed: int | None = None,
    min_shift_seconds: float = 0.0,
    behavior_fps: float = 20.0,
    si_weight_mode: str = "amplitude",
    spatial_sigma: float = 0.0,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> tuple[float, float, np.ndarray]:
    """Compute 1D spatial information with shuffle significance test.

    Same SI formula as 2D, applied to 1D bins. Circular shift shuffle.

    Returns
    -------
    tuple: (spatial_info, p_value, shuffled_sis)
    """
    rng = np.random.RandomState(random_seed)

    if unit_events.empty:
        return 0.0, 1.0, np.zeros(n_shuffles)

    weights = np.ones(len(unit_events)) if si_weight_mode == "binary" else unit_events["s"].values

    event_weights, _ = np.histogram(
        unit_events[pos_column],
        bins=edges,
        weights=weights,
    )
    event_weights = event_weights.astype(float)
    # Smooth numerator and denominator independently (Skaggs 1996).
    # occ_smooth is loop-invariant; reused inside the shuffle loop below.
    if spatial_sigma > 0:
        occ_smooth = gaussian_filter_normalized_1d(
            occupancy_time, sigma=spatial_sigma, segment_bins=segment_bins
        )
        event_smooth = gaussian_filter_normalized_1d(
            event_weights, sigma=spatial_sigma, segment_bins=segment_bins
        )
        rate_map = np.zeros_like(occupancy_time)
        rate_map[valid_mask] = event_smooth[valid_mask] / occ_smooth[valid_mask]
    else:
        rate_map = np.zeros_like(occupancy_time)
        rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]

    total_time = np.sum(occupancy_time[valid_mask])

    if total_time <= 0 or np.sum(event_weights[valid_mask]) <= 0:
        return 0.0, 1.0, np.zeros(n_shuffles)

    # Compute overall rate from smoothed map so Σ P_i × (λ_i/λ̄) = 1
    overall_lambda = np.sum(rate_map[valid_mask] * occupancy_time[valid_mask]) / total_time
    P_i = np.zeros_like(occupancy_time)
    P_i[valid_mask] = occupancy_time[valid_mask] / total_time

    if overall_lambda <= 0:
        return 0.0, 1.0, np.zeros(n_shuffles)

    valid_si = (rate_map > 0) & valid_mask
    if np.any(valid_si):
        ratio = rate_map[valid_si] / overall_lambda
        si_term = P_i[valid_si] * ratio * np.log2(ratio)
        actual_si = float(np.sum(si_term))
    else:
        actual_si = 0.0

    # Shuffling test
    traj_frames = trajectory_df["frame_index"].values
    if si_weight_mode == "binary":
        u_grouped = unit_events.groupby("frame_index").size()
    else:
        u_grouped = unit_events.groupby("frame_index")["s"].sum()
    aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

    n_frames = len(aligned_events)
    min_shift_frames = int(min_shift_seconds * behavior_fps)
    # Drop the constraint if the session is too short to leave a
    # non-empty shift range; the unrestricted branch below is used.
    if min_shift_frames >= n_frames // 2:
        min_shift_frames = 0

    traj_pos = trajectory_df[pos_column].values
    occ_for_division = occ_smooth if spatial_sigma > 0 else occupancy_time

    shuffled_sis = np.empty(n_shuffles)
    for i in range(n_shuffles):
        s_shuffled = np.roll(aligned_events, _draw_shift(rng, n_frames, min_shift_frames))
        rate_shuf = _shuffled_rate_map_1d(
            traj_pos,
            s_shuffled,
            edges,
            occ_for_division,
            valid_mask,
            spatial_sigma,
            segment_bins,
        )
        shuf_lambda = np.sum(rate_shuf[valid_mask] * occupancy_time[valid_mask]) / total_time
        valid_s = (rate_shuf > 0) & valid_mask & (shuf_lambda > 0)
        if np.any(valid_s) and shuf_lambda > 0:
            ratio_s = rate_shuf[valid_s] / shuf_lambda
            shuffled_sis[i] = np.sum(P_i[valid_s] * ratio_s * np.log2(ratio_s))
        else:
            shuffled_sis[i] = 0.0

    p_val = (np.sum(shuffled_sis >= actual_si) + 1) / (n_shuffles + 1)

    return actual_si, p_val, shuffled_sis


def compute_stability_score_1d(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    spatial_sigma: float = 1.0,
    behavior_fps: float = 20.0,
    min_occupancy: float = 0.1,
    n_split_blocks: int = 10,
    block_shift: float = 0.0,
    n_shuffles: int = 0,
    random_seed: int | None = None,
    min_shift_seconds: float = 0.0,
    si_weight_mode: str = "amplitude",
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1D split-half stability test.

    Same block-splitting approach as 2D, with 1D rate maps.
    ``n_split_blocks=2`` reduces to a first-half/second-half split
    (sensitive to session-long drift); ``n_split_blocks>=4`` interleaves
    the halves so within-session drift averages out. Combining both
    (e.g. ``stability_splits=[2, 10]``) tests stability at two different
    timescales.

    Returns
    -------
    tuple: (correlation, fisher_z, stability_p_val,
            rate_map_first, rate_map_second, shuffled_corrs)
    """
    if unit_events.empty or trajectory_df.empty:
        nan_map = np.full_like(occupancy_time, np.nan)
        return np.nan, np.nan, np.nan, nan_map, nan_map, np.array([])

    if si_weight_mode == "binary":
        unit_events = unit_events.copy()
        unit_events["s"] = 1.0

    # Split trajectory into interleaved temporal blocks
    all_frames = trajectory_df["frame_index"].values
    frame_min = all_frames.min()
    frame_max = all_frames.max()
    span = frame_max - frame_min
    if span == 0:
        nan_map = np.full_like(occupancy_time, np.nan)
        return np.nan, np.nan, np.nan, nan_map, nan_map, np.array([])
    block_width = span / n_split_blocks
    if not 0.0 <= block_shift < 1.0:
        raise ValueError(f"block_shift must be in [0, 1); got {block_shift}.")
    offset = block_shift * block_width

    traj_block_ids = np.floor((all_frames - frame_min - offset) / block_width).astype(int)
    traj_block_ids = np.clip(traj_block_ids, 0, n_split_blocks - 1)
    traj_first_mask = traj_block_ids % 2 == 0
    traj_second_mask = ~traj_first_mask

    event_frames = unit_events["frame_index"].values
    event_block_ids = np.floor((event_frames - frame_min - offset) / block_width).astype(int)
    event_block_ids = np.clip(event_block_ids, 0, n_split_blocks - 1)
    events_first_mask = event_block_ids % 2 == 0
    events_second_mask = ~events_first_mask

    traj_first = trajectory_df[traj_first_mask]
    traj_second = trajectory_df[traj_second_mask]
    events_first = unit_events[events_first_mask]
    events_second = unit_events[events_second_mask]

    time_per_frame = 1.0 / behavior_fps

    def compute_half_occupancy(traj_half: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return raw half-occupancy and a smoothed-occupancy validity mask."""
        if traj_half.empty:
            return np.zeros_like(occupancy_time), np.zeros_like(valid_mask, dtype=bool)
        counts, _ = np.histogram(traj_half[pos_column], bins=edges)
        occ = counts.astype(float) * time_per_frame
        occ_smooth = gaussian_filter_normalized_1d(
            occ, sigma=spatial_sigma, segment_bins=segment_bins
        )
        mask = occ_smooth >= min_occupancy
        return occ, mask

    occ_first, valid_first = compute_half_occupancy(traj_first)
    occ_second, valid_second = compute_half_occupancy(traj_second)

    def compute_half_rate_map(
        events: pd.DataFrame, occ: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        if events.empty or not np.any(mask):
            return np.full_like(occ, np.nan)
        event_weights, _ = np.histogram(
            events[pos_column],
            bins=edges,
            weights=events["s"],
        )
        event_weights = event_weights.astype(float)
        # Smooth numerator and denominator independently (Skaggs 1996).
        event_smooth = gaussian_filter_normalized_1d(
            event_weights, sigma=spatial_sigma, segment_bins=segment_bins
        )
        occ_smooth_half = gaussian_filter_normalized_1d(
            occ, sigma=spatial_sigma, segment_bins=segment_bins
        )
        rate_map_smoothed = np.zeros_like(occ)
        rate_map_smoothed[mask] = event_smooth[mask] / occ_smooth_half[mask]
        rate_map_smoothed[~mask] = np.nan
        return rate_map_smoothed

    rate_map_first = compute_half_rate_map(events_first, occ_first, valid_first)
    rate_map_second = compute_half_rate_map(events_second, occ_second, valid_second)

    both_valid = valid_first & valid_second
    if not np.any(both_valid):
        return np.nan, np.nan, np.nan, rate_map_first, rate_map_second, np.array([])

    vals_first = rate_map_first[both_valid]
    vals_second = rate_map_second[both_valid]

    finite_mask = np.isfinite(vals_first) & np.isfinite(vals_second)
    if np.sum(finite_mask) < 3:
        return np.nan, np.nan, np.nan, rate_map_first, rate_map_second, np.array([])

    vals_first = vals_first[finite_mask]
    vals_second = vals_second[finite_mask]

    corr = np.corrcoef(vals_first, vals_second)[0, 1]
    # Clip before arctanh: r=±1 maps to ±inf otherwise.
    corr_clipped = np.clip(corr, -0.9999, 0.9999)
    fisher_z = np.arctanh(corr_clipped)

    if n_shuffles > 0:
        # Seed offset gives an independent stream per (SI vs stability) and per
        # n_split_blocks; without the latter, multi-split tests on one unit
        # would replay identical shifts and produce correlated null p-values.
        if random_seed is not None:
            stab_seed = random_seed + 224737 + 7919 * n_split_blocks
        else:
            stab_seed = None
        rng = np.random.RandomState(stab_seed)

        traj_frames = trajectory_df["frame_index"].values
        u_grouped = unit_events.groupby("frame_index")["s"].sum()
        aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

        traj_pos = trajectory_df[pos_column].values
        # Loop-invariant: hoist masked trajectory subsets and smoothed occupancies.
        traj_pos_first = traj_pos[traj_first_mask]
        traj_pos_second = traj_pos[traj_second_mask]
        os1 = gaussian_filter_normalized_1d(
            occ_first, sigma=spatial_sigma, segment_bins=segment_bins
        )
        os2 = gaussian_filter_normalized_1d(
            occ_second, sigma=spatial_sigma, segment_bins=segment_bins
        )
        bv = valid_first & valid_second

        n_frames = len(aligned_events)
        min_shift_frames = int(min_shift_seconds * behavior_fps)
        # Drop the constraint if the session is too short to leave a
        # non-empty shift range; the unrestricted branch below is used.
        if min_shift_frames >= n_frames // 2:
            min_shift_frames = 0

        shuffled_corrs = np.empty(n_shuffles)
        for i in range(n_shuffles):
            shifted = np.roll(aligned_events, _draw_shift(rng, n_frames, min_shift_frames))
            rm1 = _shuffled_rate_map_1d(
                traj_pos_first,
                shifted[traj_first_mask],
                edges,
                os1,
                valid_first,
                spatial_sigma,
                segment_bins,
            )
            rm2 = _shuffled_rate_map_1d(
                traj_pos_second,
                shifted[traj_second_mask],
                edges,
                os2,
                valid_second,
                spatial_sigma,
                segment_bins,
            )

            if not np.any(bv):
                shuffled_corrs[i] = 0.0
                continue
            v1, v2 = rm1[bv], rm2[bv]
            fm = np.isfinite(v1) & np.isfinite(v2)
            if np.sum(fm) < 3:
                shuffled_corrs[i] = 0.0
                continue
            shuffled_corrs[i] = np.corrcoef(v1[fm], v2[fm])[0, 1]

        stability_p_val = float((np.sum(shuffled_corrs >= corr) + 1) / (n_shuffles + 1))
    else:
        stability_p_val = np.nan
        shuffled_corrs = np.array([])

    return corr, fisher_z, stability_p_val, rate_map_first, rate_map_second, shuffled_corrs


def compute_unit_analysis_1d(
    unit_id: int,
    df_filtered: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    scfg: SpatialMap1DConfig,
    behavior_fps: float,
    random_seed: int | None = None,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> dict:
    """Compute 1D rate map, SI, stability, and place field for a unit.

    Returns same dict keys as the 2D ``compute_unit_analysis``; see that
    function for the ``min_events`` gate semantics.
    """
    unit_data = (
        df_filtered[df_filtered["unit_id"] == unit_id] if not df_filtered.empty else pd.DataFrame()
    )

    # rate_map_smoothed: firing-rate units, authoritative for SI/PVO/etc.
    # rate_map_raw: unsmoothed, used for overall-rate computation below.
    # (Peak-normalized display map is a derived @property on UnitResult.)
    rate_map_smoothed = compute_rate_map_1d(
        unit_data,
        occupancy_time,
        valid_mask,
        edges,
        scfg.spatial_sigma,
        pos_column,
        segment_bins=segment_bins,
        normalize=False,
    )
    rate_map_raw = compute_raw_rate_map_1d(unit_data, occupancy_time, valid_mask, edges, pos_column)

    # Overall rate: amplitude-weighted and binary event count
    valid_occ = valid_mask & (occupancy_time > 0)
    total_time = float(np.sum(occupancy_time[valid_occ])) if np.any(valid_occ) else 0.0
    if total_time > 0:
        overall_rate = float(
            np.sum(rate_map_raw[valid_occ] * occupancy_time[valid_occ]) / total_time
        )
        event_count_rate = float(len(unit_data)) / total_time
    else:
        overall_rate = 0.0
        event_count_rate = 0.0

    below_gate = scfg.min_events > 0 and len(unit_data) < scfg.min_events

    if below_gate:
        si = 0.0
        p_val = 1.0
        shuffled_sis = np.zeros(scfg.n_shuffles)
    else:
        si, p_val, shuffled_sis = compute_spatial_information_1d(
            unit_data,
            trajectory_df,
            occupancy_time,
            valid_mask,
            edges,
            scfg.n_shuffles,
            random_seed=random_seed,
            min_shift_seconds=scfg.min_shift_seconds,
            behavior_fps=behavior_fps,
            si_weight_mode=scfg.si_weight_mode,
            spatial_sigma=scfg.spatial_sigma,
            pos_column=pos_column,
            segment_bins=segment_bins,
        )

    # Visualization events: speed-filtered only (no amplitude threshold)
    events_above = unit_data
    vis_threshold = 0.0

    # One stability test per configured split; half rate maps stay in
    # firing-rate units so display code can normalize them with the full map.
    stability_splits: list[dict] = []
    for n_splits in scfg.stability_splits:
        if below_gate:
            nan_map = np.full_like(occupancy_time, np.nan)
            s_corr = np.nan
            s_z = np.nan
            s_p = 1.0
            rm_first = nan_map
            rm_second = nan_map.copy()
            shuffled_s = np.array([])
        else:
            (
                s_corr,
                s_z,
                s_p,
                rm_first,
                rm_second,
                shuffled_s,
            ) = compute_stability_score_1d(
                unit_data,
                trajectory_df,
                occupancy_time,
                valid_mask,
                edges,
                spatial_sigma=scfg.spatial_sigma,
                behavior_fps=behavior_fps,
                min_occupancy=scfg.min_occupancy,
                n_split_blocks=n_splits,
                block_shift=scfg.block_shift,
                n_shuffles=scfg.n_shuffles,
                random_seed=random_seed,
                min_shift_seconds=scfg.min_shift_seconds,
                si_weight_mode=scfg.si_weight_mode,
                pos_column=pos_column,
                segment_bins=segment_bins,
            )
        stability_splits.append(
            {
                "n_split_blocks": n_splits,
                "corr": s_corr,
                "fisher_z": s_z,
                "p_val": s_p,
                "shuffled_corrs": shuffled_s,
                "rate_map_first": rm_first,
                "rate_map_second": rm_second,
            }
        )

    return {
        "rate_map_smoothed": rate_map_smoothed,
        "rate_map_raw": rate_map_raw,
        "overall_rate": overall_rate,
        "event_count_rate": event_count_rate,
        "si": si,
        "p_val": p_val,
        "shuffled_sis": shuffled_sis,
        "events_above_threshold": events_above,
        "vis_threshold": vis_threshold,
        "unit_data": unit_data,
        "stability_splits": stability_splits,
    }
