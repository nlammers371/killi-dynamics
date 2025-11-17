# msd_generic.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Literal

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MSDResult:
    """Container for per-entity MSD results."""
    entity_id: str | int
    lags: np.ndarray                # lag times (same for all)
    msd: np.ndarray                 # mean-squared displacement values
    alpha: float | None             # log-log slope (diffusive exponent)
    n_pairs: np.ndarray             # number of displacement pairs per lag


__all__ = ["MSDResult", "compute_msd_generic", "msd_summary_df"]


# ------------------------------------------------------------------------------

def _pairwise_displacements(
    times: np.ndarray,
    positions: np.ndarray,
    lag_minutes: Iterable[float],
    rel_tol: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute squared displacements for specified lag times.

    Returns:
        tau_eff  – effective lag time (N_lag,)
        msd_vals – mean squared displacement per lag (N_lag,)
        n_pairs  – number of pairs per lag
    """
    times = np.asarray(times, float)
    pos = np.asarray(positions, float)
    lags = np.asarray(sorted(lag_minutes), float)

    n_lags = lags.size
    msd_vals = np.full(n_lags, np.nan, float)
    tau_eff = np.full(n_lags, np.nan, float)
    n_pairs = np.zeros(n_lags, int)

    n = len(times)
    if n < 2:
        return tau_eff, msd_vals, n_pairs

    for i, tau in enumerate(lags):
        dt_low, dt_high = tau * (1 - rel_tol), tau * (1 + rel_tol)
        disp_list = []
        dt_list = []
        for j in range(n - 1):
            # Find subsequent points within the allowed lag interval
            dt = times[j + 1:] - times[j]
            mask = (dt >= dt_low) & (dt <= dt_high)
            if not np.any(mask):
                continue
            dpos = pos[j + 1:][mask] - pos[j]
            disp_list.append(np.sum(dpos**2, axis=1))
            dt_list.append(dt[mask])

        if not disp_list:
            continue
        disp_all = np.concatenate(disp_list)
        dt_all = np.concatenate(dt_list)
        msd_vals[i] = np.mean(disp_all)
        tau_eff[i] = np.mean(dt_all)
        n_pairs[i] = disp_all.size

    return tau_eff, msd_vals, n_pairs


# ------------------------------------------------------------------------------

def compute_msd_generic(
    tracks: pd.DataFrame,
    *,
    time_col: str = "t",
    xyz_cols: tuple[str, str, str] = ("x", "y", "z"),
    id_col: str = "track_id",
    group_col: Optional[str] = None,
    lag_minutes: Iterable[float] = (1, 2, 4, 8, 16, 32, 64),
    rel_tol: float = 0.25,
    min_pairs: int = 3,
) -> list[MSDResult]:
    """
    Compute mean-squared displacement (MSD) and diffusive exponent α
    for individual tracks or grouped tracks.

    Parameters
    ----------
    tracks : DataFrame
        Must contain time, position columns, and an identifier column.
    group_col : str, optional
        If provided, compute MSD pooled over all tracks sharing the same group ID.
        If None, compute per-track.
    lag_minutes : iterable of float
        Target lag times (same units as `time_col`).
    rel_tol : float
        Accept ±rel_tol fractional deviation in Δt for matching pairs.
    min_pairs : int
        Minimum number of displacement pairs to estimate α for a given entity.

    Returns
    -------
    results : list[MSDResult]
        One entry per entity (track or group).
    """

    required = {time_col, *xyz_cols, id_col}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Determine grouping variable
    if group_col is None:
        group_key = id_col
    else:
        group_key = group_col

    results: list[MSDResult] = []

    for gid, group in tracks.sort_values([group_key, time_col]).groupby(group_key):
        if group.shape[0] < 2:
            continue

        # Merge all tracks if group_col provided
        if group_col is not None:
            # group may include several tracks (different track_ids)
            times = group[time_col].to_numpy(float)
            pos = group[list(xyz_cols)].to_numpy(float)
        else:
            times = group[time_col].to_numpy(float)
            pos = group[list(xyz_cols)].to_numpy(float)

        tau_eff, msd_vals, n_pairs = _pairwise_displacements(times, pos, lag_minutes, rel_tol=rel_tol)

        # fit α (slope of log-log MSD vs τ)
        valid = (n_pairs >= min_pairs) & np.isfinite(tau_eff) & np.isfinite(msd_vals) & (tau_eff > 0) & (msd_vals > 0)
        alpha = np.nan
        if np.count_nonzero(valid) >= 2:
            x = np.log(tau_eff[valid])
            y = np.log(msd_vals[valid])
            slope, _ = np.polyfit(x, y, 1)
            alpha = float(slope)

        results.append(
            MSDResult(
                entity_id=gid,
                lags=np.asarray(lag_minutes, float),
                msd=msd_vals,
                alpha=alpha,
                n_pairs=n_pairs,
            )
        )

    return results


# ------------------------------------------------------------------------------

def msd_summary_df(results: list[MSDResult]) -> pd.DataFrame:
    """
    Flatten MSDResult list into a tidy summary DataFrame:
    one row per (entity, lag).
    """
    records = []
    for r in results:
        for tau, msd, npair in zip(r.lags, r.msd, r.n_pairs):
            records.append({
                "entity_id": r.entity_id,
                "lag_min": tau,
                "msd": msd,
                "alpha": r.alpha,
                "n_pairs": npair,
            })
    return pd.DataFrame.from_records(records)
