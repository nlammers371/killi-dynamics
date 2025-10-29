# msd_modern.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .config import WindowConfig
from .grids import GridBinResult
from .vector_field import StepTable

@dataclass(slots=True)
class MSDResult:
    """Aggregate MSD scalars for each HEALPix grid."""
    nside: int
    time_centers: np.ndarray                 # (T,)
    msd_alpha: np.ndarray                    # (T, Npix) slope of log(MSD) vs log(tau)
    msd_value: np.ndarray                    # (T, Npix) MSD at longest valid lag

__all__ = ["MSDResult", "compute_msd_metrics"]

def _pair_indices_by_time(
    times: np.ndarray,                      # (N,)
    ids: np.ndarray,                        # (N,)
    target_tau: float,                      # minutes
    rel_tol: float = 0.25,                  # accept [ (1-rel_tol)τ, (1+rel_tol)τ ]
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each step midpoint i, find j>i in the same track with Δt≈target_tau (within relative tolerance).
    Returns arrays of start and end indices (possibly empty).
    """
    if times.size < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    t = times
    # Two-pointer over a stable sort by (track, time)
    order = np.lexsort((t, ids))
    t_o = t[order]
    id_o = ids[order]

    start_idx = []
    end_idx = []

    low = (1.0 - rel_tol) * target_tau
    high = (1.0 + rel_tol) * target_tau

    # walk each track separately
    u, first = np.unique(id_o, return_index=True)
    # append sentinel for last track
    bounds = np.append(first, t_o.size)

    for k in range(u.size):
        a, b = bounds[k], bounds[k+1]
        if b - a < 2:
            continue
        ti = t_o[a:b]

        j = 1
        for i in range(b - a):
            # advance j until Δt >= low
            while j < (b - a) and (ti[j] - ti[i]) < low:
                j += 1
            if j >= (b - a):
                break
            # collect any j with Δt in [low, high]
            j2 = j
            while j2 < (b - a) and low <= (ti[j2] - ti[i]) <= high:
                start_idx.append(order[a + i])
                end_idx.append(order[a + j2])
                j2 += 1

    if not start_idx:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.asarray(start_idx, dtype=int), np.asarray(end_idx, dtype=int)

def _great_circle_sqdist(
    u1: np.ndarray,         # (M,3) unit vectors at start
    u2: np.ndarray,         # (M,3) unit vectors at end
    r1: np.ndarray,         # (M,) radii at start
    r2: np.ndarray,         # (M,) radii at end
) -> np.ndarray:
    """
    Squared geodesic distance on the sphere between directions u1 and u2 with
    arc length based on pairwise mean radius.
    """
    # robust angle via atan2(||u1×u2||, u1·u2)
    dot = np.einsum("ij,ij->i", u1, u2)
    cross = np.linalg.norm(np.cross(u1, u2), axis=1)
    dpsi = np.arctan2(cross, np.clip(dot, -1.0, 1.0))  # radians
    rbar = 0.5 * (r1 + r2)
    return (rbar * dpsi) ** 2

def compute_msd_metrics(
    tracks: pd.DataFrame,
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    step_table: StepTable | None = None,
    *,
    lag_minutes: Iterable[float] = (1, 2, 4, 8, 16, 32, 64),  # short→long to probe confinement
    lag_rel_tol: float = 0.25,                                # ±25% tolerance for Δt matching
    min_pairs_per_lag: int = 5,                               # per pixel/time minimum
) -> dict[int, MSDResult]:
    """
    Compute ensemble MSD and scaling α per HEALPix pixel/time using the StepTable:
      • embryo-centered geometry (already in StepTable),
      • spherical great-circle distances for displacements,
      • lags specified in MINUTES (with tolerance),
      • aggregates by start-step pixel inside each time window.

    Returns MSD at the longest valid lag and α (slope over valid lags).
    """
    if step_table is None:
        # Must be the embryo-centered version you just fixed.
        # It provides: mid_positions (N,3), unit_mid (N,3), radii (N,), mid_times (N,), track_ids (N,)
        from .vector_field import build_step_table
        step_table = build_step_table(tracks)

    results: dict[int, MSDResult] = {}

    if step_table.mid_times.size == 0:
        for nside, grid_result in binned.items():
            nt, npix = grid_result.counts.shape
            nan = np.full((nt, npix), np.nan, dtype=np.float32)
            results[nside] = MSDResult(nside, grid_result.time_centers, msd_alpha=nan.copy(), msd_value=nan.copy())
        return results

    # Convert lags to a monotone numpy array
    tau_targets = np.asarray(sorted(lag_minutes), dtype=float)

    # Precompute per-nside pixel assignment for step midpoints
    per_nside_pix = {n: step_table.pixel_indices(n) for n in binned.keys()}

    # Convenience handles
    times_all = step_table.mid_times       # minutes
    unit_all  = step_table.unit_mid        # (N,3)
    radii_all = step_table.radii           # (N,)
    ids_all   = step_table.track_ids       # (N,)
    pos_all   = step_table.mid_positions   # (N,3) embryo-centered (not directly used for distance)

    for nside, grid_result in binned.items():
        nt, npix = grid_result.counts.shape
        pix_all = per_nside_pix[nside]     # (N,)

        # accumulators: sums and counts per (time, pixel, lag)
        msd_sum   = np.zeros((nt, npix, tau_targets.size), dtype=float)
        msd_count = np.zeros((nt, npix, tau_targets.size), dtype=int)
        tau_sum   = np.zeros((nt, npix, tau_targets.size), dtype=float)

        # Loop over time windows
        half_window = win_cfg.win_minutes / 2.0
        for t_index, center_time in enumerate(grid_result.time_centers):
            # Select steps whose midpoint falls in this time window
            tmask = np.abs(times_all - center_time) <= half_window
            if not np.any(tmask):
                continue
            idx_t = np.nonzero(tmask)[0]

            # For binning by start step pixel, we’ll work within this selection
            pix_t = pix_all[idx_t]
            times_t = times_all[idx_t]
            unit_t  = unit_all[idx_t]
            radii_t = radii_all[idx_t]
            ids_t   = ids_all[idx_t]

            # Iterate over pixels that have any steps in this window
            unique_pix = np.unique(pix_t)
            for pix in unique_pix:
                # Optional guard: if the grid's count table is built with a different window,
                # you might skip this check; otherwise keep it for speed alignment
                # if grid_result.counts[t_index, pix] == 0:
                #     continue

                in_pix = (pix_t == pix)
                if np.count_nonzero(in_pix) < 2:
                    continue

                idx_p   = idx_t[in_pix]               # global indices for this (time, pixel)
                times_p = times_all[idx_p]
                unit_p  = unit_all[idx_p]
                radii_p = radii_all[idx_p]
                ids_p   = ids_all[idx_p]

                # For each lag τ, build pairs within (track, τ±tol) and accumulate spherical MSD
                for k, tau in enumerate(tau_targets):
                    start, end = _pair_indices_by_time(times_p, ids_p, tau, rel_tol=lag_rel_tol)
                    if start.size == 0:
                        continue

                    # Map local (pixel-window) indices to global step indices
                    g_start = idx_p[start]
                    g_end   = idx_p[end]

                    # Spherical squared distances
                    msd_sq = _great_circle_sqdist(
                        u1=unit_all[g_start],
                        u2=unit_all[g_end],
                        r1=radii_all[g_start],
                        r2=radii_all[g_end],
                    )
                    # Accumulate
                    msd_sum[t_index, pix, k]   += float(np.sum(msd_sq))
                    msd_count[t_index, pix, k] += int(msd_sq.size)
                    # Use actual Δt for pairs we accepted (could deviate from target by tolerance)
                    tau_sum[t_index, pix, k]   += float(np.sum(times_all[g_end] - times_all[g_start]))

        # Reduce to MSD value (at longest valid lag) and alpha (slope)
        msd_value = np.full((nt, npix), np.nan, dtype=np.float32)
        msd_alpha = np.full((nt, npix), np.nan, dtype=np.float32)

        for ti in range(nt):
            for pix in range(npix):
                counts = msd_count[ti, pix, :]
                valid_lags = counts >= min_pairs_per_lag
                if not np.any(valid_lags):
                    continue

                tau_eff = np.divide(tau_sum[ti, pix, valid_lags], counts[valid_lags])
                msd_eff = np.divide(msd_sum[ti, pix, valid_lags], counts[valid_lags])

                pos = (tau_eff > 0) & (msd_eff > 0)
                if not np.any(pos):
                    continue

                # MSD value = last (longest) available lag
                msd_value[ti, pix] = np.float32(msd_eff[pos][-1])

                # Fit slope on log-log for >=2 valid points
                if np.count_nonzero(pos) >= 2:
                    x = np.log(tau_eff[pos])
                    y = np.log(msd_eff[pos])
                    slope, _ = np.polyfit(x, y, 1)
                    msd_alpha[ti, pix] = np.float32(slope)

        results[nside] = MSDResult(
            nside=nside,
            time_centers=grid_result.time_centers,
            msd_alpha=msd_alpha,
            msd_value=msd_value,
        )

    return results
