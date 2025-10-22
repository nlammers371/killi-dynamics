"""Mean-squared displacement calculations for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import WindowConfig
from .grids import GridBinResult, healpix_ang2pix
from .vector_field import StepTable


@dataclass(slots=True)
class MSDResult:
    """Aggregate MSD values and scaling exponents for a HEALPix grid."""

    nside: int
    time_centers: np.ndarray
    msd_alpha: np.ndarray
    msd_value: np.ndarray


def _prepare_tracks(tracks: pd.DataFrame) -> Dict[int, dict[str, np.ndarray]]:
    if tracks.empty:
        return {}

    if "particle" not in tracks.columns:
        raise ValueError("Tracks dataframe must contain a 'particle' column for MSD calculations.")

    if "time_min" in tracks.columns:
        time_col = "time_min"
    elif "time" in tracks.columns:
        time_col = "time"
    elif "frame" in tracks.columns:
        time_col = "frame"
    else:
        raise ValueError("Tracks dataframe must include a temporal column (time_min/time/frame).")

    required = {"x", "y", "z"}
    if not required.issubset(tracks.columns):
        raise ValueError("Tracks dataframe must contain Cartesian coordinates 'x', 'y', 'z'.")

    prepared: Dict[int, dict[str, np.ndarray]] = {}
    for tid, group in tracks.sort_values(["particle", time_col]).groupby("particle"):
        coords = group[["x", "y", "z"]].to_numpy(dtype=float)
        times = group[time_col].to_numpy(dtype=float)
        if coords.shape[0] < 2:
            continue
        radii = np.linalg.norm(coords, axis=1)
        radii = np.where(radii == 0.0, 1.0, radii)
        theta = np.arccos(np.clip(coords[:, 2] / radii, -1.0, 1.0))
        phi = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
        prepared[int(tid)] = {
            "positions": coords,
            "times": times,
            "theta": theta,
            "phi": phi,
        }
    return prepared


def _lagged_displacements(indices: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    start = indices[:-lag]
    end = indices[lag:]
    valid = (end - start) == lag
    return start[valid], end[valid]


def compute_msd_metrics(
    tracks: pd.DataFrame,
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    step_table: StepTable | None = None,
) -> dict[int, MSDResult]:
    """Compute ensemble MSD curves and local scaling exponents."""

    prepared_tracks = _prepare_tracks(tracks)
    results: dict[int, MSDResult] = {}

    if not prepared_tracks:
        for nside, grid_result in binned.items():
            nt, npix = grid_result.counts.shape
            nan_array = np.full((nt, npix), np.nan, dtype=np.float32)
            results[nside] = MSDResult(
                nside=nside,
                time_centers=grid_result.time_centers,
                msd_alpha=nan_array.copy(),
                msd_value=nan_array.copy(),
            )
        return results

    lag_steps = np.array([1, 2, 4])

    for nside, grid_result in binned.items():
        nt, npix = grid_result.counts.shape
        msd_sum = np.zeros((nt, npix, lag_steps.size), dtype=float)
        msd_counts = np.zeros((nt, npix, lag_steps.size), dtype=int)
        tau_sum = np.zeros((nt, npix, lag_steps.size), dtype=float)

        for tid, data in prepared_tracks.items():
            times = data["times"]
            positions = data["positions"]
            theta = data["theta"]
            phi = data["phi"]
            pixels = healpix_ang2pix(nside, theta, phi)
            n_points = times.size

            for t_index, center_time in enumerate(grid_result.time_centers):
                half_window = win_cfg.win_minutes / 2.0
                mask = np.abs(times - center_time) <= half_window
                if not np.any(mask):
                    continue
                idx = np.nonzero(mask)[0]
                pix_subset = pixels[idx]

                for pix in np.unique(pix_subset):
                    counts = grid_result.counts[t_index, pix]
                    if counts == 0:
                        continue
                    pix_mask = pix_subset == pix
                    sel = idx[pix_mask]
                    if sel.size < 2:
                        continue
                    for lag_idx, lag in enumerate(lag_steps):
                        if sel.size <= lag:
                            continue
                        start, end = _lagged_displacements(sel, lag)
                        if start.size == 0:
                            continue
                        disp = positions[end] - positions[start]
                        dt = times[end] - times[start]
                        valid_dt = dt > 0
                        if not np.any(valid_dt):
                            continue
                        disp = disp[valid_dt]
                        dt = dt[valid_dt]
                        msd_sum[t_index, pix, lag_idx] += np.sum(np.sum(disp**2, axis=1))
                        tau_sum[t_index, pix, lag_idx] += np.sum(dt)
                        msd_counts[t_index, pix, lag_idx] += disp.shape[0]

        msd_value = np.full((nt, npix), np.nan, dtype=np.float32)
        msd_alpha = np.full((nt, npix), np.nan, dtype=np.float32)

        for t_index in range(nt):
            for pix in range(npix):
                counts = msd_counts[t_index, pix]
                valid = counts > 0
                if not np.any(valid):
                    continue
                tau = np.divide(tau_sum[t_index, pix, valid], counts[valid], out=np.zeros_like(tau_sum[t_index, pix, valid]), where=counts[valid] > 0)
                msd_vals = np.divide(msd_sum[t_index, pix, valid], counts[valid], out=np.zeros_like(msd_sum[t_index, pix, valid]), where=counts[valid] > 0)
                positive = (tau > 0) & (msd_vals > 0)
                if np.any(positive):
                    msd_value[t_index, pix] = np.float32(msd_vals[positive][-1])
                if np.count_nonzero(positive) >= 2:
                    x = np.log(tau[positive])
                    y = np.log(msd_vals[positive])
                    slope, _ = np.polyfit(x, y, 1)
                    msd_alpha[t_index, pix] = np.float32(slope)

        results[nside] = MSDResult(
            nside=nside,
            time_centers=grid_result.time_centers,
            msd_alpha=msd_alpha,
            msd_value=msd_value,
        )

    return results


__all__ = ["MSDResult", "compute_msd_metrics"]
