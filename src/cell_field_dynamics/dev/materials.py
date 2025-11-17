"""Material-response metrics for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import MaterialsConfig, WindowConfig
from .grids import GridBinResult
from src.cell_field_dynamics.vector_field import StepTable, build_step_table


@dataclass(slots=True)
class MaterialMetrics:
    """Container for cage-relative MSD and D2min statistics."""

    nside: int
    time_centers: np.ndarray
    cmsd_alpha: np.ndarray
    d2min_short: np.ndarray
    d2min_long: np.ndarray


def _relative_displacements(
    displacements: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    rel = np.empty(displacements.shape[0], dtype=float)
    for i, neigh in enumerate(neighbor_indices):
        neigh = neigh[neigh >= 0]
        if neigh.size == 0:
            rel[i] = np.nan
            continue
        neigh_disp = displacements[neigh]
        rel_vec = displacements[i] - np.nanmean(neigh_disp, axis=0)
        rel[i] = float(np.dot(rel_vec, rel_vec))
    return rel


def _d2min_for_step(
    start_positions: np.ndarray,
    end_positions: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    n_steps = start_positions.shape[0]
    d2min = np.empty(n_steps, dtype=float)
    for i, neigh in enumerate(neighbor_indices):
        neigh = neigh[neigh >= 0]
        if neigh.size < 3:
            d2min[i] = np.nan
            continue
        r0 = start_positions[neigh] - start_positions[i]
        r1 = end_positions[neigh] - end_positions[i]
        try:
            X, *_ = np.linalg.lstsq(r0, r1, rcond=None)
            residual = r1 - r0 @ X
            d2min[i] = float(np.mean(np.sum(residual**2, axis=1)))
        except np.linalg.LinAlgError:  # pragma: no cover - numerical guard
            d2min[i] = np.nan
    return d2min


def _query_neighbors(points: np.ndarray, k: int) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.full((points.shape[0], k), -1, dtype=int)
    tree = cKDTree(points)
    k_query = min(points.shape[0], max(k + 1, 2))
    dists, idxs = tree.query(points, k=k_query)
    if k_query == 1:
        idxs = idxs[:, None]
    idxs = np.atleast_2d(idxs)
    # remove self
    mask = idxs != np.arange(points.shape[0])[:, None]
    filtered = np.full((points.shape[0], k), -1, dtype=int)
    for i, row in enumerate(idxs):
        valid = row[mask[i]][:k]
        if valid.size > 0:
            filtered[i, : valid.size] = valid
    return filtered


def compute_material_metrics(
    tracks: pd.DataFrame,
    binned: dict[int, GridBinResult],
    mat_cfg: MaterialsConfig,
    win_cfg: WindowConfig,
    step_table: StepTable | None = None,
) -> dict[int, MaterialMetrics]:
    """Compute cage-relative MSD slopes and D2min rates for each grid."""

    if step_table is None:
        step_table = build_step_table(tracks)

    results: dict[int, MaterialMetrics] = {}

    if step_table.mid_times.size == 0:
        for nside, grid_result in binned.items():
            nt, npix = grid_result.counts.shape
            zeros = np.full((nt, npix), np.nan, dtype=np.float32)
            results[nside] = MaterialMetrics(
                nside=nside,
                time_centers=grid_result.time_centers,
                cmsd_alpha=zeros.copy(),
                d2min_short=zeros.copy(),
                d2min_long=zeros.copy(),
            )
        return results

    median_dt = float(np.median(step_table.dt)) if step_table.dt.size else 1.0
    short_tau = mat_cfg.d2min_deltas_frames[0] * median_dt
    long_tau = mat_cfg.d2min_deltas_frames[1] * median_dt

    for nside, grid_result in binned.items():
        nt, npix = grid_result.counts.shape
        cmsd_alpha = np.full((nt, npix), np.nan, dtype=np.float32)
        d2min_short = np.full((nt, npix), np.nan, dtype=np.float32)
        d2min_long = np.full((nt, npix), np.nan, dtype=np.float32)
        pixel_idx = step_table.pixel_indices(nside)

        for t_index, center_time in enumerate(grid_result.time_centers):
            time_mask = np.abs(step_table.mid_times - center_time) <= win_cfg.win_minutes / 2.0
            if not np.any(time_mask):
                continue
            indices_time = np.nonzero(time_mask)[0]
            pix_time = pixel_idx[indices_time]

            for pix in np.unique(pix_time):
                if grid_result.counts[t_index, pix] == 0:
                    continue
                indices = indices_time[pix_time == pix]
                if indices.size < 3:
                    continue

                starts = step_table.start_positions[indices]
                ends = step_table.end_positions[indices]
                displacements = step_table.displacements[indices]
                dt_sel = step_table.dt[indices]

                neighbor_idx = _query_neighbors(starts, mat_cfg.knn_neighbors)
                rel_sq = _relative_displacements(displacements, neighbor_idx)
                d2min_vals = _d2min_for_step(starts, ends, neighbor_idx)

                short_mask = dt_sel <= short_tau
                long_mask = dt_sel >= long_tau

                if np.any(short_mask) and np.any(long_mask):
                    rel_short = np.nanmean(rel_sq[short_mask])
                    rel_long = np.nanmean(rel_sq[long_mask])
                    if rel_short > 0 and rel_long > 0 and long_tau > short_tau:
                        alpha = np.log(rel_long / rel_short) / np.log(long_tau / short_tau)
                        cmsd_alpha[t_index, pix] = np.float32(alpha)

                if np.any(short_mask):
                    d2min_short[t_index, pix] = np.float32(np.nanmean(d2min_vals[short_mask]))
                if np.any(long_mask):
                    d2min_long[t_index, pix] = np.float32(np.nanmean(d2min_vals[long_mask]))

        results[nside] = MaterialMetrics(
            nside=nside,
            time_centers=grid_result.time_centers,
            cmsd_alpha=cmsd_alpha,
            d2min_short=d2min_short,
            d2min_long=d2min_long,
        )

    return results


__all__ = ["MaterialMetrics", "compute_material_metrics"]
