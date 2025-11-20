"""Material-response metrics for the cell-dynamics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.cell_field_dynamics.config import MaterialsConfig, WindowConfig
from src.cell_field_dynamics.grids import GridBinResult, healpix_nside2npix, healpix_pix2vec
from src.cell_field_dynamics.vector_field import StepTable, build_step_table


@dataclass(slots=True)
class MaterialMetrics:
    """Container for cage-relative MSD and D2min statistics."""

    nside: int
    time_centers: np.ndarray
    cmsd_alpha: np.ndarray       # (nt, npix)
    d2min_short: np.ndarray      # (nt, npix)
    d2min_long: np.ndarray       # (nt, npix)


# -----------------------------------------------------------
# Helper functions reused from the old module (unchanged core)
# -----------------------------------------------------------

def _relative_displacements(
    displacements: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """
    Cage-relative squared displacement per step:

        rel[i] = |Δr_i - mean_j Δr_j|^2

    where j runs over spatial neighbors of i.
    """
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
    """
    Local non-affine deformation D2min per step.

    For each step i, fit an affine map X that takes neighbor
    relative start positions r0 to relative end positions r1:

        r1 ≈ r0 @ X

    D2min is the mean squared residual norm.
    """
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
        except np.linalg.LinAlgError:  # numerical guard
            d2min[i] = np.nan

    return d2min


def _query_neighbors(points: np.ndarray, k: int) -> np.ndarray:
    """
    k-NN neighbor indices for each point (excluding self).

    Returns an array of shape (N, k) with -1 for missing neighbors.
    """
    if points.shape[0] <= 1:
        return np.full((points.shape[0], k), -1, dtype=int)

    tree = cKDTree(points)
    k_query = min(points.shape[0], max(k + 1, 2))  # include self then drop
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


# -----------------------------------------------------------
# Per-time-bin worker with HEALPix neighbors + spatial weights
# -----------------------------------------------------------

def _compute_material_single_t(
    t_index: int,
    time_centers: np.ndarray,
    step_table: StepTable,
    pixel_vectors: np.ndarray,
    pix_indices: np.ndarray,               # (N_steps,) step→pixel index at this nside
    neighbors_pix: list[np.ndarray],       # neighbors_pix[pix] = array of pixel IDs
    half_window: float,
    space_sigma: float,                    # angular sigma in radians
    mat_cfg: MaterialsConfig,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cmsd_alpha, d2min_short, d2min_long for a single time bin t_index
    at a given HEALPix resolution.
    """

    npix = pixel_vectors.shape[0]
    center_time = time_centers[t_index]

    cmsd_row = np.full(npix, np.nan, dtype=np.float32)
    d2min_short_row = np.full(npix, np.nan, dtype=np.float32)
    d2min_long_row = np.full(npix, np.nan, dtype=np.float32)

    # ---- Temporal window selection ----
    times = step_table.mid_times
    mask_time = np.abs(times - center_time) <= half_window
    if not np.any(mask_time):
        return t_index, cmsd_row, d2min_short_row, d2min_long_row

    idx_time = np.nonzero(mask_time)[0]      # global indices of steps in window
    unit_t   = step_table.unit_mid[idx_time] # (Nt,3)
    pix_t    = pix_indices[idx_time]         # (Nt,)

    # Shorthands into big arrays
    starts_all = step_table.start_positions
    ends_all   = step_table.end_positions
    disp_all   = step_table.displacements
    dt_all     = step_table.dt

    # time-scale thresholds in *minutes*
    median_dt = float(np.median(step_table.dt)) if step_table.dt.size else 1.0
    short_tau = mat_cfg.d2min_deltas_frames[0] * median_dt
    long_tau  = mat_cfg.d2min_deltas_frames[1] * median_dt

    # -------------------------------------------------------
    # Loop over all pixels; spatial smoothing uses neighbors_pix
    # -------------------------------------------------------
    for pix in range(npix):
        center_vec = pixel_vectors[pix]          # unit vector for this pixel
        neigh_pix_ids = neighbors_pix[pix]       # HEALPix neighbors (including itself)

        # restrict time-window steps to these pixels
        mask_local = np.isin(pix_t, neigh_pix_ids)
        if not np.any(mask_local):
            continue

        local_idx = np.nonzero(mask_local)[0]    # indices into idx_time
        step_idx  = idx_time[local_idx]          # global indices

        unit_loc  = unit_t[local_idx]            # (N_loc,3)
        # angular distance to pixel center
        cosang = np.clip(unit_loc @ center_vec, -1.0, 1.0)
        ang = np.arccos(cosang)                  # radians

        # Gaussian spatial weights
        w_space = np.exp(-0.5 * (ang / space_sigma) ** 2)
        valid = w_space > 1e-2
        if np.count_nonzero(valid) < 3:
            continue

        step_idx = step_idx[valid]
        w_steps  = w_space[valid]

        starts = starts_all[step_idx]      # (N,3)
        ends   = ends_all[step_idx]        # (N,3)
        disp   = disp_all[step_idx]        # (N,3)
        dt_sel = dt_all[step_idx]          # (N,)

        # Per-step spatial neighbors (kNN in 3D space at the start positions)
        neigh_idx_local = _query_neighbors(starts, mat_cfg.knn_neighbors)
        rel_sq = _relative_displacements(disp, neigh_idx_local)    # cage-relative MSD
        d2min_vals = _d2min_for_step(starts, ends, neigh_idx_local)

        # Separate short vs long dt steps
        short_mask = dt_sel <= short_tau
        long_mask  = dt_sel >= long_tau

        # ----- cmsd_alpha: log-slope of cage-relative MSD vs tau -----
        # NOTE: This only makes sense if there are both short and long dt values.
        if np.any(short_mask) and np.any(long_mask) and (long_tau > short_tau):
            rel_short = np.nanmean(rel_sq[short_mask])
            rel_long  = np.nanmean(rel_sq[long_mask])

            if rel_short > 0 and rel_long > 0:
                alpha = np.log(rel_long / rel_short) / np.log(long_tau / short_tau)
                cmsd_row[pix] = np.float32(alpha)

        # ----- D2min on short and long steps (use weighted means if possible) -----
        if np.any(short_mask):
            w_short = w_steps[short_mask]
            d2_short = d2min_vals[short_mask]
            # guard against all-NaN or zero weights
            good_s = np.isfinite(d2_short) & (w_short > 0)
            if np.any(good_s):
                d2min_short_row[pix] = np.float32(
                    np.average(d2_short[good_s], weights=w_short[good_s])
                )

        if np.any(long_mask):
            w_long = w_steps[long_mask]
            d2_long = d2min_vals[long_mask]
            good_l = np.isfinite(d2_long) & (w_long > 0)
            if np.any(good_l):
                d2min_long_row[pix] = np.float32(
                    np.average(d2_long[good_l], weights=w_long[good_l])
                )

    return t_index, cmsd_row, d2min_short_row, d2min_long_row


# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------

def compute_material_metrics(
    tracks: pd.DataFrame,
    binned: Dict[int, GridBinResult],
    mat_cfg: MaterialsConfig,
    win_cfg: WindowConfig,
    step_table: StepTable | None = None,
    *,
    neighbors: Optional[Dict[int, list[np.ndarray]]] = None,
    sigma_space_um: Optional[float] = None,
    sphere_radius: Optional[float] = None,
    n_workers: int = 1,
) -> Dict[int, MaterialMetrics]:
    """
    Compute cage-relative MSD exponent and D2min statistics on HEALPix grids.

    For each nside and time window:
      - select steps within |t - t_center| <= win_cfg.win_minutes / 2
      - for each pixel:
          * gather steps in HEALPix neighbors[pix]
          * apply angular Gaussian weights with sigma ~ sigma_space_um / sphere_radius
          * compute per-step cage-relative MSD and D2min using kNN neighbors
          * summarize into:
              cmsd_alpha   : cage-relative MSD scaling exponent (short vs long dt)
              d2min_short  : weighted average D2min over short dt steps
              d2min_long   : weighted average D2min over long dt steps
    """

    if step_table is None:
        step_table = build_step_table(tracks)

    results: Dict[int, MaterialMetrics] = {}

    # empty case
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

    if sphere_radius is None:
        sphere_radius = step_table.mean_radius

    if neighbors is None:
        raise ValueError("compute_material_metrics requires a 'neighbors' dict keyed by nside.")

    half_window = win_cfg.win_minutes / 2.0

    for nside, grid_result in tqdm(binned.items(), desc="Material metrics HEALPix levels"):
        nt, npix = grid_result.counts.shape

        cmsd_alpha = np.full((nt, npix), np.nan, dtype=np.float32)
        d2min_short = np.full((nt, npix), np.nan, dtype=np.float32)
        d2min_long = np.full((nt, npix), np.nan, dtype=np.float32)

        time_centers = grid_result.time_centers
        pix_indices = step_table.pixel_indices(nside)
        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))

        npix_total = healpix_nside2npix(nside)
        if sigma_space_um is not None:
            # user-specified physical smoothing scale
            space_sigma = max(sigma_space_um / sphere_radius, 1e-6)
        else:
            # fallback: ~1 pixel radius on the unit sphere
            pixel_area_unit = 4.0 * np.pi / npix_total
            space_sigma = max(np.sqrt(pixel_area_unit), 1e-6)

        if nside not in neighbors:
            raise ValueError(f"neighbors dict missing entry for nside={nside}")
        neighbors_pix = neighbors[nside]

        # prepare partial for this nside
        from functools import partial as _partial

        worker = _partial(
            _compute_material_single_t,
            time_centers=time_centers,
            step_table=step_table,
            pixel_vectors=pixel_vectors,
            pix_indices=pix_indices,
            neighbors_pix=neighbors_pix,
            half_window=half_window,
            space_sigma=space_sigma,
            mat_cfg=mat_cfg,
        )

        tasks = list(range(nt))

        if n_workers > 1:
            outputs = process_map(
                worker,
                tasks,
                max_workers=n_workers,
                chunksize=1,
                desc=f"material nside={nside}",
            )
        else:
            outputs = []
            for t_idx in tqdm(tasks, desc=f"material nside={nside}"):
                outputs.append(worker(t_idx))

        # reassemble
        for t_idx, cmsd_row, d2s_row, d2l_row in outputs:
            cmsd_alpha[t_idx] = cmsd_row
            d2min_short[t_idx] = d2s_row
            d2min_long[t_idx] = d2l_row

        results[nside] = MaterialMetrics(
            nside=nside,
            time_centers=time_centers,
            cmsd_alpha=cmsd_alpha,
            d2min_short=d2min_short,
            d2min_long=d2min_long,
        )

    return results


__all__ = ["MaterialMetrics", "compute_material_metrics"]
