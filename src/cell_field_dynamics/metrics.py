"""Scalar metric calculations for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.cell_field_dynamics.config import WindowConfig, SmoothingConfig
from src.cell_field_dynamics.vector_field import (
    StepTable,
    VectorFieldResult,
    build_step_table,
    tangent_basis,
)
from src.cell_field_dynamics.grids import (
    GridBinResult,
    healpix_nside2npix,
    healpix_pix2vec,
)


@dataclass(slots=True)
class MetricCollection:
    """Bundle of scalar per-pixel metrics for a given HEALPix grid."""

    nside: int
    time_centers: np.ndarray
    data: Dict[str, np.ndarray]


DEFAULT_METRIC_NAMES = [
    "path_speed",
    "drift_speed",
    "theta_entropy",
    "alignment",
    "diffusivity_total",
    "diffusivity_idio",
]


def _initialise_arrays(result: GridBinResult) -> dict[str, np.ndarray]:
    nt, npix = result.counts.shape
    return {
        name: np.full((nt, npix), np.nan, dtype=np.float32)
        for name in DEFAULT_METRIC_NAMES
    }


def weighted_theta_entropy(theta: np.ndarray, weights: np.ndarray, n_bins: int = 36) -> float:
    """
    Weighted entropy of misalignment angles θ ∈ [0, π].
    Returns value normalized to [0, 1].
    """
    theta = np.asarray(theta, float)
    weights = np.asarray(weights, float)

    if theta.size == 0 or weights.size == 0:
        return np.nan

    hist, _ = np.histogram(theta, bins=n_bins, range=(0.0, np.pi), weights=weights)
    total = np.sum(hist)
    if total <= 0:
        return np.nan

    p = hist / total
    mask = p > 0
    if not np.any(mask):
        return np.nan

    ent = -np.sum(p[mask] * np.log(p[mask]))
    return float(ent / np.log(n_bins))


# ---------------- helpers ----------------
def _residual_displacements_tangent(
    coords_centered_3d: np.ndarray,   # (N,3) mid-positions centered on pixel center vec
    disp_3d: np.ndarray,              # (N,3) step displacements (already embryo-centered)
    dt: np.ndarray,                   # (N,)
    e_theta: np.ndarray,              # (3,)
    e_phi: np.ndarray,                # (3,)
    drift_vec_3d: np.ndarray | None = None,  # (3,), μm/min
    jac_tan: np.ndarray | None = None        # (2,2), ∂v/∂x in tangent coords
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project to tangent plane and subtract modeled flow.
    Returns: disp_tan_res (N,2), coords_tan (N,2), disp_tan (N,2 original)
    """
    # project 3D to 2D tangent plane
    coords_tan = np.column_stack((coords_centered_3d @ e_theta,
                                  coords_centered_3d @ e_phi))  # (N,2)
    disp_tan   = np.column_stack((disp_3d @ e_theta,
                                  disp_3d @ e_phi))             # (N,2)

    if drift_vec_3d is None and jac_tan is None:
        return disp_tan, coords_tan, disp_tan

    v_drift_tan = np.zeros(2, dtype=float)
    if drift_vec_3d is not None:
        v_drift_tan = np.array(
            [drift_vec_3d @ e_theta, drift_vec_3d @ e_phi], dtype=float
        )

    if jac_tan is None:
        pred_disp_tan = v_drift_tan[None, :] * dt[:, None]
    else:
        # affine velocity: v = v0 + J * x_tan
        pred_vel_tan  = v_drift_tan[None, :] + coords_tan @ jac_tan.T
        pred_disp_tan = pred_vel_tan * dt[:, None]

    return (disp_tan - pred_disp_tan), coords_tan, disp_tan


def _cve_diffusivity_with_times(
    disp_tan: np.ndarray,       # (N,2) tangent displacements (residuals or raw)
    dt: np.ndarray,             # (N,)
    track_ids: np.ndarray,      # (N,)
    times: np.ndarray | None,   # (N,) window-local mid-times (minutes); used for sorting
    min_pairs: int = 1,
) -> tuple[float, float]:
    """
    CVE diffusivity and localization variance (σ_loc^2) from 2D displacements.
    Sorts by (track_id, time) so lag-1 covariance is meaningful.
    Returns (D, sigma2). Units: D in μm^2/min, sigma2 in μm^2.
    """
    n, d = disp_tan.shape if disp_tan.ndim == 2 else (0, 0)
    if n < 2:
        return np.nan, np.nan

    if times is None:
        order = np.lexsort((np.arange(n), track_ids))
    else:
        order = np.lexsort((times, track_ids))

    x = disp_tan[order]
    ids = track_ids[order]
    dt_o = dt[order]

    # mean-free
    mu = x.mean(axis=0)
    c = x - mu  # (N,2)

    # per-step per-axis variance averaged
    var = float(np.mean(np.sum(c * c, axis=1) / d))

    # lag-1 cov within tracks
    cov_terms = []
    for i in range(n - 1):
        if ids[i] == ids[i + 1]:
            cov_terms.append(np.dot(c[i], c[i + 1]) / d)
    if len(cov_terms) < min_pairs:
        return np.nan, np.nan
    cov = float(np.mean(cov_terms))

    dt_mean = float(np.mean(dt_o))
    sigma2 = max(-cov, 0.0)
    D = (var + 2.0 * cov) / (2.0 * max(dt_mean, 1e-6))
    return D, sigma2


def local_alignment_C(vel3d: np.ndarray, speed_tol: float = 1e-6) -> float:
    """
    Local directional alignment:
        C_local = mean_{i<j} ( v̂_i ⋅ v̂_j ).

    vel3d : (N,3) array of 3D velocities.
    speed_tol : filter tiny speeds.

    Returns scalar alignment measure in [-1,1].
    """
    if vel3d.shape[0] < 2:
        return np.nan

    spd = np.linalg.norm(vel3d, axis=1)
    good = spd > speed_tol
    if np.count_nonzero(good) < 2:
        return np.nan

    v = vel3d[good]
    vhat = v / np.linalg.norm(v, axis=1, keepdims=True)

    # pairwise dot products (upper triangle)
    dots = vhat @ vhat.T
    np.clip(dots, -1.0, 1.0, out=dots)

    i, j = np.triu_indices(vhat.shape[0], k=1)
    return float(np.mean(dots[i, j]))


# ---------------- core worker (neighbor-based) ----------------
def _compute_scalar_metrics_single_t(
    t_index: int,
    time_centers: np.ndarray,
    pixel_vectors: np.ndarray,
    pix_indices: np.ndarray,           # (N,) step → pixel index at this nside
    neighbors: list[np.ndarray],       # neighbors[pix] = array of pixel IDs
    step_table: StepTable,
    vf: Optional[VectorFieldResult],
    half_window: float,
    space_sigma: float,                # radians
    min_steps_per_bin: int,
) -> tuple[int, dict[str, np.ndarray]]:

    npix = pixel_vectors.shape[0]
    center_time = time_centers[t_index]

    # Allocate one row per metric
    row = {
        name: np.full(npix, np.nan, dtype=np.float32)
        for name in DEFAULT_METRIC_NAMES
    }

    # ---- Select steps in temporal window ----
    times = step_table.mid_times
    mask_time = np.abs(times - center_time) <= half_window
    if not np.any(mask_time):
        return t_index, row

    idx_time = np.nonzero(mask_time)[0]
    unit_t   = step_table.unit_mid[idx_time]      # (Nt, 3)
    pix_t    = pix_indices[idx_time]              # (Nt,)

    # Pre-extract arrays we’ll need
    disp_all  = step_table.displacements
    dt_all    = step_table.dt
    vel_all   = step_table.velocities
    coords_all = step_table.mid_positions
    track_ids_all = step_table.track_ids
    times_all     = step_table.mid_times

    mean_R = step_table.mean_radius

    # ---- Loop over all pixels (neighbor logic will keep things sparse) ----
    for pix in range(npix):
        center_vec = pixel_vectors[pix]  # unit 3D direction of pixel center

        # restrict steps to pixels in neighbor list
        neigh = neighbors[pix]           # array of neighbor pixel IDs
        mask_local = np.isin(pix_t, neigh)
        if not np.any(mask_local):
            continue

        local_idx = np.nonzero(mask_local)[0]   # indices into idx_time
        step_idx  = idx_time[local_idx]         # global step indices

        # spatial weights: angular distance
        unit_loc = unit_t[local_idx]           # (N_loc, 3)
        cosang = np.clip(unit_loc @ center_vec, -1.0, 1.0)
        ang = np.arccos(cosang)                # radians
        w_space = np.exp(-0.5 * (ang / space_sigma) ** 2)

        valid_local = w_space > 1e-2
        if np.count_nonzero(valid_local) < min_steps_per_bin:
            continue

        step_idx = step_idx[valid_local]
        w = w_space[valid_local]

        # Extract step-wise arrays in this neighborhood
        disp3d   = disp_all[step_idx]          # (N,3)
        dt       = dt_all[step_idx]            # (N,)
        vel3d_pix = vel_all[step_idx]          # (N,3)
        coords3d = coords_all[step_idx]        # (N,3)
        track_ids = track_ids_all[step_idx]
        times_sel = times_all[step_idx]

        # Tangent basis & centered coords
        e_theta, e_phi = tangent_basis(center_vec)
        patch_center_xyz = center_vec * mean_R
        coords_centered = coords3d - patch_center_xyz[None, :]

        # ---- PATH SPEED (weighted) ----
        disp_tan = np.column_stack((disp3d @ e_theta, disp3d @ e_phi))
        speed_tan = np.linalg.norm(disp_tan, axis=1) / np.maximum(dt, 1e-6)
        try:
            row["path_speed"][pix] = np.float32(np.average(speed_tan, weights=w))
        except ZeroDivisionError:
            # extremely unlikely given valid_local check, but guard anyway
            pass

        # ---- DRIFT SPEED (grid-level) ----
        if vf is not None:
            row["drift_speed"][pix] = np.float32(
                np.linalg.norm(vf.drift[t_index, pix])
            )

        # ---- THETA ENTROPY + ALIGNMENT (weighted) ----
        spd3d = np.linalg.norm(vel3d_pix, axis=1)
        good = spd3d > 1e-6

        if np.count_nonzero(good) >= 2:
            v = vel3d_pix[good]               # (M,3)
            w_good = w[good]                  # (M,)

            vhat = v / np.linalg.norm(v, axis=1, keepdims=True)
            dots = vhat @ vhat.T
            np.clip(dots, -1.0, 1.0, out=dots)

            i, j = np.triu_indices(vhat.shape[0], k=1)
            theta = np.arccos(dots[i, j])     # pairwise misalignment angles

            # pairwise weights ~ w_i * w_j
            pair_w = (w_good[:, None] * w_good[None, :])[i, j]

            # weighted theta entropy
            row["theta_entropy"][pix] = np.float32(
                weighted_theta_entropy(theta, pair_w)
            )

            # weighted alignment C_local
            C_num = np.sum(pair_w * dots[i, j])
            C_den = np.sum(pair_w)
            if C_den > 0:
                row["alignment"][pix] = np.float32(C_num / C_den)

        # ---- DIFFUSIVITY TOTAL (unweighted, but same subset) ----
        D_total, _ = _cve_diffusivity_with_times(
            disp_tan=disp_tan,
            dt=dt,
            track_ids=track_ids,
            times=times_sel,
        )
        row["diffusivity_total"][pix] = np.float32(D_total)

        # ---- DIFFUSIVITY IDIO (unweighted, but same subset) ----
        if vf is not None:
            drift_vec = vf.drift[t_index, pix]
            J_tan = vf.jacobian[t_index, pix] if vf.jacobian is not None else None

            disp_res_aff, coords_tan, _ = _residual_displacements_tangent(
                coords_centered_3d=coords_centered,
                disp_3d=disp3d,
                dt=dt,
                e_theta=e_theta,
                e_phi=e_phi,
                drift_vec_3d=drift_vec,
                jac_tan=J_tan,
            )

            D_idio, _ = _cve_diffusivity_with_times(
                disp_tan=disp_res_aff,
                dt=dt,
                track_ids=track_ids,
                times=times_sel,
            )
            row["diffusivity_idio"][pix] = np.float32(D_idio)

    return t_index, row


# ---------------- public API ----------------
def compute_scalar_metrics(
    tracks: pd.DataFrame,
    vector_results: dict[int, VectorFieldResult],
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    smooth_cfg: SmoothingConfig,
    step_table: StepTable | None = None,
    neighbors: Optional[Dict[int, list[np.ndarray]]] = None,
    min_steps_per_bin: int = 6,
    n_workers: int = 1,
) -> dict[int, MetricCollection]:

    if step_table is None:
        step_table = build_step_table(tracks)

    metrics: dict[int, MetricCollection] = {}

    if step_table.mid_times.size == 0:
        for nside, grid_result in binned.items():
            arrays = _initialise_arrays(grid_result)
            metrics[nside] = MetricCollection(
                nside=nside,
                time_centers=grid_result.time_centers,
                data=arrays,
            )
        return metrics

    # if sphere_radius is None:
    #     sphere_radius = step_table.mean_radius
    space_sigma = smooth_cfg.sigma_radians

    half_window = win_cfg.win_minutes / 2.0

    for nside, grid_result in tqdm(binned.items(), desc="Scalar metric levels"):
        nt, npix = grid_result.counts.shape
        time_centers = grid_result.time_centers

        # geometry
        npix_total = healpix_nside2npix(nside)
        # if sigma_space_um is not None:
        #     space_sigma = max(sigma_space_um / sphere_radius, 1e-6)
        # else:
        #     pixel_area_unit = 4.0 * np.pi / npix_total
        #     space_sigma = max(np.sqrt(pixel_area_unit), 1e-6)

        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))
        pix_indices = step_table.pixel_indices(nside)
        vf = vector_results.get(nside, None)

        if neighbors is None or nside not in neighbors:
            raise ValueError("neighbors[nside] must be provided for scalar metrics")

        neighbors_nside = neighbors[nside]

        arrays = _initialise_arrays(grid_result)

        # build partial for this nside
        calculate_metrics = partial(
            _compute_scalar_metrics_single_t,
            time_centers=time_centers,
            pixel_vectors=pixel_vectors,
            pix_indices=pix_indices,
            neighbors=neighbors_nside,
            step_table=step_table,
            vf=vf,
            half_window=half_window,
            space_sigma=space_sigma,
            min_steps_per_bin=min_steps_per_bin,
        )

        tasks = list(range(nt))

        # Parallel execution
        if n_workers > 1:
            outputs = process_map(
                calculate_metrics,
                tasks,
                max_workers=n_workers,
                chunksize=1,
                desc=f"scalar nside={nside}",
            )
        else:
            outputs = []
            for t in tqdm(tasks, desc=f"scalar nside={nside}"):
                outputs.append(calculate_metrics(t))

        # Reassemble rows into arrays
        for t_index, row in outputs:
            for name, vec in row.items():
                arrays[name][t_index] = vec

        metrics[nside] = MetricCollection(
            nside=nside,
            time_centers=grid_result.time_centers,
            data=arrays,
        )

    return metrics


__all__ = ["MetricCollection", "compute_scalar_metrics"]
