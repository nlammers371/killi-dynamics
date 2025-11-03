"""Scalar metric calculations for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import WindowConfig
from .grids import GridBinResult, healpix_pix2vec
from .vector_field import StepTable, VectorFieldResult, build_step_table, tangent_basis


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
    "diffusivity_total",
    "diffusivity_idio",
]


def _initialise_arrays(result: GridBinResult) -> dict[str, np.ndarray]:
    nt, npix = result.counts.shape
    return {name: np.full((nt, npix), np.nan, dtype=np.float32) for name in DEFAULT_METRIC_NAMES}


def _theta_entropy(angles: np.ndarray, n_bins: int = 36) -> float:
    if angles.size == 0:
        return np.nan
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    total = hist.sum()
    if total == 0:
        return np.nan
    p = hist.astype(float) / total
    mask = p > 0
    if np.count_nonzero(mask) == 0:
        return np.nan
    entropy = -np.sum(p[mask] * np.log(p[mask]))
    return float(entropy / np.log(n_bins))


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
    coords_tan = np.column_stack((coords_centered_3d @ e_theta, coords_centered_3d @ e_phi))  # (N,2)
    disp_tan   = np.column_stack((disp_3d            @ e_theta, disp_3d            @ e_phi))  # (N,2)

    if drift_vec_3d is None and jac_tan is None:
        return disp_tan, coords_tan, disp_tan

    v_drift_tan = np.zeros(2, dtype=float)
    if drift_vec_3d is not None:
        v_drift_tan = np.array([drift_vec_3d @ e_theta, drift_vec_3d @ e_phi], dtype=float)

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



# ---------------- replacement for compute_scalar_metrics ----------------

def compute_scalar_metrics(
    tracks: pd.DataFrame,
    vector_results: dict[int, VectorFieldResult],
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    step_table: StepTable | None = None,
    *,
    min_steps_per_bin: int = 6
) -> dict[int, MetricCollection]:
    """
    Compute scalar metrics per HEALPix pixel and time window:
      - path_speed (tangent)
      - drift_speed (||drift||)
      - theta_entropy (directional entropy in tangent)
      - diffusivity_total (CVE on tangent displacements, no flow subtraction)
      - diffusivity_idio  (CVE after subtracting local AFFINE flow)
    Assumes step_table coordinates are embryo-centered.
    """
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

    half_window = win_cfg.win_minutes / 2.0

    for nside, grid_result in binned.items():
        arrays = _initialise_arrays(grid_result)
        vf = vector_results.get(nside, None)

        # pixel geometry
        npix = grid_result.counts.shape[1]
        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))  # (npix,3)

        # precomputed per-step pixel assignment for this nside
        pix_idx_all = step_table.pixel_indices(nside)  # (N,)

        for t_index, center_time in enumerate(grid_result.time_centers):
            # time window selection
            time_mask = np.abs(step_table.mid_times - center_time) <= half_window
            if not np.any(time_mask):
                continue
            idx_time = np.nonzero(time_mask)[0]
            pix_time = pix_idx_all[idx_time]

            # iterate only over pixels that actually have steps in this window
            for pix in np.unique(pix_time):
                # optional: drop the discrete counts gate if it was built with a different window
                # if grid_result.counts[t_index, pix] == 0:
                #     continue

                indices = idx_time[pix_time == pix]
                if indices.size < min_steps_per_bin:
                    continue

                center_vec = pixel_vectors[pix]
                e_theta, e_phi = tangent_basis(center_vec)

                # gather per-step arrays
                disp3d  = step_table.displacements[indices]   # (N,3)
                dt      = step_table.dt[indices]              # (N,)
                vel3d   = step_table.velocities[indices]      # (N,3)
                coords3d= step_table.mid_positions[indices]   # (N,3) embryo-centered mids
                # center coords on the pixel center vector used for the tangent basis
                coords_centered = coords3d - center_vec[None, :]

                # ---- path_speed (tangent) ----
                # use tangent projection of displacements
                disp_tan = np.column_stack((disp3d @ e_theta, disp3d @ e_phi))  # (N,2)
                speed_tan = np.linalg.norm(disp_tan, axis=1) / np.maximum(dt, 1e-6)
                arrays["path_speed"][t_index, pix] = np.float32(np.nanmean(speed_tan))

                # ---- drift_speed ----
                if vf is not None:
                    arrays["drift_speed"][t_index, pix] = np.float32(
                        np.linalg.norm(vf.drift[t_index, pix])
                    )

                # ---- theta_entropy (mask nearly-zero speeds) ----
                vel_tan = np.column_stack((vel3d @ e_theta, vel3d @ e_phi))
                spd = np.linalg.norm(vel_tan, axis=1)
                good = spd > 1e-6
                if np.any(good):
                    angles = np.arctan2(vel_tan[good, 1], vel_tan[good, 0])
                    arrays["theta_entropy"][t_index, pix] = np.float32(_theta_entropy(angles))

                # ---- D_total (no flow subtraction; tangent) ----
                D_total, sigma2 = _cve_diffusivity_with_times(
                    disp_tan=disp_tan,
                    dt=dt,
                    track_ids=step_table.track_ids[indices],
                    times=step_table.mid_times[indices],
                )
                arrays["diffusivity_total"][t_index, pix] = np.float32(D_total)

                # ---- D_idio (AFFINE flow subtraction; tangent) ----
                if vf is not None:
                    drift_vec3d = vf.drift[t_index, pix]                     # (3,)
                    J_tan = vf.jacobian[t_index, pix] if vf.jacobian is not None else None  # (2,2) or None

                    disp_res_aff, coords_tan, _ = _residual_displacements_tangent(
                        coords_centered_3d=coords_centered,
                        disp_3d=disp3d,
                        dt=dt,
                        e_theta=e_theta,
                        e_phi=e_phi,
                        drift_vec_3d=drift_vec3d,
                        jac_tan=J_tan,
                    )
                    D_idio, _ = _cve_diffusivity_with_times(
                        disp_tan=disp_res_aff,
                        dt=dt,
                        track_ids=step_table.track_ids[indices],
                        times=step_table.mid_times[indices],
                    )
                    arrays["diffusivity_idio"][t_index, pix] = np.float32(D_idio)

        metrics[nside] = MetricCollection(
            nside=nside,
            time_centers=grid_result.time_centers,
            data=arrays,
        )

    return metrics



__all__ = ["MetricCollection", "compute_scalar_metrics"]
