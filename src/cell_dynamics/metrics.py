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


def _cve_diffusivity(
    displacements: np.ndarray,
    dt: np.ndarray,
    track_ids: np.ndarray,
) -> tuple[float, float]:
    """Compute CVE diffusivity (total) and localisation variance."""

    n = displacements.shape[0]
    if n < 2:
        return np.nan, np.nan

    order = np.lexsort((track_ids,))
    disp = displacements[order]
    ids = track_ids[order]
    dt_sel = dt[order]

    mean_disp = disp.mean(axis=0)
    centered = disp - mean_disp
    dims = disp.shape[1]
    var = float(np.mean(np.sum(centered**2, axis=1))) / dims

    cov_terms = []
    for i in range(n - 1):
        if ids[i] == ids[i + 1]:
            cov_terms.append(float(np.dot(centered[i], centered[i + 1])))
    cov = 0.0 if not cov_terms else float(np.mean(cov_terms)) / dims

    dt_mean = float(np.mean(dt_sel)) if dt_sel.size else 1.0
    sigma2 = max(-cov, 0.0)
    D_total = (var + 2.0 * cov) / (2.0 * max(dt_mean, 1e-6))
    return D_total, sigma2


def compute_scalar_metrics(
    tracks: pd.DataFrame,
    vector_results: dict[int, VectorFieldResult],
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    step_table: StepTable | None = None,
) -> dict[int, MetricCollection]:
    """Compute scalar kinematic metrics for each grid cell and time window."""

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
        vf = vector_results.get(nside)
        arrays = _initialise_arrays(grid_result)
        pixel_vectors = healpix_pix2vec(nside, np.arange(grid_result.counts.shape[1], dtype=int))
        pixel_idx = step_table.pixel_indices(nside)

        for t_index, center_time in enumerate(grid_result.time_centers):
            time_mask = np.abs(step_table.mid_times - center_time) <= half_window
            if not np.any(time_mask):
                continue
            idx_time = np.nonzero(time_mask)[0]
            pix_time = pixel_idx[idx_time]

            for pix in np.unique(pix_time):
                counts = grid_result.counts[t_index, pix]
                if counts == 0:
                    continue
                indices = idx_time[pix_time == pix]
                if indices.size == 0:
                    continue
                speeds = np.linalg.norm(step_table.displacements[indices], axis=1) / np.maximum(step_table.dt[indices], 1e-6)
                arrays["path_speed"][t_index, pix] = np.float32(np.nanmean(speeds))

                if vf is not None and vf.drift.shape[2] >= 3:
                    arrays["drift_speed"][t_index, pix] = np.float32(np.linalg.norm(vf.drift[t_index, pix]))
                elif vf is not None:
                    arrays["drift_speed"][t_index, pix] = np.float32(np.linalg.norm(vf.drift[t_index, pix]))

                center_vec = pixel_vectors[pix]
                e_theta, e_phi = tangent_basis(center_vec)
                vel_proj = np.column_stack(
                    (
                        step_table.velocities[indices] @ e_theta,
                        step_table.velocities[indices] @ e_phi,
                    )
                )
                angles = np.arctan2(vel_proj[:, 1], vel_proj[:, 0])
                arrays["theta_entropy"][t_index, pix] = np.float32(_theta_entropy(angles))

                D_total, sigma2 = _cve_diffusivity(
                    step_table.displacements[indices],
                    step_table.dt[indices],
                    step_table.track_ids[indices],
                )
                arrays["diffusivity_total"][t_index, pix] = np.float32(D_total)
                if np.isnan(D_total) or np.isnan(sigma2):
                    arrays["diffusivity_idio"][t_index, pix] = np.float32(np.nan)
                else:
                    dt_mean = float(np.mean(step_table.dt[indices]))
                    arrays["diffusivity_idio"][t_index, pix] = np.float32(max(D_total - sigma2 / max(dt_mean, 1e-6), 0.0))

        metrics[nside] = MetricCollection(
            nside=nside,
            time_centers=grid_result.time_centers,
            data=arrays,
        )

    return metrics


__all__ = ["MetricCollection", "compute_scalar_metrics"]
