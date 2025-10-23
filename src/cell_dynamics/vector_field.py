"""Vector-field estimation utilities for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import SmoothingConfig, WindowConfig
from .grids import (
    GridBinResult,
    healpix_ang2pix,
    healpix_nside2npix,
    healpix_pix2vec,
)


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert a full-width at half-maximum value to the corresponding sigma."""

    fwhm = float(fwhm)
    if fwhm <= 0:
        return 1.0
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal tangent basis ``(e_theta, e_phi)`` at ``normal``."""

    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal) if np.linalg.norm(normal) else 1.0
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, normal)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    e_phi = np.cross(ref, normal)
    e_phi_norm = np.linalg.norm(e_phi)
    if e_phi_norm == 0.0:
        e_phi = np.array([1.0, 0.0, 0.0])
    else:
        e_phi /= e_phi_norm
    e_theta = np.cross(normal, e_phi)
    e_theta /= np.linalg.norm(e_theta) if np.linalg.norm(e_theta) else 1.0
    return e_theta, e_phi


@dataclass(slots=True)
class StepTable:
    """Container for per-step kinematic quantities on the sphere."""

    track_ids: np.ndarray
    start_positions: np.ndarray
    end_positions: np.ndarray
    mid_positions: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    dt: np.ndarray
    mid_times: np.ndarray
    unit_mid: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    radii: np.ndarray
    pixel_cache: Dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def pixel_indices(self, nside: int) -> np.ndarray:
        """Return cached HEALPix pixel indices for the supplied ``nside``."""

        nside = int(nside)
        cache = self.pixel_cache.get(nside)
        if cache is None:
            cache = healpix_ang2pix(nside, self.theta, self.phi)
            self.pixel_cache[nside] = cache
        return cache

    @property
    def mean_radius(self) -> float:
        if self.radii.size == 0:
            return 1.0
        return float(np.nanmean(self.radii))


@dataclass(slots=True)
class VectorFieldResult:
    """Container storing drift vectors, Jacobians, and surface derivatives."""

    nside: int
    time_centers: np.ndarray
    drift: np.ndarray
    divergence: np.ndarray
    curl: np.ndarray
    jacobian: np.ndarray


def smooth_tracks(tracks: pd.DataFrame, smooth_cfg: SmoothingConfig) -> pd.DataFrame:
    """Apply Savitzky–Golay smoothing to Cartesian coordinates per track."""

    if tracks.empty:
        return tracks.copy()

    coord_cols = [c for c in ("x", "y", "z") if c in tracks.columns]
    if len(coord_cols) != 3:
        return tracks.copy()

    track_col = "track_id" if "track_id" in tracks.columns else None
    time_col = "frame" if "frame" in tracks.columns else "time_min" if "time_min" in tracks.columns else None

    if track_col is None or time_col is None:
        return tracks.copy()

    smoothed = tracks.copy()
    smoothed = smoothed.sort_values([track_col, time_col])

    for _, group in smoothed.groupby(track_col):
        idx = group.index
        n = len(group)
        if n < 2:
            continue
        window = min(smooth_cfg.sg_window_frames, n)
        if window % 2 == 0:
            window = max(3, window - 1)
        if window < 3:
            continue
        for col in coord_cols:
            smoothed.loc[idx, col] = savgol_filter(
                group[col].to_numpy(dtype=float),
                window_length=window,
                polyorder=min(smooth_cfg.sg_poly, window - 1),
                mode="interp",
            )
    return smoothed


def _angles_from_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, phi) for an array of 3D vectors."""

    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = vectors / norms[:, None]
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    return theta, phi


def build_step_table(tracks: pd.DataFrame) -> StepTable:
    """Construct a :class:`StepTable` from smoothed track positions."""

    if tracks.empty:
        empty = np.empty((0, 3), dtype=float)
        empty_scalar = np.empty((0,), dtype=float)
        return StepTable(
            track_ids=np.empty((0,), dtype=int),
            start_positions=empty.copy(),
            end_positions=empty.copy(),
            mid_positions=empty.copy(),
            displacements=empty.copy(),
            velocities=empty.copy(),
            dt=empty_scalar.copy(),
            mid_times=empty_scalar.copy(),
            unit_mid=empty.copy(),
            theta=empty_scalar.copy(),
            phi=empty_scalar.copy(),
            radii=empty_scalar.copy(),
        )

    required = {"x", "y", "z"}
    if not required.issubset(tracks.columns):
        raise ValueError("Tracks dataframe must contain Cartesian coordinates 'x', 'y', 'z'.")

    track_col = "particle" if "particle" in tracks.columns else None
    if track_col is None:
        raise ValueError("Tracks dataframe must contain a 'particle' column.")

    if "time_min" in tracks.columns:
        time_col = "time_min"
    elif "time" in tracks.columns:
        time_col = "time"
    elif "frame" in tracks.columns:
        time_col = "frame"
    else:
        raise ValueError("Tracks dataframe requires a temporal column (time_min/time/frame).")

    track_ids: list[int] = []
    start_positions: list[np.ndarray] = []
    end_positions: list[np.ndarray] = []
    mid_positions: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    mid_times: list[np.ndarray] = []

    for idx, (tid, group) in enumerate(tracks.sort_values([track_col, time_col]).groupby(track_col)):
        coords = group[["x", "y", "z"]].to_numpy(dtype=float)
        times = group[time_col].to_numpy(dtype=float)
        if coords.shape[0] < 2:
            continue
        dt = np.diff(times)
        disp = np.diff(coords, axis=0)
        valid = np.isfinite(dt) & (dt > 0)
        if not np.any(valid):
            continue
        dt = dt[valid]
        disp = disp[valid]
        starts = coords[:-1][valid]
        ends = coords[1:][valid]
        mids = 0.5 * (starts + ends)
        vel = disp / dt[:, None]
        t_mid = 0.5 * (times[:-1][valid] + times[1:][valid])

        n_steps = dt.size
        track_ids.append(np.full(n_steps, idx, dtype=int))
        start_positions.append(starts)
        end_positions.append(ends)
        mid_positions.append(mids)
        displacements.append(disp)
        velocities.append(vel)
        dt_list.append(dt)
        mid_times.append(t_mid)

    if not track_ids:
        empty = np.empty((0, 3), dtype=float)
        empty_scalar = np.empty((0,), dtype=float)
        return StepTable(
            track_ids=np.empty((0,), dtype=int),
            start_positions=empty.copy(),
            end_positions=empty.copy(),
            mid_positions=empty.copy(),
            displacements=empty.copy(),
            velocities=empty.copy(),
            dt=empty_scalar.copy(),
            mid_times=empty_scalar.copy(),
            unit_mid=empty.copy(),
            theta=empty_scalar.copy(),
            phi=empty_scalar.copy(),
            radii=empty_scalar.copy(),
        )

    track_ids_arr = np.concatenate(track_ids)
    start_arr = np.vstack(start_positions)
    end_arr = np.vstack(end_positions)
    mid_arr = np.vstack(mid_positions)
    disp_arr = np.vstack(displacements)
    vel_arr = np.vstack(velocities)
    dt_arr = np.concatenate(dt_list)
    mid_times_arr = np.concatenate(mid_times)

    radii = np.linalg.norm(mid_arr, axis=1)
    radii = np.where(radii == 0.0, 1.0, radii)
    unit_mid = mid_arr / radii[:, None]
    theta, phi = _angles_from_vectors(unit_mid)

    return StepTable(
        track_ids=track_ids_arr,
        start_positions=start_arr,
        end_positions=end_arr,
        mid_positions=mid_arr,
        displacements=disp_arr,
        velocities=vel_arr,
        dt=dt_arr,
        mid_times=mid_times_arr,
        unit_mid=unit_mid,
        theta=theta,
        phi=phi,
        radii=radii,
    )


def _weighted_affine_fit(
    coords: np.ndarray,
    velocities: np.ndarray,
    weights: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted affine regression in the local tangent frame."""

    coords2d = np.column_stack((coords @ e_theta, coords @ e_phi))
    vel2d = np.column_stack((velocities @ e_theta, velocities @ e_phi))
    design = np.concatenate((np.ones((coords2d.shape[0], 1)), coords2d), axis=1)
    w = np.sqrt(weights)[:, None]
    try:
        beta, *_ = np.linalg.lstsq(design * w, vel2d * w, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical guard
        beta = np.zeros((3, 2), dtype=float)
    drift_local = beta[0]
    jac_local = beta[1:].T
    return drift_local, jac_local


def compute_vector_field(
    tracks: pd.DataFrame,
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    smooth_cfg: SmoothingConfig,
    step_table: StepTable | None = None,
) -> dict[int, VectorFieldResult]:
    """Estimate coarse drift vectors and derivatives for each grid."""

    if step_table is None:
        step_table = build_step_table(tracks)

    results: dict[int, VectorFieldResult] = {}
    if step_table.mid_times.size == 0:
        for nside, result in binned.items():
            nt, npix = result.counts.shape
            drift = np.full((nt, npix, 3), np.nan, dtype=np.float32)
            divergence = np.full((nt, npix), np.nan, dtype=np.float32)
            curl = np.full((nt, npix), np.nan, dtype=np.float32)
            jac = np.full((nt, npix, 2, 2), np.nan, dtype=np.float32)
            results[nside] = VectorFieldResult(
                nside=nside,
                time_centers=result.time_centers,
                drift=drift,
                divergence=divergence,
                curl=curl,
                jacobian=jac,
            )
        return results

    time_sigma = _fwhm_to_sigma(win_cfg.coarse_minutes)

    for nside, grid_result in binned.items():
        nt, npix = grid_result.counts.shape
        drift = np.full((nt, npix, 3), np.nan, dtype=np.float32)
        divergence = np.full((nt, npix), np.nan, dtype=np.float32)
        curl = np.full((nt, npix), np.nan, dtype=np.float32)
        jacobian = np.full((nt, npix, 2, 2), np.nan, dtype=np.float32)

        if nt == 0 or npix == 0:
            results[nside] = VectorFieldResult(
                nside=nside,
                time_centers=grid_result.time_centers,
                drift=drift,
                divergence=divergence,
                curl=curl,
                jacobian=jacobian,
            )
            continue

        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))
        unit_mid = step_table.unit_mid
        step_times = step_table.mid_times
        coords = step_table.mid_positions
        velocities = step_table.velocities

        pixel_area = 4.0 * np.pi / healpix_nside2npix(nside)
        space_sigma = max(np.sqrt(pixel_area), 1e-3)

        for t_index, center_time in enumerate(grid_result.time_centers):
            time_weights = np.exp(-0.5 * ((step_times - center_time) / max(time_sigma, 1e-6)) ** 2)
            valid_time = time_weights > 1e-6
            if not np.any(valid_time):
                continue
            indices_time = np.nonzero(valid_time)[0]
            time_w = time_weights[indices_time]
            coords_t = coords[indices_time]
            velocities_t = velocities[indices_time]
            unit_t = unit_mid[indices_time]

            for pix in range(npix):
                if grid_result.counts[t_index, pix] == 0:
                    continue
                center_vec = pixel_vectors[pix]
                e_theta, e_phi = tangent_basis(center_vec)
                cosang = np.clip(unit_t @ center_vec, -1.0, 1.0)
                ang = np.arccos(cosang)
                spatial_weights = np.exp(-0.5 * (ang / space_sigma) ** 2)
                weights = spatial_weights * time_w
                valid = weights > 1e-6
                if np.count_nonzero(valid) < 6:
                    continue
                weights = weights[valid]
                coords_sel = coords_t[valid]
                velocities_sel = velocities_t[valid]

                drift_local, jac_local = _weighted_affine_fit(coords_sel - center_vec, velocities_sel, weights, e_theta, e_phi)
                drift_vec = drift_local[0] * e_theta + drift_local[1] * e_phi

                drift[t_index, pix] = drift_vec.astype(np.float32)
                divergence[t_index, pix] = np.float32(np.trace(jac_local))
                curl[t_index, pix] = np.float32(jac_local[1, 0] - jac_local[0, 1])
                jacobian[t_index, pix] = jac_local.astype(np.float32)

        results[nside] = VectorFieldResult(
            nside=nside,
            time_centers=grid_result.time_centers,
            drift=drift,
            divergence=divergence,
            curl=curl,
            jacobian=jacobian,
        )

    return results


__all__ = [
    "StepTable",
    "VectorFieldResult",
    "build_step_table",
    "compute_vector_field",
    "tangent_basis",
    "smooth_tracks",
]
