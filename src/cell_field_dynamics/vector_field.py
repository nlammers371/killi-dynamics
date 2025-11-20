"""
Vector-field estimation utilities for the cell-dynamics pipeline.
REFORMATTED: fixed-time windows + neighbor-based spatial smoothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from src.cell_field_dynamics.config import WindowConfig, SmoothingConfig
from src.cell_field_dynamics.grids import (
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
    """Container for per-step kinematic and scalar quantities on the sphere."""

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

    # NEW OPTIONAL PER-STEP FLUORESCENCE FIELD
    fluo: Optional[np.ndarray] = None

    pixel_cache: Dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def pixel_indices(self, nside: int) -> np.ndarray:
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




def _angles_from_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, phi) for an array of 3D vectors."""

    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = vectors / norms[:, None]
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    return theta, phi


def build_step_table(
    tracks: pd.DataFrame,
    *,
    fluo_col: Optional[str] = None,
) -> StepTable:
    """Construct a StepTable from smoothed track positions."""

    if tracks.empty:
        empty = np.empty((0, 3), dtype=float)
        empty_s = np.empty((0,), dtype=float)
        return StepTable(
            track_ids=np.empty((0,), dtype=int),
            start_positions=empty,
            end_positions=empty,
            mid_positions=empty,
            displacements=empty,
            velocities=empty,
            dt=empty_s,
            mid_times=empty_s,
            unit_mid=empty,
            theta=empty_s,
            phi=empty_s,
            radii=empty_s,
            fluo=None,
        )

    required = {"x", "y", "z"}
    if not required.issubset(tracks.columns):
        raise ValueError("Tracks dataframe must contain x,y,z.")

    if "track_id" in tracks.columns:
        track_col = "track_id"
    else:
        raise ValueError("Tracks dataframe must contain track_id.")

    if "time_min" in tracks.columns:
        time_col = "time_min"
    elif "t" in tracks.columns:
        time_col = "t"
    else:
        raise ValueError("Tracks dataframe requires time_min or t column.")

    # collectors
    track_ids = []
    start_positions = []
    end_positions = []
    mid_positions = []
    displacements = []
    velocities = []
    dt_list = []
    mid_times = []
    fluo_list = [] if fluo_col else None

    # ----------------------------------------------
    # main loop over tracklets
    # ----------------------------------------------
    for idx, (tid, group) in enumerate(
        tqdm(tracks.sort_values([track_col, time_col]).groupby(track_col), desc="Building step table")
    ):
        coords = (
            group[["x", "y", "z"]].to_numpy(float)
            - group[["center_x_smooth", "center_y_smooth", "center_z_smooth"]].to_numpy(float)
        )
        times = group[time_col].to_numpy(float)

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

        if fluo_col:
            vals = group[fluo_col].to_numpy(float)
            # midpoints => average of fluorescence for the two frames
            fluo_vals = 0.5 * (vals[:-1][valid] + vals[1:][valid])
            fluo_list.append(fluo_vals)

    if not track_ids:
        empty = np.empty((0, 3), float)
        empty_s = np.empty((0,), float)
        return StepTable(
            track_ids=np.empty((0,), int),
            start_positions=empty,
            end_positions=empty,
            mid_positions=empty,
            displacements=empty,
            velocities=empty,
            dt=empty_s,
            mid_times=empty_s,
            unit_mid=empty,
            theta=empty_s,
            phi=empty_s,
            radii=empty_s,
            fluo=None,
        )

    # ------------------------------------------
    # concatenate all pieces
    # ------------------------------------------
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

    if fluo_col:
        fluo_arr = np.concatenate(fluo_list)
    else:
        fluo_arr = None

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
        fluo=fluo_arr,
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

def _compute_single_timebin_neighbors(
    t_index: int,
    *,
    center_times: np.ndarray,
    step_times: np.ndarray,
    unit_mid: np.ndarray,
    coords: np.ndarray,
    velocities: np.ndarray,
    pix_indices: np.ndarray,
    neighbors: list[np.ndarray],
    pixel_vectors: np.ndarray,
    space_sigma: float,
    half_window: float,
    mean_radius: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute drift/div/curl/J for a single time bin using:
      - hard temporal window |t - center_time| <= half_window
      - neighbor prefilter in HEALPix space
      - spatial Gaussian weights within that neighbor set

    This is designed to be called via functools.partial with everything
    bound except `t_index`, e.g.:

        calculate = partial(
            _compute_single_timebin_neighbors,
            center_times=center_times,
            step_times=step_table.mid_times,
            unit_mid=step_table.unit_mid,
            coords=step_table.mid_positions,
            velocities=step_table.velocities,
            pix_indices=step_table.pixel_indices(nside),
            neighbors=neighbor_indexers[nside],
            pixel_vectors=pixel_vectors,
            space_sigma=space_sigma,
            half_window=half_window,
            mean_radius=step_table.mean_radius,
        )
    """

    center_time = center_times[t_index]
    npix = pixel_vectors.shape[0]

    # Output rows for this time bin
    drift_row = np.full((npix, 3), np.nan, dtype=np.float32)
    div_row   = np.full(npix, np.nan, dtype=np.float32)
    curl_row  = np.full(npix, np.nan, dtype=np.float32)
    jac_row   = np.full((npix, 2, 2), np.nan, dtype=np.float32)

    # ---- temporal window ----
    mask_t = np.abs(step_times - center_time) <= half_window
    if not np.any(mask_t):
        return t_index, drift_row, div_row, curl_row, jac_row

    idx_t   = np.nonzero(mask_t)[0]
    unit_t  = unit_mid[idx_t]         # (Nt, 3)
    coords_t = coords[idx_t]          # (Nt, 3)
    vel_t   = velocities[idx_t]       # (Nt, 3)
    pix_t   = pix_indices[idx_t]      # (Nt,)

    # Only bother with pixels that actually see any steps in this time window
    for pix in np.unique(pix_t):
        center_vec = pixel_vectors[pix]

        # ---- restrict to neighbor pixels FIRST ----
        neigh = neighbors[pix]  # array of pixel IDs for this center
        if neigh.size == 0:
            continue

        mask_local_pix = np.isin(pix_t, neigh)
        if not np.any(mask_local_pix):
            continue

        # restrict to steps inside time window AND in neighbor pixels
        local_idx = np.nonzero(mask_local_pix)[0]   # indices into idx_t
        unit_loc   = unit_t[local_idx]
        coords_loc = coords_t[local_idx]
        vel_loc    = vel_t[local_idx]

        # ---- spatial Gaussian within neighbor set ----
        # angular distance between step position and pixel center
        cosang = np.clip(unit_loc @ center_vec, -1.0, 1.0)
        ang    = np.arccos(cosang)
        w_space = np.exp(-0.5 * (ang / space_sigma) ** 2)

        keep = w_space > 1e-3
        if np.count_nonzero(keep) < 6:
            continue

        w          = w_space[keep]
        coords_use = coords_loc[keep]
        vel_use    = vel_loc[keep]

        # ---- local tangent frame at pixel center ----
        e_theta, e_phi = tangent_basis(center_vec)

        # center coordinates on sphere
        patch_center_xyz = center_vec * mean_radius
        coords_centered  = coords_use - patch_center_xyz[None, :]

        # ---- fit weighted affine flow in tangent plane ----
        drift_loc, jac_loc = _weighted_affine_fit(
            coords_centered,
            vel_use,
            w,
            e_theta,
            e_phi,
        )

        # map drift back to 3D
        drift_vec = drift_loc[0] * e_theta + drift_loc[1] * e_phi

        drift_row[pix] = drift_vec.astype(np.float32)
        div_row[pix]   = np.float32(np.trace(jac_loc))
        curl_row[pix]  = np.float32(jac_loc[1, 0] - jac_loc[0, 1])
        jac_row[pix]   = jac_loc.astype(np.float32)

    return t_index, drift_row, div_row, curl_row, jac_row


def compute_vector_field(
    tracks: pd.DataFrame,
    binned: dict[int, GridBinResult],
    win_cfg: WindowConfig,
    smooth_cfg: SmoothingConfig,
    step_table: StepTable,
    neighbors: Optional[Dict[int, list[np.ndarray]]] = None,
    min_steps: int = 5,
    n_workers: int = 1,
) -> dict[int, VectorFieldResult]:

    # Build step table if needed
    if step_table is None:
        step_table = build_step_table(tracks)

    results: dict[int, VectorFieldResult] = {}

    # trivial empty case
    if step_table.mid_times.size == 0:
        for nside, gr in binned.items():
            nt, npix = gr.counts.shape
            drift = np.full((nt, npix, 3), np.nan, np.float32)
            div   = np.full((nt, npix),    np.nan, np.float32)
            curl  = np.full((nt, npix),    np.nan, np.float32)
            jac   = np.full((nt, npix, 2, 2), np.nan, np.float32)
            results[nside] = VectorFieldResult(
                nside=nside,
                time_centers=gr.time_centers,
                drift=drift,
                divergence=div,
                curl=curl,
                jacobian=jac,
            )
        return results

    # Reusable references
    step_times = step_table.mid_times
    unit_mid   = step_table.unit_mid
    coords     = step_table.mid_positions
    vel        = step_table.velocities
    mean_R     = step_table.mean_radius

    # spatial sigma already in radians
    space_sigma = smooth_cfg.sigma_radians
    half_window = win_cfg.win_minutes / 2.0

    for nside, gr in tqdm(binned.items(), desc="Vector field levels"):
        nt, npix = gr.counts.shape

        drift = np.full((nt, npix, 3), np.nan, np.float32)
        div   = np.full((nt, npix),    np.nan, np.float32)
        curl  = np.full((nt, npix),    np.nan, np.float32)
        jac   = np.full((nt, npix, 2, 2), np.nan, np.float32)

        # No data → store empty container
        if nt == 0 or npix == 0:
            results[nside] = VectorFieldResult(
                nside=nside,
                time_centers=gr.time_centers,
                drift=drift,
                divergence=div,
                curl=curl,
                jacobian=jac,
            )
            continue

        # Safety check
        if neighbors is None or nside not in neighbors:
            raise ValueError(f"neighbors[{nside}] missing for vector field")

        neigh_nside = neighbors[nside]

        # Healpix geometry
        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))
        pix_indices = step_table.pixel_indices(nside)

        # Build partial for multiprocessing
        calculate_single = partial(
            _compute_single_timebin_neighbors,
            center_times=gr.time_centers,
            step_times=step_times,
            unit_mid=unit_mid,
            coords=coords,
            velocities=vel,
            pix_indices=pix_indices,
            neighbors=neigh_nside,
            pixel_vectors=pixel_vectors,
            space_sigma=space_sigma,
            half_window=half_window,
            mean_radius=mean_R,
        )

        tasks = list(range(nt))

        # Run worker pool
        if n_workers == 1:
            outputs = [
                calculate_single(t_idx)
                for t_idx in tqdm(tasks, desc=f"nside={nside}")
            ]
        else:
            outputs = process_map(
                calculate_single,
                tasks,
                max_workers=n_workers,
                chunksize=1,
                desc=f"vector nside={nside}",
            )

        # Unpack
        for t_idx, drift_row, div_row, curl_row, jac_row in outputs:
            drift[t_idx] = drift_row
            div[t_idx]   = div_row
            curl[t_idx]  = curl_row
            jac[t_idx]   = jac_row

        # Store result
        results[nside] = VectorFieldResult(
            nside=nside,
            time_centers=gr.time_centers,
            drift=drift,
            divergence=div,
            curl=curl,
            jacobian=jac,
        )

    return results



# def compute_vector_field(
#     tracks: pd.DataFrame,
#     binned: dict[int, GridBinResult],
#     win_cfg: WindowConfig,
#     step_table: StepTable | None = None,
# ) -> dict[int, VectorFieldResult]:
#     """Estimate coarse drift vectors and derivatives for each grid."""
#
#     if step_table is None:
#         step_table = build_step_table(tracks)
#
#     results: dict[int, VectorFieldResult] = {}
#     if step_table.mid_times.size == 0:
#         for nside, result in tqdm(binned.items(), desc="Initializing vector fields"):
#             nt, npix = result.counts.shape
#             drift = np.full((nt, npix, 3), np.nan, dtype=np.float32)
#             divergence = np.full((nt, npix), np.nan, dtype=np.float32)
#             curl = np.full((nt, npix), np.nan, dtype=np.float32)
#             jac = np.full((nt, npix, 2, 2), np.nan, dtype=np.float32)
#             results[nside] = VectorFieldResult(
#                 nside=nside,
#                 time_centers=result.time_centers,
#                 drift=drift,
#                 divergence=divergence,
#                 curl=curl,
#                 jacobian=jac,
#             )
#         return results
#
#     time_sigma = _fwhm_to_sigma(win_cfg.coarse_minutes)
#
#     for nside, grid_result in tqdm(binned.items(), desc="Computing vector fields"):
#         nt, npix = grid_result.counts.shape
#         drift = np.full((nt, npix, 3), np.nan, dtype=np.float32)
#         divergence = np.full((nt, npix), np.nan, dtype=np.float32)
#         curl = np.full((nt, npix), np.nan, dtype=np.float32)
#         jacobian = np.full((nt, npix, 2, 2), np.nan, dtype=np.float32)
#
#         if nt == 0 or npix == 0:
#             results[nside] = VectorFieldResult(
#                 nside=nside,
#                 time_centers=grid_result.time_centers,
#                 drift=drift,
#                 divergence=divergence,
#                 curl=curl,
#                 jacobian=jacobian,
#             )
#             continue
#
#         pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))
#         unit_mid = step_table.unit_mid
#         step_times = step_table.mid_times
#         coords = step_table.mid_positions
#         velocities = step_table.velocities
#
#         pixel_area = 4.0 * np.pi / healpix_nside2npix(nside)
#         space_sigma = max(2*np.sqrt(pixel_area), 1e-3)
#
#         for t_index, center_time in enumerate(grid_result.time_centers):
#             time_weights = np.exp(-0.5 * ((step_times - center_time) / max(time_sigma, 1e-6)) ** 2)
#             valid_time = time_weights > 1e-6
#             if not np.any(valid_time):
#                 continue
#             indices_time = np.nonzero(valid_time)[0]
#             time_w = time_weights[indices_time]
#             coords_t = coords[indices_time]
#             velocities_t = velocities[indices_time]
#             unit_t = unit_mid[indices_time]
#
#             for pix in range(npix):
#                 if grid_result.counts[t_index, pix] == 0:
#                     continue
#                 center_vec = pixel_vectors[pix]
#                 e_theta, e_phi = tangent_basis(center_vec)
#                 cosang = np.clip(unit_t @ center_vec, -1.0, 1.0)
#                 ang = np.arccos(cosang)
#                 spatial_weights = np.exp(-0.5 * (ang / space_sigma) ** 2)
#                 weights = spatial_weights * time_w
#                 valid = weights > 1e-6
#                 if np.count_nonzero(valid) < 6:
#                     continue
#                 weights = weights[valid]
#                 coords_sel = coords_t[valid]
#                 velocities_sel = velocities_t[valid]
#
#                 patch_center_xyz = center_vec * step_table.mean_radius
#                 coords_centered = coords_sel - patch_center_xyz
#
#
#                 drift_local, jac_local = _weighted_affine_fit(coords_centered, velocities_sel, weights, e_theta, e_phi)
#                 drift_vec = drift_local[0] * e_theta + drift_local[1] * e_phi
#
#                 drift[t_index, pix] = drift_vec.astype(np.float32)
#                 divergence[t_index, pix] = np.float32(np.trace(jac_local))
#                 curl[t_index, pix] = np.float32(jac_local[1, 0] - jac_local[0, 1])
#                 jacobian[t_index, pix] = jac_local.astype(np.float32)
#
#         results[nside] = VectorFieldResult(
#             nside=nside,
#             time_centers=grid_result.time_centers,
#             drift=drift,
#             divergence=divergence,
#             curl=curl,
#             jacobian=jacobian,
#         )
#
#     return results


__all__ = [
    "StepTable",
    "VectorFieldResult",
    "build_step_table",
    "compute_vector_field",
    "tangent_basis",
    # "smooth_tracks",
]
