import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from tqdm import tqdm
from src.cell_field_dynamics.config import SmoothingConfig
from src.cell_field_dynamics.grids import (
    GridBinResult,
    healpix_nside2npix,
    healpix_pix2vec,
)
from src.cell_field_dynamics.vector_field import StepTable
from src.cell_field_dynamics.config import WindowConfig
from functools import partial
from tqdm.contrib.concurrent import process_map

@dataclass(slots=True)
class DensityFieldResult:
    density: np.ndarray                 # (nt, npix)
    mean_fluo: Optional[np.ndarray]      # (nt, npix) or None


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM (minutes) to Gaussian sigma (same units)."""
    fwhm = float(fwhm)
    if fwhm <= 0:
        return 1.0
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def _compute_density_single_frame(
    t_idx,
    center_times,
    times,
    unit_mid,
    fluo,
    pix_indices,          # (N,) step → pixel index at this nside
    neighbors,            # list[np.ndarray], neighbors[pix] = array of pixel IDs
    npix,
    pixel_vectors,
    pixel_area_phys,
    space_sigma,          # radians
    half_window,          # minutes
    min_steps_per_bin,
):
    """Compute density for one time center t_idx using **fixed time window**."""

    center_t = center_times[t_idx]

    # Fixed time window, matching metrics module
    valid_t = np.abs(times - center_t) <= half_window
    if not np.any(valid_t):
        return t_idx, None, None

    idx_time = np.nonzero(valid_t)[0]
    unit_t = unit_mid[idx_time]                 # (Nt, 3)
    pix_t = pix_indices[idx_time]               # (Nt,)
    fluo_t = fluo[idx_time] if fluo is not None else None
    times_t = times[idx_time]

    dens_row = np.full(npix, 0.0, dtype=np.float32)
    mbc_row = np.full(npix, np.nan, dtype=np.float32) if fluo is not None else None

    # Loop over pixels
    for pix in range(npix):
        center_vec = pixel_vectors[pix]

        # Restrict to neighbor pixels only
        neigh = neighbors[pix]                  # array of pixel IDs
        mask_local = np.isin(pix_t, neigh)
        if not np.any(mask_local):
            continue

        local_idx = np.nonzero(mask_local)[0]
        unit_loc = unit_t[local_idx]
        fluo_loc = fluo_t[local_idx] if fluo_t is not None else None
        time_loc = times_t[local_idx]

        # spatial Gaussian on angular distance
        cosang = np.clip(unit_loc @ center_vec, -1.0, 1.0)
        ang = np.arccos(cosang)
        w_space = np.exp(-0.5 * (ang / space_sigma)**2)

        valid_w = w_space > 1e-2
        if np.count_nonzero(valid_w) < min_steps_per_bin:
            continue

        w = w_space[valid_w]
        nt = len(np.unique(time_loc[valid_w]))
        # Density
        dens_row[pix] = float(w.sum() / pixel_area_phys) / nt

        # Fluorescence
        if mbc_row is not None:
            mbc_row[pix] = float(
                np.sum(w * fluo_loc[valid_w]) / np.sum(w)
            )

    return t_idx, dens_row, mbc_row




def compute_healpix_density_field(
    step_table: StepTable,
    binned: Dict[int, GridBinResult],
    win_cfg: WindowConfig,
    smooth_cfg: SmoothingConfig,
    fluo: Optional[np.ndarray] = None,
    neighbors: Optional[Dict[int, list[np.ndarray]]] = None,
    min_steps_per_bin: int = 1,
    n_workers: int = 1,
) -> Dict[int, DensityFieldResult]:

    # Pick fluorescence source
    if fluo is None and getattr(step_table, "fluo", None) is not None:
        fluo = step_table.fluo

    times = step_table.mid_times
    unit_mid = step_table.unit_mid
    N = times.size

    sphere_radius = smooth_cfg.sphere_radius_um
    space_sigma = smooth_cfg.sigma_radians
    # if sphere_radius is None:
    #     sphere_radius = step_table.mean_radius

    if fluo is not None and fluo.shape[0] != N:
        raise ValueError("fluo length mismatch")

    # Fixed sliding window width (minutes)
    half_window = win_cfg.win_minutes / 2.0

    results = {}

    for nside, grid_result in tqdm(binned.items(), desc="Density HEALPix levels"):
        nt, npix = grid_result.counts.shape

        pix_indices = step_table.pixel_indices(nside)

        dens = np.full((nt, npix), np.nan, dtype=np.float32)
        mbc = None
        if fluo is not None:
            mbc = np.full((nt, npix), np.nan, dtype=np.float32)

        # geometry
        npix_total = healpix_nside2npix(nside)
        pixel_area_phys = 4.0 * np.pi * (sphere_radius ** 2) / npix_total

        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))

        if neighbors is None or nside not in neighbors:
            raise ValueError("neighbors[nside] missing")

        neighbors_nside = neighbors[nside]

        run_density_calc = partial(
            _compute_density_single_frame,
            center_times=grid_result.time_centers,
            times=times,
            unit_mid=unit_mid,
            fluo=fluo,
            pix_indices=pix_indices,
            neighbors=neighbors_nside,
            npix=npix,
            pixel_vectors=pixel_vectors,
            pixel_area_phys=pixel_area_phys,
            space_sigma=space_sigma,
            half_window=half_window,
            min_steps_per_bin=min_steps_per_bin,
        )

        if n_workers > 1:
            outputs = process_map(
                run_density_calc, range(nt),
                max_workers=n_workers,
                chunksize=1,
                desc=f"density nside={nside}"
            )
        else:
            outputs = []
            for t in tqdm(range(nt), desc=f"density nside={nside}"):
                outputs.append(run_density_calc(t))

        for t_idx, dens_row, mbc_row in outputs:
            if dens_row is None:
                continue
            dens[t_idx] = dens_row
            if mbc is not None and mbc_row is not None:
                mbc[t_idx] = mbc_row

        results[nside] = DensityFieldResult(density=dens, mean_fluo=mbc)

    return results

