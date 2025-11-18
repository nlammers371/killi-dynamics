import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from tqdm import tqdm
from src.cell_field_dynamics.grids import (
    GridBinResult,
    healpix_nside2npix,
    healpix_pix2vec,
)
from src.cell_field_dynamics.vector_field import StepTable
from src.cell_field_dynamics.config import WindowConfig


@dataclass(slots=True)
class DensityFieldResult:
    density: np.ndarray                 # (nt, npix)
    mean_bc1: Optional[np.ndarray]      # (nt, npix) or None


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM (minutes) to Gaussian sigma (same units)."""
    fwhm = float(fwhm)
    if fwhm <= 0:
        return 1.0
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def compute_healpix_density_field(
    step_table: StepTable,
    binned: Dict[int, GridBinResult],
    win_cfg: WindowConfig,
    *,
    fluo: Optional[np.ndarray] = None,
    sphere_radius: Optional[float] = None,
    min_steps_per_bin: int = 1,
) -> Dict[int, DensityFieldResult]:
    """
    Compute HEALPix pixel-level density and (optionally) mean fluorescence
    with **Gaussian spatiotemporal smoothing**, consistent with vector_field.
    """

    # --- pick fluorescence source ---
    if fluo is None and getattr(step_table, "fluo", None) is not None:
        fluo = step_table.fluo

    times = step_table.mid_times          # (N_steps,)
    unit_mid = step_table.unit_mid        # (N_steps, 3)
    N = times.size

    if sphere_radius is None:
        sphere_radius = step_table.mean_radius

    if fluo is not None and fluo.shape[0] != N:
        raise ValueError(
            "fluo array must match number of steps in step_table.mid_times"
        )

    # --- temporal smoothing scale (match vector_field style) ---
    # Use coarse_minutes if present, else fall back to win_minutes
    time_fwhm = getattr(win_cfg, "coarse_minutes", None)
    if time_fwhm is None:
        time_fwhm = win_cfg.win_minutes
    time_sigma = _fwhm_to_sigma(time_fwhm)
    time_sigma = max(time_sigma, 1e-6)

    results: Dict[int, DensityFieldResult] = {}

    for nside, grid_result in tqdm(binned.items(), desc="Density HEALPix levels"):
        nt, npix = grid_result.counts.shape

        # per-step pixel assignment for this resolution (not directly used for smoothing now,
        # but could be used for masking or QC if desired)
        pix_indices = step_table.pixel_indices(nside)  # (N,)

        # outputs
        dens = np.full((nt, npix), np.nan, dtype=np.float32)
        mbc = None
        if fluo is not None:
            mbc = np.full((nt, npix), np.nan, dtype=np.float32)

        # --- geometry: pixel area and angular smoothing scale ---
        npix_total = healpix_nside2npix(nside)
        pixel_area_phys = 4.0 * np.pi * (sphere_radius ** 2) / npix_total
        pixel_area_unit = 4.0 * np.pi / npix_total    # area on unit sphere

        # spatial sigma in radians (same trick as vector_field)
        space_sigma = max(2.0 * np.sqrt(pixel_area_unit), 1e-3)

        # HEALPix pixel center vectors on unit sphere
        pixel_vectors = healpix_pix2vec(nside, np.arange(npix, dtype=int))  # (npix,3)

        for t_idx, center_time in enumerate(grid_result.time_centers):
            # ---- temporal weights for ALL steps ----
            dt = times - center_time
            w_time = np.exp(-0.5 * (dt / time_sigma) ** 2)  # (N,)

            valid_t = w_time > 1e-6
            if not np.any(valid_t):
                continue

            idx_time = np.nonzero(valid_t)[0]
            unit_t = unit_mid[idx_time]            # (M,3)
            w_time_t = w_time[idx_time]            # (M,)
            fluo_t = fluo[idx_time] if fluo is not None else None

            # optional: you can gate by discrete counts if you want consistency with binned counts
            # but it's not strictly necessary for the smoothing
            for pix in range(npix):
                # skip pixels that have no discrete counts in this window if you want to
                if grid_result.counts[t_idx, pix] == 0:
                    continue

                center_vec = pixel_vectors[pix]     # (3,)
                # angular distance to all steps in this time window
                cosang = np.clip(unit_t @ center_vec, -1.0, 1.0)  # (M,)
                ang = np.arccos(cosang)                           # (M,)

                w_space = np.exp(-0.5 * (ang / space_sigma) ** 2)  # (M,)
                weights = w_space * w_time_t                      # (M,)

                valid = weights > 1e-6
                if np.count_nonzero(valid) < min_steps_per_bin:
                    continue

                w = weights[valid]

                # ---- density: weighted "number of steps" per physical area ----
                dens[t_idx, pix] = float(w.sum() / pixel_area_phys)

                # ---- mean fluorescence: weighted by same kernel ----
                if mbc is not None and fluo_t is not None:
                    mbc[t_idx, pix] = float(
                        np.sum(w * fluo_t[valid]) / np.sum(w)
                    )

        results[nside] = DensityFieldResult(density=dens, mean_bc1=mbc)

    return results
