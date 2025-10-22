"""Utilities for constructing simplified HEALPix-style grids and binning tracks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GridConfig, WindowConfig

try:  # pragma: no cover - optional dependency
    import healpy as _hp

    def healpix_nside2npix(nside: int) -> int:
        return int(_hp.nside2npix(nside))

    def healpix_ang2pix(nside: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _hp.ang2pix(nside, theta, phi, nest=False)

    def healpix_pix2ang(nside: int, pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _hp.pix2ang(nside, pix, nest=False)

    def healpix_pix2vec(nside: int, pix: np.ndarray) -> np.ndarray:
        return np.asarray(_hp.pix2vec(nside, pix, nest=False)).T

except ModuleNotFoundError:  # pragma: no cover - fallback path
    from astropy_healpix import HEALPix
    import astropy.units as u

    def _healpix_obj(nside: int) -> HEALPix:
        return HEALPix(nside, order="ring")

    def healpix_nside2npix(nside: int) -> int:
        return 12 * int(nside) ** 2

    def healpix_ang2pix(nside: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        hp_obj = _healpix_obj(int(nside))
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        return np.asarray(hp_obj.angle_to_healpix(theta * u.rad, phi * u.rad), dtype=int)

    def healpix_pix2ang(nside: int, pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hp_obj = _healpix_obj(int(nside))
        pix = np.asarray(pix, dtype=int)
        lon, lat = hp_obj.healpix_to_lonlat(pix)
        phi = lon.to_value(u.rad)
        theta = (np.pi / 2.0) - lat.to_value(u.rad)
        return theta, phi

    def healpix_pix2vec(nside: int, pix: np.ndarray) -> np.ndarray:
        theta, phi = healpix_pix2ang(nside, pix)
        sin_theta = np.sin(theta)
        return np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)))


@dataclass(slots=True)
class HealpixIndexer:
    """Lightweight stand-in for HEALPix indexing.

    The implementation discretises :math:`\theta` and :math:`\phi` uniformly. It
    is not numerically equivalent to true HEALPix indexing, but captures the
    interface needed by the pipeline and can later be swapped out with a proper
    spherical pixelisation.
    """

    nside: int

    def __post_init__(self) -> None:
        self.npix = healpix_nside2npix(self.nside)

    def assign(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Return pixel indices for the provided angular coordinates."""

        theta = np.asarray(theta)
        phi = np.asarray(phi)
        theta = np.clip(theta, 0.0, np.pi)
        phi = np.mod(phi, 2 * np.pi)

        return healpix_ang2pix(self.nside, theta, phi)


@dataclass(slots=True)
class GridBinResult:
    """Aggregation container for per-grid binning results."""

    nside: int
    time_centers: np.ndarray
    counts: np.ndarray


def build_healpix_indexers(nsides: Iterable[int]) -> dict[int, HealpixIndexer]:
    """Create a :class:`HealpixIndexer` for each requested ``nside`` value."""

    return {int(nside): HealpixIndexer(int(nside)) for nside in nsides}


def _compute_time_centers(tracks: pd.DataFrame, win_cfg: WindowConfig, time_col: str) -> np.ndarray:
    times = tracks[time_col].to_numpy(dtype=float)
    if times.size == 0:
        return np.array([], dtype=float)

    t_min, t_max = times.min(), times.max()
    if not np.isfinite([t_min, t_max]).all():
        return np.array([], dtype=float)

    stride = max(1.0, float(win_cfg.stride_minutes))
    centers = np.arange(t_min, t_max + stride, stride)
    if centers.size == 0:
        centers = np.array([t_min])
    return centers.astype(float)


def _bin_single_grid(
    tracks: pd.DataFrame,
    indexer: HealpixIndexer,
    grid_cfg: GridConfig,
    win_cfg: WindowConfig,
) -> GridBinResult:
    time_col = grid_cfg.time_col
    theta_col, phi_col = grid_cfg.pos_cols_2d

    time_centers = _compute_time_centers(tracks, win_cfg, time_col)
    npix = indexer.npix
    counts = np.zeros((time_centers.size, npix), dtype=np.int32)

    if time_centers.size == 0 or tracks.empty:
        return GridBinResult(nside=indexer.nside, time_centers=time_centers, counts=counts)

    theta = tracks[theta_col].to_numpy(dtype=float)
    phi = tracks[phi_col].to_numpy(dtype=float)
    pixel_indices = indexer.assign(theta, phi)

    time_values = tracks[time_col].to_numpy(dtype=float)
    half_win = win_cfg.win_minutes / 2.0

    for i, center in enumerate(time_centers):
        mask = np.abs(time_values - center) <= half_win
        if not np.any(mask):
            continue
        counts[i] = np.bincount(pixel_indices[mask], minlength=npix).astype(np.int32)

    return GridBinResult(nside=indexer.nside, time_centers=time_centers, counts=counts)


def bin_tracks_over_time(
    tracks: pd.DataFrame,
    indexers: dict[int, HealpixIndexer],
    grid_cfg: GridConfig,
    win_cfg: WindowConfig,
) -> dict[int, GridBinResult]:
    """Bin tracks for each grid configuration across temporal windows."""

    results: dict[int, GridBinResult] = {}
    for nside, indexer in indexers.items():
        results[nside] = _bin_single_grid(tracks, indexer, grid_cfg, win_cfg)
    return results


__all__ = [
    "HealpixIndexer",
    "GridBinResult",
    "build_healpix_indexers",
    "bin_tracks_over_time",
    "healpix_nside2npix",
    "healpix_ang2pix",
    "healpix_pix2ang",
    "healpix_pix2vec",
]
