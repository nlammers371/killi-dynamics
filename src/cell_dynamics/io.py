"""I/O helpers for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .config import GridConfig, SmoothingConfig, WindowConfig, NoiseConfig
from .grids import GridBinResult, HealpixIndexer
from .materials import MaterialMetrics
from .metrics import MetricCollection
from .msd import MSDResult
from .qc import QCResult
from .vector_field import VectorFieldResult


METRIC_GROUPS = {
    "path_speed": ("path_speed", "um/min"),
    "drift_speed": ("drift_speed", "um/min"),
    "theta_entropy": ("theta_entropy", "dimensionless"),
    "diffusivity_total": ("diffusivity_total", "um^2/min"),
    "diffusivity_idio": ("diffusivity_idio", "um^2/min"),
}


@dataclass(slots=True)
class ZarrDescriptor:
    nside: int
    store_path: Path


def _write_root_attrs(
    root: zarr.Group,
    nside: int,
    grid_cfg: GridConfig,
    win_cfg: WindowConfig,
    smooth_cfg: SmoothingConfig,
    noise_cfg: NoiseConfig,
) -> None:
    root.attrs.update(
        {
            "nside": int(nside),
            "win_minutes": float(win_cfg.win_minutes),
            "stride_minutes": float(win_cfg.stride_minutes),
            "coarse_minutes": float(win_cfg.coarse_minutes),
            "fine_minutes": float(win_cfg.fine_minutes),
            "sg_window_frames": int(smooth_cfg.sg_window_frames),
            "sg_poly": int(smooth_cfg.sg_poly),
            "sigma_loc_um": (noise_cfg.fixed_sigma_loc_um or 0.0),
            "sigma_loc_method": noise_cfg.method,
        }
    )


def _write_array(group: zarr.Group, name: str, data: np.ndarray, *, units: str) -> None:
    array = group.require_dataset(
        name,
        shape=data.shape,
        dtype=data.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
        overwrite=True,
    )
    array[:] = data
    array.attrs["units"] = units


def write_zarr_stores(
    out_root: Path,
    indexers: dict[int, HealpixIndexer],
    binned: dict[int, GridBinResult],
    vector_results: dict[int, VectorFieldResult],
    metric_results: dict[int, MetricCollection],
    msd_results: dict[int, MSDResult],
    material_results: dict[int, MaterialMetrics],
    flux_results: dict[int, dict[str, np.ndarray]],
    qc_results: dict[int, QCResult],
    grid_cfg: GridConfig,
    win_cfg: WindowConfig,
    smooth_cfg: SmoothingConfig,
    noise_cfg: NoiseConfig,
) -> dict[int, Path]:
    """Write out Zarr stores for each grid."""

    zarr_paths: dict[int, Path] = {}
    for nside, indexer in indexers.items():
        store_path = Path(out_root) / f"fields_nside{nside:04d}.zarr"
        store = zarr.DirectoryStore(store_path)
        root = zarr.group(store=store, overwrite=True)

        _write_root_attrs(root, nside, grid_cfg, win_cfg, smooth_cfg, noise_cfg)

        metrics_group = root.require_group("metrics")
        scalar_metrics = metric_results.get(nside)
        if scalar_metrics:
            for key, (dataset_name, units) in METRIC_GROUPS.items():
                data = scalar_metrics.data.get(key)
                if data is not None:
                    _write_array(metrics_group, dataset_name, data.astype(np.float32), units=units)

        vf = vector_results.get(nside)
        if vf:
            drift_group = root.require_group("drift")
            _write_array(drift_group, "vector", vf.drift.astype(np.float32), units="um/min")
            _write_array(drift_group, "divergence", vf.divergence.astype(np.float32), units="1/min")
            _write_array(drift_group, "curl", vf.curl.astype(np.float32), units="1/min")

        msd_res = msd_results.get(nside)
        if msd_res:
            msd_group = root.require_group("msd")
            _write_array(msd_group, "alpha", msd_res.msd_alpha.astype(np.float32), units="dimensionless")
            _write_array(msd_group, "value", msd_res.msd_value.astype(np.float32), units="um^2")

        mat_res = material_results.get(nside)
        if mat_res:
            mat_group = root.require_group("materials")
            _write_array(mat_group, "cmsd_alpha", mat_res.cmsd_alpha.astype(np.float32), units="dimensionless")
            _write_array(mat_group, "d2min_short", mat_res.d2min_short.astype(np.float32), units="a.u.")
            _write_array(mat_group, "d2min_long", mat_res.d2min_long.astype(np.float32), units="a.u.")

        flux_res = flux_results.get(nside)
        if flux_res:
            flux_group = root.require_group("flux")
            _write_array(flux_group, "net", flux_res["net"].astype(np.float32), units="1/min")
            _write_array(flux_group, "throughput", flux_res["throughput"].astype(np.float32), units="um/min")

        qc_res = qc_results.get(nside)
        if qc_res:
            qc_group = root.require_group("qc")
            for key, mask in qc_res.masks.items():
                _write_array(qc_group, f"{key}_valid", mask.astype(bool), units="bool")
            for key, count in qc_res.counts.items():
                _write_array(qc_group, f"{key}_counts", count.astype(np.int32), units="count")

        zarr_paths[nside] = store_path

    return zarr_paths


TRACK_COLUMNS = {
    "path_speed_um_min": "path_speed",
    "drift_speed_um_min_at_cell": "drift_speed",
    "theta_entropy_local": "theta_entropy",
    "D_total_um2_min": "diffusivity_total",
    "D_idio_um2_min": "diffusivity_idio",
    "msd_alpha_short": "msd_alpha",
    "cmsd_alpha": "cmsd_alpha",
    "d2min_rate_short": "d2min_short",
    "d2min_rate_med": "d2min_long",
    "divergence_local": "divergence",
    "curl_local": "curl",
}


def augment_tracks_df(
    tracks: pd.DataFrame,
    indexers: dict[int, HealpixIndexer],
    metric_results: dict[int, MetricCollection],
    msd_results: dict[int, MSDResult],
    material_results: dict[int, MaterialMetrics],
    flux_results: dict[int, dict[str, np.ndarray]],
    qc_results: dict[int, QCResult],
    grid_cfg: GridConfig,
    out_root: Path,
    *,
    overwrite: bool,
) -> tuple[pd.DataFrame, Path | None]:
    """Augment the tracks dataframe with per-cell columns and persist it."""

    augmented = tracks.copy()

    for column in TRACK_COLUMNS:
        augmented[column] = np.nan

    path: Path | None = None
    if not augmented.empty:
        path = Path(out_root) / "tracks_augmented.csv"
        if overwrite or not path.exists():
            augmented.to_csv(path, index=False)

    return augmented, path


__all__ = ["write_zarr_stores", "augment_tracks_df"]
