"""Configuration dataclasses for the :mod:`cell_field_dynamics` pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class GridConfig:
    """Parameters controlling HEALPix grid construction and indexing."""

    nsides: Tuple[int, int] = (8, 16, 32, 64, 128)
    frame_col: str = "frame"
    time_col: str = "time_min"
    xyz_cols: Tuple[str, str, str] = ("x", "y", "z")
    pos_cols_2d: Tuple[str, str] = ("theta", "phi")


@dataclass(slots=True)
class WindowConfig:
    """Parameters describing temporal windowing of the tracks."""

    win_minutes: float = 60.0  # window size for binned dynamics
    stride_minutes: float = 15.0  # stride between windows
    coarse_minutes: float = 45.0  # ??
    fine_minutes: float = 15.0  # ??


@dataclass(slots=True)
class SmoothingConfig:
    """Parameters for Savitzky–Golay smoothing of particle tracks."""

    sg_window_minutes: float = 15.0
    sg_poly: int = 2


@dataclass(slots=True)
class QCConfig:
    """Quality-control thresholds applied to per-pixel aggregates."""

    min_steps_drift: int = 50
    min_steps_derivatives: int = 150
    min_pairs_msd: int = 100
    min_cells_region: int = 8


@dataclass(slots=True)
class NoiseConfig:
    """Settings for localisation noise estimation."""

    method: str = "cve_cov"  # "cve_cov" | "msd_intercept" | "fixed"
    fixed_sigma_loc_um: float | None = None


@dataclass(slots=True)
class MaterialsConfig:
    """Parameters controlling material-response metrics."""

    knn_neighbors: int = 10
    d2min_deltas_frames: Tuple[int, int] = (3, 12)
    cmsd_radius_um: float | None = None


@dataclass(slots=True)
class RunPaths:
    """Return object describing the artefacts produced by :func:`pipeline.run`."""

    out_root: Path
    zarr_paths: dict[int, Path]
    tracks_table: Path | None


__all__ = [
    "GridConfig",
    "WindowConfig",
    "SmoothingConfig",
    "QCConfig",
    "NoiseConfig",
    "MaterialsConfig",
    "RunPaths",
]
