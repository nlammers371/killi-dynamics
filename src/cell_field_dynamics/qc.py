"""Quality-control helpers for the cell-dynamics pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from src.cell_field_dynamics.dev.config import QCConfig
from src.cell_field_dynamics.dev.grids import GridBinResult
from src.cell_field_dynamics.dev.metrics import MetricCollection
from src.cell_field_dynamics.dev.msd import MSDResult
from src.cell_field_dynamics.dev.materials import MaterialMetrics
from .vector_field import VectorFieldResult


@dataclass(slots=True)
class QCResult:
    """Boolean validity masks and counts per metric."""

    nside: int
    time_centers: np.ndarray
    masks: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]


def apply_quality_control(
    binned: dict[int, GridBinResult],
    metric_results: dict[int, MetricCollection],
    vector_results: dict[int, VectorFieldResult],
    msd_results: dict[int, MSDResult],
    material_results: dict[int, MaterialMetrics],
    flux_results: dict[int, dict[str, np.ndarray]],
    qc_cfg: QCConfig,
) -> dict[int, QCResult]:
    """Derive simple QC masks based on observation counts."""

    qc_data: dict[int, QCResult] = {}

    for nside, grid_result in binned.items():
        counts = grid_result.counts.astype(np.int32)
        nt, npix = counts.shape
        masks: dict[str, np.ndarray] = {}
        per_metric_counts: dict[str, np.ndarray] = {}

        masks["drift"] = counts >= qc_cfg.min_steps_drift
        per_metric_counts["drift"] = counts

        masks["derivatives"] = counts >= qc_cfg.min_steps_derivatives
        per_metric_counts["derivatives"] = counts

        masks["msd"] = counts >= qc_cfg.min_pairs_msd
        per_metric_counts["msd"] = counts

        masks["materials"] = counts >= qc_cfg.min_cells_region
        per_metric_counts["materials"] = counts

        qc_data[nside] = QCResult(
            nside=nside,
            time_centers=grid_result.time_centers,
            masks=masks,
            counts=per_metric_counts,
        )

    return qc_data


__all__ = ["QCResult", "apply_quality_control"]
