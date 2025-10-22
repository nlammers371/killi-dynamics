"""Flux metrics derived from the coarse vector field."""
from __future__ import annotations

import numpy as np

from .grids import healpix_nside2npix
from .vector_field import VectorFieldResult


def compute_flux_from_vector_field(
    vector_results: dict[int, VectorFieldResult],
    *,
    radius: float = 1.0,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute net flux and throughput for each grid from the drift field."""

    flux_data: dict[int, dict[str, np.ndarray]] = {}
    radius = max(float(radius), 1e-6)

    for nside, result in vector_results.items():
        if result.drift.size == 0:
            flux_data[nside] = {"net": np.empty((0, 0), dtype=np.float32), "throughput": np.empty((0, 0), dtype=np.float32)}
            continue
        pixel_area = 4.0 * np.pi * radius**2 / healpix_nside2npix(nside)
        drift_speed = np.linalg.norm(result.drift, axis=-1)
        net_flux = result.divergence * pixel_area
        throughput = drift_speed * np.sqrt(pixel_area)
        flux_data[nside] = {
            "net": net_flux.astype(np.float32),
            "throughput": throughput.astype(np.float32),
        }
    return flux_data


__all__ = ["compute_flux_from_vector_field"]
