"""Morphological QC helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from skimage.measure import regionprops


def ellipsoid_axis_lengths(central_moments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute semi-axis lengths of ellipsoids defined by central moments."""
    if  central_moments.shape[1:] == (4, 4, 4):
        central_moments = central_moments[:, :3, :3, :3]
    if central_moments.ndim != 4 or central_moments.shape[1:] != (3, 3, 3):
        raise ValueError("central_moments must have shape (N, 3, 3, 3)")
        # return np.array([np.inf, 0, 0]), np.array([])

    m0 = central_moments[:, 0, 0, 0]
    sxx = central_moments[:, 2, 0, 0] / m0
    syy = central_moments[:, 0, 2, 0] / m0
    szz = central_moments[:, 0, 0, 2] / m0
    sxy = central_moments[:, 1, 1, 0] / m0
    sxz = central_moments[:, 1, 0, 1] / m0
    syz = central_moments[:, 0, 1, 1] / m0

    cov = np.stack(
        [
            np.stack([sxx, sxy, sxz], axis=1),
            np.stack([sxy, syy, syz], axis=1),
            np.stack([sxz, syz, szz], axis=1),
        ],
        axis=1,
    )
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals, axis=1)[:, ::-1]
    radii = np.sqrt(5 * eigvals)
    return radii, cov


def filter_by_eccentricity(
    mask: np.ndarray,
    scale_vec: Iterable[float],
    max_eccentricity: float,
    min_minor_radius: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove labels whose ellipsoid fit exceeds ``max_eccentricity``."""
    props = regionprops(mask, spacing=tuple(scale_vec))
    if not props:
        return mask, np.array([], dtype=np.int32)

    moments = np.stack([p.moments_central for p in props], axis=0)
    radii, _ = ellipsoid_axis_lengths(moments)
    ecc = radii[:, 0] / (radii[:, 1] + 1e-4)
    valid = (ecc <= max_eccentricity) & (radii[:, 1] >= min_minor_radius)

    keep_labels = np.array([p.label for i, p in enumerate(props) if valid[i]], dtype=np.int32)
    filtered = np.where(np.isin(mask, keep_labels), mask, 0)
    return filtered.astype(mask.dtype, copy=False), keep_labels
