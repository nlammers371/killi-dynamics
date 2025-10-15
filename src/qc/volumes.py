"""Volume-based QC utilities for labeled masks."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def compute_label_volumes(mask: np.ndarray, scale_vec: Iterable[float]) -> np.ndarray:
    """Return physical volumes for each label in ``mask``.

    Parameters
    ----------
    mask:
        Integer-labeled array with background encoded as ``0``.
    scale_vec:
        Iterable of voxel spacings in ``(z, y, x)`` order.

    Returns
    -------
    np.ndarray
        Array of physical volumes whose ``i``\ th entry corresponds to
        label ``i``. The first entry encodes the background volume and can
        be ignored by callers.
    """
    vox_vol = float(np.prod(tuple(scale_vec)))
    counts = np.bincount(mask.ravel())
    return counts * vox_vol


def filter_by_minimum_volume(
    mask: np.ndarray,
    scale_vec: Iterable[float],
    min_volume: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out labels whose physical volume falls below ``min_volume``."""
    volumes = compute_label_volumes(mask, scale_vec)
    labels = np.arange(len(volumes))
    keep = labels[volumes > min_volume]
    keep = keep[keep > 0]

    filtered = np.where(np.isin(mask, keep), mask, 0)
    return filtered.astype(mask.dtype, copy=False), keep
