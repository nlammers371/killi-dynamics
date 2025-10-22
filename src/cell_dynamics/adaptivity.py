"""Scoping utilities for future adaptive grid support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class AdaptiveGridDesign:
    """Prototype container describing an adaptive grid."""

    centers: np.ndarray
    weights: np.ndarray


def propose_adaptive_centers(density_map: np.ndarray, target_steps: int) -> AdaptiveGridDesign:
    """Produce a toy set of candidate adaptive grid centres.

    The current implementation selects the top ``target_steps`` locations from
    the density map and normalises them into weights. It is intended purely as a
    placeholder for the more sophisticated strategies described in the brief.
    """

    flat = density_map.ravel()
    if flat.size == 0:
        return AdaptiveGridDesign(centers=np.zeros((0, 3)), weights=np.zeros(0))

    k = min(target_steps, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    values = flat[idx]
    weights = values / values.sum() if values.sum() > 0 else np.ones_like(values) / len(values)
    centers = np.column_stack(np.unravel_index(idx, density_map.shape))
    return AdaptiveGridDesign(centers=centers.astype(float), weights=weights.astype(float))


__all__ = ["AdaptiveGridDesign", "propose_adaptive_centers"]
