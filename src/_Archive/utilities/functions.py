"""Legacy geometry helpers retained for reference."""
from __future__ import annotations

import math
import numpy as np


def cart_to_sphere(xyz: np.ndarray) -> np.ndarray:
    """Convert cartesian coordinates to spherical coordinates."""
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def sphereFit(spX, spY, spZ):
    """Fit a sphere to the provided coordinates using least squares."""
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residuals, rank, singval = np.linalg.lstsq(A, f, rcond=None)

    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, C
