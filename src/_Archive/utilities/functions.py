"""Deprecated helper routines preserved for legacy workflows."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def cart_to_sphere(xyz: np.ndarray) -> np.ndarray:
    """Convert cartesian coordinates ``xyz`` (N×3) to spherical coordinates."""

    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # elevation from +Z axis
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def sphereFit(spX: Iterable[float], spY: Iterable[float], spZ: Iterable[float]):
    """Least-squares fit of a sphere to input points."""

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
    C, *_ = np.linalg.lstsq(A, f, rcond=None)

    radius = math.sqrt((C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3])
    return radius, C
