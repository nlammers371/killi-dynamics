"""Spherical harmonics helpers for embryo surface modeling."""
from __future__ import annotations

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R
from scipy.special import sph_harm

from .sphere import create_sphere_mesh, fit_sphere


def build_sh_basis(L_max: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return spherical harmonics basis functions up to order ``L_max``."""
    basis = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            basis.append(sph_harm(m, l, phi, theta).real)
    return np.column_stack(basis).T


def cart2sph(points_xyz: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates relative to ``v``."""
    if v is None:
        v = np.array([0, 0, 1], float)
    else:
        v = np.asarray(v, float)

    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("Up-vector must be nonzero")
    v_unit = v / norm_v
    target = np.array([0, 0, 1], float)
    rot, _ = R.align_vectors([v_unit], [target])
    pts_rot = points_xyz @ rot.as_matrix().T

    x, y, z = pts_rot.T
    r = np.linalg.norm(pts_rot, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return np.column_stack([r, theta, phi])


def sph2cart(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def create_sh_mesh(coeffs: np.ndarray, sphere_mesh: tuple[np.ndarray, np.ndarray]):
    """Evaluate spherical harmonics on a sphere mesh."""
    vertices, faces = sphere_mesh
    center = vertices.mean(axis=0)
    vertices_c = vertices - center
    _, theta, phi = cart2sph(vertices_c)

    L_max = int(np.sqrt(len(coeffs)) - 1)
    basis = build_sh_basis(L_max, phi=phi, theta=theta)
    r_sh = coeffs[None, :] @ basis
    x, y, z = sph2cart(r_sh, theta, phi)
    vertices_sh = np.column_stack([x.T, y.T, z.T]) + center
    return (vertices_sh, faces), r_sh


def fit_sphere_and_sh(
    points: np.ndarray,
    L_max: int = 10,
    knn: int = 3,
    k_thresh: float = 50.0,
    sphere_quantile: float = 0.25,
):
    """Fit a sphere and spherical harmonics deviations to a point cloud."""
    center, radius, inner_radius, outer_radius = fit_sphere(points, quantile=sphere_quantile)
    scale_factor = inner_radius / radius
    points_c = points - center

    r, theta, phi = cart2sph(points_c)
    vertices, faces = create_sphere_mesh(np.asarray([0, 0, 0]), radius, resolution=100)
    r_v, theta_v, phi_v = cart2sph(vertices)

    surf_dist = distance_matrix(vertices, points_c)
    closest_idx = np.argsort(surf_dist, axis=1)[:, :knn]
    closest_dist = np.sort(surf_dist, axis=1)[:, :knn]
    r_samples = r[closest_idx]
    r_samples[closest_dist > k_thresh] = np.nan
    radial = np.nanmean(r_samples, axis=1)
    radial[np.isnan(radial)] = radius
    radial *= scale_factor

    basis = build_sh_basis(L_max, phi=phi_v, theta=theta_v)
    Y = np.column_stack(basis)
    coeffs, *_ = np.linalg.lstsq(Y, radial, rcond=None)
    return np.array(coeffs), center, inner_radius, radial
