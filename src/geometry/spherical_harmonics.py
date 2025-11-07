"""Spherical harmonics helpers for embryo surface modeling."""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy_healpix import HEALPix
from scipy.spatial.transform import Rotation as R
from scipy.special import sph_harm
from scipy.spatial import cKDTree


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
    s_arr = cart2sph(vertices_c)
    r, theta, phi = s_arr[:, 0], s_arr[:, 1], s_arr[:, 2]

    L_max = int(np.sqrt(len(coeffs)) - 1)
    basis = build_sh_basis(L_max, phi=phi, theta=theta)
    r_sh = coeffs[None, :] @ basis.T
    x, y, z = sph2cart(r_sh, theta, phi)
    vertices_sh = np.column_stack([x.T, y.T, z.T]) + center
    return vertices_sh, faces, r_sh


def fit_sh_healpix(
    points, center, radius,
    L_max=8, nside=None, ridge=1e-3,
    knn=5, max_ang_deg=30.0, sigma_deg=15.0,
):
    """
    HEALPix binning + spherical KNN fill + real SH fit on deviations (r - radius).
    Returns (coeffs, r_fit_map).
    """
    # ---- spherical coords of points (expects points/center in XYZ, same units) ----
    P = points - center
    r = np.linalg.norm(P, axis=1)
    theta = np.arccos(np.clip(P[:, 2] / np.maximum(r, 1e-12), -1, 1))
    phi   = (np.arctan2(P[:, 1], P[:, 0]) + 2*np.pi) % (2*np.pi)

    # ---- HEALPix binning ----
    if nside is None:
        nside = max(1, int(np.ceil((L_max + 1)/2)))
    hp = HEALPix(nside=nside, order="ring")
    lon = (phi * 180/np.pi) * u.deg
    lat = ((np.pi/2 - theta) * 180/np.pi) * u.deg
    pix = hp.lonlat_to_healpix(lon, lat)

    npix   = hp.npix
    sums   = np.bincount(pix, weights=r, minlength=npix)
    counts = np.bincount(pix, minlength=npix)
    r_map  = np.full(npix, np.nan, dtype=float)
    hit    = counts > 0
    r_map[hit] = sums[hit] / counts[hit]

    # ---- pixel center angles & unit vectors ----
    lon_pix, lat_pix = hp.healpix_to_lonlat(np.arange(npix))
    theta_pix = (90.0 - lat_pix.value) * np.pi/180.0
    phi_pix   = lon_pix.value * np.pi/180.0
    u_pix     = _unit_vec_from_angles(theta_pix, phi_pix)

    # ---- Spherical KNN inpainting for empty pixels (information sharing) ----
    # Build KDTree on occupied pixel unit vectors (chord distance on unit sphere)
    hit_idx = np.where(hit)[0]
    if hit_idx.size == 0:
        raise RuntimeError("No populated HEALPix pixels to fit.")
    tree   = cKDTree(u_pix[hit_idx])
    queryN = min(knn, hit_idx.size)

    empty_idx = np.where(~hit)[0]
    if empty_idx.size:
        # chord distance d relates to angular distance α via: α = 2*arcsin(d/2)
        d, nn = tree.query(u_pix[empty_idx], k=queryN)
        d     = np.atleast_2d(d)
        nn    = np.atleast_2d(nn)
        ang   = 2.0 * np.arcsin(np.clip(d*0.5, 0.0, 1.0))
        ang_deg = ang * 180.0/np.pi

        # weights: Gaussian in angle; mask out neighbors beyond max_ang_deg
        mask_nn = ang_deg <= max_ang_deg
        w = np.exp(-(ang_deg / max_ang_deg if sigma_deg <= 0 else ang_deg / sigma_deg)**2)
        w *= mask_nn

        # gather neighbor radii
        r_nn = r_map[hit_idx][nn]  # (E, queryN)
        # weighted average; if all weights zero, fall back to sphere radius
        num = np.nansum(w * r_nn, axis=1)
        den = np.nansum(w, axis=1)
        fill_vals = np.where(den > 0, num / den, float(radius))
        r_map[empty_idx] = fill_vals

    # Final safety: any remaining NaN → radius
    r_map[np.isnan(r_map)] = float(radius)

    # ---- Build SH design on populated pixels & fit deviations ----
    # (after fill, all pixels are populated)
    Y = build_sh_basis(L_max, theta_pix, phi_pix)
    y_abs = r_map.astype(float)
    y = y_abs - float(radius)  # fit deviations for stability

    # Weight by counts (downweight filled pixels implicitly via small counts=0)
    w = np.clip(counts.astype(float), 1.0, None)  # at least 1 to keep matrix well-defined
    W = np.sqrt(w)[:, None]
    Yw, yw = W * Y, W[:, 0] * y

    # Ridge scale heuristic
    if ridge is None or ridge <= 0:
        ridge = 1e-3 * np.median(np.diag(Yw.T @ Yw))
    A = Yw.T @ Yw + ridge * np.eye(Yw.shape[1])
    b = Yw.T @ yw
    coeffs = np.linalg.solve(A, b)

    # Evaluate on full grid
    r_fit = float(radius) + (Y @ coeffs)
    return coeffs, r_fit


def real_sh(l, m, theta, phi):
    """Real-valued spherical harmonics."""
    Y = sph_harm(abs(m), l, phi, theta)
    if m > 0:
        return np.sqrt(2) * Y.real
    elif m < 0:
        return np.sqrt(2) * (-1)**m * Y.imag
    else:
        return Y.real

def build_sh_basis(L_max, theta, phi):
    """Stack real SH basis up to degree L_max."""
    return np.column_stack([
        real_sh(l, m, theta, phi)
        for l in range(L_max + 1)
        for m in range(-l, l + 1)
    ])

def _unit_vec_from_angles(theta, phi):
    # theta: colat [0,π], phi: lon [0,2π)
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    return np.stack([st*cp, st*sp, ct], axis=1)  # (N,3)
# ---------- Main fit function ----------
