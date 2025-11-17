import numpy as np
from astropy_healpix import HEALPix
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import BallTree
import astropy.units as u


def laplacian_smooth(field, tau=0.2, steps=2):
    npix = field.size
    nside = int(np.sqrt(npix / 12))
    hp = HEALPix(nside, order="nested")
    rows, cols, data = [], [], []
    for p in range(npix):
        nbrs = hp.neighbours(p)
        nbrs = nbrs[nbrs >= 0]
        deg = len(nbrs)
        rows.extend([p] * deg + [p])
        cols.extend(nbrs.tolist() + [p])
        data.extend([-1.0] * deg + [deg])
    L = csr_matrix((data, (rows, cols)), shape=(npix, npix))
    A = eye(npix, format="csr") + tau * L
    f = field.astype(float)
    for _ in range(steps):
        f = spsolve(A, f)
    return f


def smooth_field_on_sphere_knn(field, sigma_deg=10.0, k=25, max_mult=3.0):
    """
    Smooth a HEALPix field using Gaussian weights over k nearest neighbors.
    Vectorized, no explicit Python loop over pixels.
    """
    npix = field.size
    nside = int(np.sqrt(npix / 12))
    hp = HEALPix(nside, order="nested")
    lon, lat = hp.healpix_to_lonlat(np.arange(npix))
    lon = lon.to_value(u.rad)
    lat = lat.to_value(u.rad)

    coords = np.column_stack((lat, lon))
    tree = BallTree(coords, metric="haversine")

    # query k nearest neighbors (returns distances in radians)
    dists, idxs = tree.query(coords, k=k)

    sigma_rad = np.deg2rad(sigma_deg)
    mask = dists <= max_mult * sigma_rad  # (npix, k)

    # Gaussian weights
    w = np.exp(-(dists ** 2) / (2 * sigma_rad**2))
    w *= mask

    # Normalize weights so each row sums to 1
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    w /= w_sum

    # Gather neighbor values and weight them
    v = field[idxs]  # (npix, k)
    smoothed = np.sum(w * v, axis=1)

    return smoothed


def smooth_on_sphere(field: np.ndarray, sigma_deg: float = 10.0, k: int = 25) -> np.ndarray:
    """Compatibility wrapper used by the analysis pipeline."""

    return smooth_field_on_sphere_knn(field, sigma_deg=sigma_deg, k=k)


__all__ = [
    "laplacian_smooth",
    "smooth_field_on_sphere_knn",
    "smooth_on_sphere",
]
