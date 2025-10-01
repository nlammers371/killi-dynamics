import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import healpy as hp

import numpy as np
import healpy as hp


def laea_forward(theta, phi):
    """Lambert azimuthal equal-area, centered at north pole.
    theta, phi in radians; returns x, y in [-1,1] disk."""
    r = np.sqrt(2.0) * np.sin(theta / 2.0)
    return r * np.cos(phi), r * np.sin(phi)

def laea_inverse(x, y):
    """Inverse LAEA back to (theta, phi)."""
    r = np.hypot(x, y)
    r = np.clip(r, 0.0, 1.0)  # stay within hemisphere
    theta = 2.0 * np.arcsin(r / np.sqrt(2.0))
    phi = np.arctan2(y, x)
    return theta, phi

def recenter_longitude(phi, phi0):
    """Rotate longitudes by phi0 (radians)."""
    return (phi - phi0 + np.pi) % (2*np.pi) - np.pi

def healpix_to_mesh(values, radius, center=(0,0,0)):
    """
    Convert a HEALPix map to a triangular mesh suitable for napari.add_surface.

    Parameters
    ----------
    values : (Npix,) array
        Values per HEALPix pixel.
    radius : float
        Sphere radius (physical units).
    center : (3,) array
        Sphere center (z,y,x).

    Returns
    -------
    verts : (N,3) array
        Vertex coordinates (z,y,x).
    faces : (M,3) array
        Mesh faces (triangle indices).
    vvals : (N,) array
        Vertex values (duplicated from pixel values).
    """
    npix = len(values)
    nside = int(np.sqrt(npix // 12))
    if 12 * nside**2 != npix:
        raise ValueError(f"Array length {npix} is not valid for a HEALPix map.")

    # get pixel boundaries on the unit sphere: shape (3, nverts, npix)
    boundaries = hp.boundaries(nside, np.arange(npix), step=1, nest=False)

    verts = []
    faces = []
    vvals = []
    vert_index = {}
    idx = 0

    for pix in range(npix):
        # slice out one pixel -> shape (3,nverts)
        corners = boundaries[pix].T        # (nverts,3), in (x,y,z)
        corners = corners[:, ::-1]               # flip -> (z,y,x)
        corners = corners * radius + np.array(center)[None, :]

        quad = []
        for corner in corners:
            key = tuple(np.round(corner, 8))
            if key not in vert_index:
                vert_index[key] = idx
                verts.append(corner)
                vvals.append(values[pix])
                idx += 1
            quad.append(vert_index[key])

        # triangulate quad
        faces.append([quad[0], quad[1], quad[2]])
        faces.append([quad[0], quad[2], quad[3]])

    return np.array(verts, float), np.array(faces, int), np.array(vvals, float)


def project_to_healpix(vol, center, radius, scale_vec,
                       nside=64, mode="mean", dist_thresh=50.0):
    dz, dy, dx = scale_vec
    Z, Y, X = np.indices(vol.shape)
    coords = np.c_[Z.ravel()*dz, Y.ravel()*dy, X.ravel()*dx]
    vals = vol.ravel().astype(float)

    # restrict to spherical shell
    dR = np.linalg.norm(coords - center[None, :], axis=1) - radius
    mask = np.abs(dR) <= dist_thresh
    coords, vals = coords[mask], vals[mask]

    # spherical angles
    rel = coords - center[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arccos(np.clip(rel[:, 0] / r, -1, 1))
    phi = np.arctan2(rel[:, 1], rel[:, 2]) % (2*np.pi)

    # map to healpix pixels
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, theta, phi)

    # counts per pixel
    counts = np.bincount(pix, minlength=npix)

    if mode == "sum":
        values = np.bincount(pix, weights=vals, minlength=npix)
    elif mode == "mean":
        sums = np.bincount(pix, weights=vals, minlength=npix)
        values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    elif mode == "max":
        # np.bincount can't do max; need groupby-style reduction
        values = np.full(npix, -np.inf)
        np.maximum.at(values, pix, vals)
        values[values == -np.inf] = 0.0
    else:
        raise ValueError(f"Unknown mode {mode}")

    return values, counts



def smooth_spherical_grid(grid, sigma_theta=1.0, sigma_phi=1.0, counts=None):
    """
    Smooth a (theta, phi) grid with seam- and pole-aware Gaussian filtering.

    Parameters
    ----------
    grid : (n_theta, n_phi) array
        Values on spherical grid (theta rows, phi cols).
    sigma_theta, sigma_phi : float
        Gaussian std dev along theta and phi, in grid bins.
    counts : (n_theta, n_phi) array or None
        Optional counts array for weighted smoothing (e.g. voxel hits).

    Returns
    -------
    smoothed : (n_theta, n_phi) array
        Smoothed grid with seam (phi wrap) and pole handling.
    """
    g = np.asarray(grid, float).copy()
    n_theta, n_phi = g.shape

    # --- Pole fix before smoothing: collapse values across φ ---
    for pole in [0, n_theta-1]:
        row = g[pole, :]
        if np.any(np.isfinite(row)):
            g[pole, :] = np.nanmean(row)
        else:
            neighbor = 1 if pole == 0 else n_theta-2
            g[pole, :] = np.nanmean(g[neighbor, :])

    if counts is None:
        smoothed = gaussian_filter(g, sigma=(sigma_theta, sigma_phi),
                                   mode=('reflect', 'wrap'))
    else:
        c = np.asarray(counts, float).clip(min=0)
        # weight by sin(theta) to respect patch area
        theta = np.linspace(0, np.pi, n_theta)
        area_w = np.sin(theta)[:, None]
        w = c * area_w
        num = gaussian_filter(g * w, sigma=(sigma_theta, sigma_phi),
                              mode=('reflect', 'wrap'))
        den = gaussian_filter(w, sigma=(sigma_theta, sigma_phi),
                              mode=('reflect', 'wrap'))
        smoothed = np.divide(num, den,
                             out=np.zeros_like(num),
                             where=den > 1e-12)

    # --- Pole fix after smoothing: enforce uniform values across φ ---
    smoothed[0, :] = smoothed[0, :].mean()
    smoothed[-1, :] = smoothed[-1, :].mean()

    return smoothed

def project_to_sphere(vol, center, radius, scale_vec,
                           n_theta=180, n_phi=360,
                           mode="mean", dist_thresh=50.0):
    """
    Project intensities onto a spherical mesh suitable for napari.add_surface,
    with pole/seam handling.

    Parameters
    ----------
    vol : ndarray
        3D image (z,y,x).
    center : (3,) array
        Sphere center (z,y,x) in physical units.
    radius : float
        Sphere radius in physical units.
    scale_vec : (3,) array
        Voxel size (dz, dy, dx) in physical units.
    n_theta, n_phi : int
        Angular resolution of spherical mesh.
    mode : str
        Projection mode: "mean", "sum", "max".
    dist_thresh : float
        Maximum distance (in physical units) from surface to include voxels.

    Returns
    -------
    verts : (N,3) array
        Vertex coordinates (z,y,x) in physical units.
    faces : (M,3) array
        Mesh faces as indices into verts.
    values : (N,) array
        Intensity values per vertex.
    """
    dz, dy, dx = scale_vec
    Z, Y, X = np.indices(vol.shape)
    coords = np.c_[Z.ravel()*dz, Y.ravel()*dy, X.ravel()*dx]
    vals = vol.ravel().astype(float)

    # Restrict to shell band
    dR = np.linalg.norm(coords - center[None, :], axis=1) - radius
    mask = np.abs(dR) <= dist_thresh
    coords, vals = coords[mask], vals[mask]

    # Convert to spherical coords relative to fitted center
    rel = coords - center[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arccos(np.clip(rel[:, 0] / r, -1, 1))   # polar (z)
    phi = np.arctan2(rel[:, 1], rel[:, 2])             # azimuth (y vs x)
    phi = (phi + 2*np.pi) % (2*np.pi)

    # Grid definition
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    # Assign voxels to nearest grid bin
    ti = np.clip(np.round(theta / np.pi * (n_theta - 1)).astype(int), 0, n_theta-1)
    pi = np.round(phi / (2*np.pi) * n_phi).astype(int) % n_phi
    vi = ti * n_phi + pi

    values = np.zeros(n_theta * n_phi, float)
    counts = np.zeros(n_theta * n_phi, int)

    for idx, v in zip(vi, vals):
        if mode == "sum":
            values[idx] += v
        elif mode == "max":
            values[idx] = max(values[idx], v)
        else:  # mean
            values[idx] += v
            counts[idx] += 1

    if mode == "mean":
        values = np.divide(values, counts,
                           out=np.zeros_like(values),
                           where=counts > 0)

    # --- Fix poles: unify values across all φ bins ---
    for pole in [0, n_theta-1]:
        idxs = np.arange(pole * n_phi, (pole+1) * n_phi)
        if counts[idxs].sum() > 0:
            m = values[idxs][counts[idxs] > 0].mean()
        else:
            m = 0.0
        values[idxs] = m

    # --- Build mesh ---
    Theta, Phi = np.meshgrid(thetas, phis, indexing="ij")
    Xs = radius * np.sin(Theta) * np.cos(Phi) + center[2]
    Ys = radius * np.sin(Theta) * np.sin(Phi) + center[1]
    Zs = radius * np.cos(Theta) + center[0]
    verts = np.stack([Zs.ravel(), Ys.ravel(), Xs.ravel()], axis=1)

    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            p0 = i * n_phi + j
            p1 = i * n_phi + (j+1) % n_phi   # wrap around φ
            p2 = (i+1) * n_phi + j
            p3 = (i+1) * n_phi + (j+1) % n_phi
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])
    faces = np.array(faces, dtype=np.int32)

    return verts, faces, values, mask


def remove_background_dog(vol, sigma_small_um=2.0, sigma_large_um=8.0, scale_vec=None):

    if scale_vec is None:
        scale_vec = np.array([1.0, 1.0, 1.0])

    scale_vec = np.asarray(scale_vec, dtype=float)

    # Convert physical sigmas into voxel units by dividing by voxel spacing
    sigma_small = tuple(sigma_small_um / scale_vec)
    sigma_large = tuple(sigma_large_um / scale_vec)

    vol = vol.astype(np.float32)
    blur_small = gaussian_filter(vol, sigma_small)
    blur_large = gaussian_filter(vol, sigma_large)

    dog = blur_small - blur_large
    dog = np.clip(dog, 0, None)  # keep positive contrast only
    return dog

def fit_sphere(points_phys, im_shape, R0=None, weights=None, fit_radius=False, loss="huber"):
    """
    Fit a sphere center (and optionally radius) to 3D points.

    Parameters
    ----------
    points_phys : (N,3) array
        Candidate shell points in physical units (z,y,x).
    R0 : float or None
        Known/prior radius. Required if fit_radius=False.
        Used as an initial guess if fit_radius=True.
    weights : (N,) array or None
        Optional nonnegative weights (e.g. DoG intensities).
    fit_radius : bool
        If False: fix radius to R0 and fit center only.
        If True: fit both center and radius.
    loss : str
        Robust loss for least_squares ("linear", "huber", "soft_l1", ...).

    Returns
    -------
    c_fit : (3,) array
        Estimated center in physical units.
    R_fit : float
        Estimated (or fixed) radius in physical units.
    """
    pts = np.asarray(points_phys, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_phys must be (N,3)")
    N = len(pts)

    if weights is None:
        w = np.ones(N)
    else:
        w = np.asarray(weights, dtype=float).clip(min=0)

    # --- residual functions ---
    if not fit_radius:
        if R0 is None:
            raise ValueError("R0 must be provided if fit_radius=False")

        def residuals(c):
            d = np.linalg.norm(pts - c[None, :], axis=1)
            return (d - R0) * np.sqrt(w)

        c0 = [im_shape[0]+R0/2, im_shape[1] / 2,  im_shape[2] / 2]
        c0[0] = im_shape[0]
        lb = [0, (im_shape[1] // 3), (im_shape[2] // 3)]  # , (im_shape[2]] // 3)]
        ub = [im_shape[0]+R0, (2 * im_shape[1] // 3), (2 * im_shape[2] // 3)]

        res = least_squares(residuals, c0, loss=loss, bounds=(lb, ub))
        return res.x, R0

    else:
        if R0 is None:
            # crude guess: mean distance to centroid
            c0 = np.mean(pts, axis=0)
            R0 = np.mean(np.linalg.norm(pts - c0[None, :], axis=1))
        else:
            c0 = np.mean(pts, axis=0)

        def residuals(p):
            c, R = p[:3], p[3]
            d = np.linalg.norm(pts - c[None, :], axis=1)
            return (d - R) * np.sqrt(w)

        lb = [0, (im_shape[1] // 3), (im_shape[2] // 3), 0]#, (im_shape[2]] // 3)]
        ub = [1000, (2 * im_shape[1] // 3), (2 * im_shape[2] // 3), 1000]

        res = least_squares(residuals, np.hstack([c0, R0]), loss=loss, bounds=(lb, ub))
        return res.x[:3], res.x[3]


def make_sphere_mesh(n_phi, n_theta, center, radius):
    """
    Create a spherical mesh (vertices, faces) suitable for napari.add_surface.

    Parameters
    ----------
    n_phi, n_theta : int
        Angular resolution of spherical mesh.
    radius : float
        Sphere radius.

    Returns
    -------
    verts : (N,3) array
        Vertex coordinates (z,y,x).
    faces : (M,3) array
        Mesh faces as indices into verts.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    Theta, Phi = np.meshgrid(thetas, phis, indexing="ij")
    Xs = radius * np.sin(Theta) * np.cos(Phi) + center[2]
    Ys = radius * np.sin(Theta) * np.sin(Phi) + center[1]
    Zs = radius * np.cos(Theta) + center[0]
    verts = np.stack([Zs.ravel(), Ys.ravel(), Xs.ravel()], axis=1)

    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            p0 = i * n_phi + j
            p1 = i * n_phi + (j+1) % n_phi   # wrap around φ
            p2 = (i+1) * n_phi + j
            p3 = (i+1) * n_phi + (j+1) % n_phi
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])
    faces = np.array(faces, dtype=np.int32)

    return verts, faces