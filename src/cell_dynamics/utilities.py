from pathlib import Path
import pandas as pd
import numpy as np
import zarr
from astropy_healpix import HEALPix
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import BallTree
import astropy.units as u

def load_tracking_data(root: str,
                       project_name: str,
                       track_config_name: str,
                       track_instance_str: str | None = None):
    root = Path(root)
    track_dir = root / "tracking" / project_name / track_config_name
    if track_instance_str is None:
        instance_dirs = sorted(track_dir.glob("track_*"), key=lambda p: p.stat().st_mtime)
        if len(instance_dirs) == 0:
            raise ValueError(f"No tracking instance directories found in {track_dir}")
        elif len(instance_dirs) > 1:
            print(f"Multiple tracking instances found. Using latest: {instance_dirs[-1].name}")
        track_instance_str = instance_dirs[-1].name
    track_dir = track_dir / track_instance_str

    track_path = track_dir / "tracks_fluo.csv"
    if not track_path.exists():
        track_path = track_dir / "tracks.csv"
    tracks_df = pd.read_csv(track_path)  # tracking
    sphere_df = pd.read_csv(track_dir / "sphere_fit.csv")  # sphere fits
    class_path = track_dir / "track_class_df.csv"
    if class_path.exists():
        class_df = pd.read_csv(class_path)  # track classifications
    else:
        class_df = None

    # we have to rescale x,y,z to microns...
    im_zarr = zarr.open(root / "built_data" / "zarr_image_files" / f"{project_name}_fused.zarr", mode="r")
    scale_vec = np.array(im_zarr.attrs["voxel_Size_um"])
    tracks_df[["z", "y", "x"]] = np.multiply(tracks_df[["z", "y", "x"]].to_numpy(), scale_vec[None, :])

    return tracks_df, sphere_df, class_df


def laplacian_smooth(field, tau=0.2, steps=2):

    npix = field.size
    nside = int(np.sqrt(npix / 12))
    hp = HEALPix(nside, order="nested")
    rows, cols, data = [], [], []
    for p in range(npix):
        nbrs = hp.neighbours(p)
        nbrs = nbrs[nbrs >= 0]
        deg = len(nbrs)
        rows.extend([p]*deg + [p])
        cols.extend(nbrs.tolist() + [p])
        data.extend([-1.0]*deg + [deg])
    L = csr_matrix((data, (rows, cols)), shape=(npix, npix))
    A = eye(npix, format='csr') + tau * L
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
    tree = BallTree(coords, metric='haversine')

    # query k nearest neighbors (returns distances in radians)
    dists, idxs = tree.query(coords, k=k)

    sigma_rad = np.deg2rad(sigma_deg)
    mask = dists <= max_mult * sigma_rad  # (npix, k)

    # Gaussian weights
    w = np.exp(-(dists**2) / (2 * sigma_rad**2))
    w *= mask

    # Normalize weights so each row sums to 1
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    w /= w_sum

    # Gather neighbor values and weight them
    v = field[idxs]  # (npix, k)
    smoothed = np.sum(w * v, axis=1)

    return smoothed