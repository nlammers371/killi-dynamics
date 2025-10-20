import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import zarr
from astropy_healpix import HEALPix
import matplotlib
# matplotlib.use("TkAgg")
import astropy.units as u
from tqdm import tqdm
from pathlib import Path
from plotting import plot_density_mollweide
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve


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

def add_speed(df, tangential=False):
    """
    Adds vx, vy, vz, and speed assuming unit time step between frames.
    NaNs from the first frame of each track are filled with the nearest
    non-NaN value (forward/backward fill).
    """
    df = df.sort_values(["track_id", "t"]).copy()

    # per-track finite differences (unit Δt)
    vcols = ["vx", "vy", "vz"]
    df[vcols] = (
        df.groupby("track_id")[["x", "y", "z"]]
          .diff()
          .groupby(df["track_id"])
          .apply(lambda g: g.fillna(method="bfill").fillna(method="ffill"))
          .reset_index(level=0, drop=True)
    )

    if tangential:
        rel = df[["x", "y", "z"]].to_numpy() - df[["xs", "ys", "zs"]].to_numpy()
        rel_unit = rel / np.linalg.norm(rel, axis=1, keepdims=True)
        v = df[vcols].to_numpy()
        v_tan = v - (v * rel_unit).sum(1, keepdims=True) * rel_unit
        df[vcols] = v_tan

    df["speed"] = np.linalg.norm(df[vcols].to_numpy(), axis=1)
    return df




def add_local_degree(projected_df: pd.DataFrame,
                     r_cut_um: float = 30.0) -> pd.DataFrame:
    """
    Adds a 'degree' column (neighbor count within r_cut_um on the sphere).
    Operates directly on spherical coordinates; no gridding.
    """
    # approximate global radius (µm) for angular cutoff
    R_um = projected_df["r"].median()
    rho = r_cut_um / R_um  # radians

    # store results here
    deg_all = np.zeros(len(projected_df), dtype=np.float32)

    for t, sub in projected_df.groupby("t", sort=True):
        if len(sub) < 2:
            continue

        # coords for haversine metric: [lat, lon] in radians
        lat = (np.pi / 2 - sub["theta"].to_numpy())
        lon = sub["phi"].to_numpy()
        coords = np.column_stack((lat, lon))

        # build BallTree with haversine metric
        tree = BallTree(coords, metric="haversine")

        # fixed k query for speed; mask by radius
        k = min(64, max(8, int(np.sqrt(len(sub)))))
        dists, idxs = tree.query(coords, k=k)
        mask = dists <= rho
        # degree = count of neighbors (exclude self)
        deg = (mask.sum(axis=1) - 1).astype(np.float32)

        # write back into full array
        deg_all[sub.index] = deg

    projected_df = projected_df.copy()
    projected_df["degree"] = deg_all
    return projected_df


def create_field_dset(store, name, shape, nside, cname="zstd", clevel=3, dtype="float32"):
    if name in store:
        del store[name]
    return store.create_dataset(
        name,
        shape=shape,
        chunks=(1, 12 * nside**2),
        dtype=dtype,
        compressor=zarr.Blosc(cname=cname, clevel=clevel, shuffle=2),
    )

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
    scale_vec = np.array([im_zarr.attrs["PhysicalSizeZ"],
                          im_zarr.attrs["PhysicalSizeY"],
                          im_zarr.attrs["PhysicalSizeX"]])
    tracks_df[["z", "y", "x"]] = np.multiply(tracks_df[["z", "y", "x"]].to_numpy(), scale_vec[None, :])

    return tracks_df, sphere_df, class_df


# ==============================================================
# (1) Projection and QC
# ==============================================================

def project_tracks_to_sphere(tracks_df, sphere_df, outlier_thresh=3.0):
    """
    Project cell coordinates onto the fitted sphere at matching timepoints.
    """
    # join on exact time
    df = tracks_df.merge(sphere_df, on="t", how="inner",
                         suffixes=("", "_sphere"))

    # vectorized projection
    coords  = df[["x", "y", "z"]].to_numpy()
    centers = df[["xs", "ys", "zs"]].to_numpy()
    rel     = coords - centers
    r_obs   = np.linalg.norm(rel, axis=1)
    rel_unit = rel / r_obs[:, None]

    # analytic projection to sphere surface
    proj_xyz = centers + df["r"].to_numpy()[:, None] * rel_unit

    # optional QC
    r_dev   = r_obs - df["r"].to_numpy()
    outlier = np.abs(r_dev) > outlier_thresh * np.std(r_dev)

    # spherical coordinates (colatitude, longitude)
    x, y, z = (proj_xyz - centers).T
    theta = np.arccos(np.clip(z / df["r"].to_numpy(), -1, 1))
    phi   = np.arctan2(y, x)

    df["theta"]   = theta
    df["phi"]     = phi
    df["r_obs"]   = r_obs
    df["r_dev"]   = r_dev
    df["outlier"] = outlier

    return df


# ==============================================================
# (2) Wrapper to compute per-time Healpix maps
# ==============================================================

def analyze_on_sphere(root: str | Path,
                      project_name: str,
                      track_config_name: str,
                      track_instance_str: str | None = None,
                      nside: int = 64,
                      dT: float = 90,
                      smooth_sigma_t: float | None = None):
    """
    Project tracks, compute per-pixel quantities, and save to Zarr store.

    Computes:
      - density map (counts per pixel)
      - mean of each quantity_col per pixel

    Parameters
    ----------
    tracks_df : DataFrame with track coordinates
    sphere_df : DataFrame with sphere fits
    zarr_path : path to output zarr store
    quantity_cols : list of scalar quantities to average
    nside : Healpix resolution
    smooth_sigma_t : optional temporal smoothing for sphere fit
    """

    # Load in data
    track_df, sphere_df, class_df = load_tracking_data(
                                                root=root,
                                                project_name=project_name,
                                                track_config_name=track_config_name,
                                                track_instance_str=track_instance_str,
                                            )


    hp = HEALPix(nside, order="nested")

    projected_df = project_tracks_to_sphere(track_df, sphere_df)

    if class_df is not None:
        projected_df = projected_df.merge(class_df[["track_id", "track_class"]], on="track_id", how="left")
        # filter for deep cells
        projected_df = projected_df[projected_df["track_class"] == 0].reset_index(drop=True)

    # Unique timepoints
    time_points = np.unique(projected_df["t"].to_numpy())
    n_t = len(time_points)
    n_pix = hp.npix
    # Output zarr for projected fields

    zarr_path = Path(root) / "analysis" / "cell_fields" / project_name / "cell_fields_sphere.zarr"
    field_store = zarr.open(zarr_path, mode="a")

    field_store.attrs["project_name"] = project_name
    field_store.attrs["tracking_config"] = track_config_name
    field_store.attrs["nside"] = nside
    field_store.attrs["description"] = "Per-time cell field maps on spherical surface"

    # calculate speed and network info
    print("Calculating surface speed...")
    projected_df = add_speed(projected_df)
    print("Calculating NN network degree...")
    projected_df = add_local_degree(projected_df)

    #####
    # Initialize fields
    # density
    d_arr = create_field_dset(field_store, "density", (n_t, n_pix), nside)
    # speed
    s_arr = create_field_dset(field_store, "speed", (n_t, n_pix), nside)
    # k arr
    k_arr = create_field_dset(field_store, "degree", (n_t, n_pix), nside)

    rad = sphere_df[["r"]].values.mean()
    A = 4 * np.pi * rad**2
    a = A / n_pix

    for ti, t in enumerate(tqdm(time_points, desc="Computing maps")):
        sub = projected_df[(projected_df["t"] == t) & (~projected_df["outlier"])]

        if sub.empty:
            continue

        # Convert to the longitude/latitude convention Astropy expects:
        lon = sub["phi"].to_numpy() * u.rad
        lat = (np.pi / 2 - sub["theta"].to_numpy()) * u.rad  # colatitude → latitude

        pix = hp.lonlat_to_healpix(lon, lat)

        # Density
        counts = np.bincount(pix, minlength=n_pix)
        d_arr[ti, :] = counts.astype(np.float32) / a

        # speed, degree → median per pixel
        for arr, col in [(s_arr, "speed"), (k_arr, "degree")]:
            g = pd.DataFrame({"pix": pix, col: sub[col]}).groupby("pix")
            mu = g.mean()[col]

            arr[ti, mu.index.to_numpy()] = mu

            if col == "speed":
                 arr[ti, :] = arr[ti, :] / dT * 60


    return field_store


if __name__ == "__main__":
    project_name = "20250311_LCP1-NLSMSC"
    root = r"E:\Nick\killi_tracker"
    track_config_name = "tracking_20250328_redux"

    field_store = analyze_on_sphere(root=root,
                      project_name=project_name,
                      track_config_name=track_config_name,
                      nside=8)


    frame = 2000
    d_field = field_store["density"][frame]
    field_mask = d_field > 0
    d_field[np.isnan(d_field)] = 0

    d_field_s = smooth_field_on_sphere_knn(d_field, sigma_deg=8, k=50)
    # denom0 = smooth_field_on_sphere_knn(field_mask, sigma_deg=8, k=50)
    # d_field_s = np.divide(d_field_s, denom0)
    # denom = laplacian_smooth(field_mask, tau=0.2, steps=5)
    # d_field_s = np.divide(laplacian_smooth(d_field, tau=0.2, steps=5), denom)
    fig = plot_density_mollweide(field=d_field_s)
    fig.savefig(Path(root) / "sheesh_d.png", dpi=300, bbox_inches="tight")

    print("check")
    #
    # from scipy.interpolate import griddata
    # import matplotlib.pyplot as plt
    #
    # nside = int(np.sqrt(len(d_field_s) / 12))
    # order="nested"
    # nlon=720
    # nlat=360
    # hp = HEALPix(nside=nside, order=order)
    # lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    # lon = lon.to_value(u.rad)
    # lat = lat.to_value(u.rad)
    # lon = ((lon + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-π, π]
    #
    # # Regular grid (edges for pcolormesh)
    # lon_edges = np.linspace(-np.pi, np.pi, nlon)
    # lat_edges = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    # Lon, Lat = np.meshgrid(lon_edges, lat_edges)
    #
    # # Interpolate field to regular grid
    # vals = griddata(
    #     points=np.column_stack((lon, lat)),
    #     values=d_field_s,
    #     xi=np.column_stack((Lon.ravel(), Lat.ravel())),
    #     method="linear",
    #     fill_value=np.nan
    # ).reshape(Lat.shape)
    #
    # vals = np.nan_to_num(vals, nan=0)
    #
    # fig = plt.figure(figsize=(9, 4.5))
    # ax = fig.add_subplot(111, projection="mollweide")
    #
    # # use pcolormesh, which respects lon/lat orientation explicitly
    # pcm = ax.pcolormesh(Lon, Lat, vals,
    #                     cmap="viridis",
    #                     shading="auto")
    # plt.colorbar(pcm, orientation="horizontal", pad=0.05, label="Density")
    #
    # ax.grid(True)
    #
    # fig.savefig(Path(root) / "sheesh.png", dpi=300, bbox_inches="tight")