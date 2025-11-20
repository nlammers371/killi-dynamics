from astropy_healpix import HEALPix
from astropy.coordinates import CartesianRepresentation
import astropy.units as u
from src.tracking.track_processing import smooth_tracks
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from src.tracking.track_processing import preprocess_tracks
from sklearn.neighbors import BallTree
from astropy.coordinates import SphericalRepresentation
from src.cell_track_dynamics.msd_statistics import compute_track_msd

def compute_appearance_hp_region(tracks, frame_min, hp_col="hp_region", calculate_vel_corr=False):
    """
    Compute healpix region where each track first appears inside the viewer's
    time window.

    appearance_t = max(track_min_t, frame_min)
    appearance_hp_region = hp_region at appearance_t
    """

    df = tracks.copy()

    # 1) get each track's global first frame
    track_first_t = df.groupby("track_id")["t"].min()

    # 2) clamp to frame_min
    appearance_t = track_first_t.clip(lower=frame_min)   # Series indexed by track_id

    # attach appearance_t to every row
    df = df.merge(
        appearance_t.rename("appearance_t"),
        on="track_id",
        how="left"
    )

    # 3) lookup (track_id, appearance_t) → hp_region
    # We filter rows where t == appearance_t
    key = (df["t"] == df["appearance_t"])
    lookup = (
        df.loc[key]
          .groupby("track_id")[hp_col]
          .first()     # safe; at most one row per ID
          .to_dict()
    )

    # 4) map into a full-length column (aligned with df rows)
    df["appearance_hp_region"] = df["track_id"].map(lookup)


    return df["appearance_hp_region"]

#
#
#
#
#
#
# def compute_vecolicy_correlation(
#     df,
#     sphere_df=None,
#     xcol="x",
#     ycol="y",
#     zcol="z",
#     xcol_sphere="center_x_smooth",
#     ycol_sphere="center_y_smooth",
#     zcol_sphere="center_z_smooth",
#     radius=50.0,
#     project_to_sphere=False,
# ):
#     """
#     Fixed-radius local density.
#     *** IMPORTANT: preserves original df index ***
#     """
#
#     # do NOT sort or reset index
#     density = np.full(len(df), np.nan)
#
#     # group using the *existing* index values
#     groups = df.groupby("t").groups   # this returns correct indices
#
#     for t, idxs in tqdm(groups.items(), desc="Computing fixed-radius densities..."):
#         idxs = np.array(idxs)
#
#         pts = df.loc[idxs, [xcol, ycol, zcol]].to_numpy()
#
#         if project_to_sphere:
#             if sphere_df is None:
#                 raise ValueError("sphere_df required when project_to_sphere=True")
#
#             row = sphere_df.loc[sphere_df["t"] == t].iloc[0]
#             cx, cy, cz = row[xcol_sphere], row[ycol_sphere], row[zcol_sphere]
#             R = row["radius_smooth"]
#
#             v = pts - np.array([cx, cy, cz])
#             norm = np.linalg.norm(v, axis=1, keepdims=True)
#             unit = v / norm
#             pts = unit * R
#
#             cap_area = 2 * np.pi * R * radius
#         else:
#             cap_area = np.pi * radius * radius
#
#         tree = BallTree(pts)
#         # ind = tree.query_radius(pts, r=radius, count_only=False)
#
#         counts = tree.query_radius(pts, r=radius, count_only=True) - 1
#         dens = counts / cap_area
#
#         # This now correctly aligns with the original df index
#         density[idxs] = dens
#
#     return density



def assign_healpix_pixels_astropy(
        tracks_df,
        sphere_df,
        nside=16,
        tcol="t",
        xcol="x", ycol="y", zcol="z",
):
    """
    Assigns astropy-healpix pixel index to each point in tracks_df based on
    time-varying sphere center & radius from sphere_df.
    """

    # --- 1. Attach sphere center & radius to each cell-row ---
    merged = tracks_df.merge(
        sphere_df,
        on=tcol,
        how="left"
    )

    # Extract arrays
    x = merged[xcol].to_numpy()
    y = merged[ycol].to_numpy()
    z = merged[zcol].to_numpy()

    cx = merged["center_x_smooth"].to_numpy()
    cy = merged["center_y_smooth"].to_numpy()
    cz = merged["center_z_smooth"].to_numpy()

    # --- 2. Compute unit vectors pointing from center to each cell ---
    dx = x - cx
    dy = y - cy
    dz = z - cz

    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    norm[norm == 0] = np.nan

    ux = dx / norm
    uy = dy / norm
    uz = dz / norm

    # --- 3. Build CartesianRepresentation for astropy ---
    vec = CartesianRepresentation(ux, uy, uz)

    # Convert to SphericalRepresentation (required)
    sph = vec.represent_as(SphericalRepresentation)

    lon = sph.lon.to(u.rad)
    lat = sph.lat.to(u.rad)

    # --- 4. HEALPix indexing ---
    hp = HEALPix(nside=nside, order='ring', frame=None)
    pix = hp.lonlat_to_healpix(lon, lat)

    return pix

# ---------------------------------------------------------------
# Temporal smoothing (per-track moving average)
# ---------------------------------------------------------------
def _temporal_smooth(tracks, cols, window):
    out = tracks.copy()
    g = tracks.groupby("track_id", sort=False)

    for col in cols:
        # rolling() applied once per group internally in optimized cython
        out[col] = (
            g[col]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return out


# ---------------------------------------------------------------
# Spatial KNN smoothing (per time point)
# ---------------------------------------------------------------
def _spatial_knn_smooth(tracks, cols, knn):
    """
    Optimized spatial KNN smoothing.

    - Pre-splits tracks by time once
    - Builds KNN trees once per timepoint
    - Vectorized neighbor mean calculation (no inner Python loops)
    """
    tracks = tracks.copy()
    all_times = np.sort(tracks["t"].unique())
    N = len(tracks)

    # ------------------------------------------------------------
    # Pre-split tracks by time to avoid repeated indexing
    # ------------------------------------------------------------
    time_groups = {t: tracks.index[tracks["t"].values == t] for t in all_times}

    # ------------------------------------------------------------
    # Precompute KNN neighbor lists per time (indices relative to sub-index)
    # ------------------------------------------------------------
    knn_cache = {}
    for t, idxs in time_groups.items():
        pts = tracks.loc[idxs, ["x", "y", "z"]].to_numpy(float)
        n = len(idxs)
        if n == 0:
            continue

        K = min(knn, n)
        nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(pts)
        _, neigh = nbrs.kneighbors(pts)  # shape (n, K)
        knn_cache[t] = neigh

    # ------------------------------------------------------------
    # For each metric, smooth using cached KNN
    # ------------------------------------------------------------
    for col in cols:
        out = np.full(N, np.nan)

        for t, idxs in time_groups.items():
            neigh = knn_cache.get(t, None)
            if neigh is None:
                continue

            vals = tracks.loc[idxs, col].to_numpy(float)   # (n,)
            # Broadcast neighbor values into matrix (n, K)
            nn_vals = vals[neigh]                           # (n, K)

            # Mask non-finite
            mask = np.isfinite(nn_vals)

            # Compute mean over valid neighbors
            # If no finite neighbors → NaN
            with np.errstate(invalid="ignore"):
                mean_vals = np.divide(
                    nn_vals.sum(axis=1),
                    mask.sum(axis=1),
                    where=mask.sum(axis=1) > 0
                )

            # Store back into output (global)
            out[idxs] = mean_vals

        tracks[col] = out

    return tracks



# ---------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------
def process_tracks(
        tracks,
        sphere,
        metrics=None,
        deep_cells_only=True,
        remove_stationary=True,
        smooth_metrics=True,
        smooth_window=5,
        smooth_knn=5
):
    # -----------------------------------------------------------
    # Pre-filtering
    # -----------------------------------------------------------
    if deep_cells_only:
        tracks = tracks.loc[tracks["track_class"] == 0].copy().reset_index(drop=True)

    added_cols = []
    metrics_to_smooth = []

    print("Preprocessing tracks...")
    tracks = preprocess_tracks(tracks)

    if remove_stationary:
        tracks = tracks[~tracks["track_mostly_stationary"]].reset_index(drop=True)
    print("Done.")
    # -----------------------------------------------------------
    # Merge external metrics
    # -----------------------------------------------------------
    if metrics is not None:
        metric_cols = [c for c in metrics.columns if c not in ["track_id", "t"]]
        tracks = tracks.merge(metrics, on=["track_id", "t"], how="left")

        added_cols += metric_cols
        metrics_to_smooth += metric_cols

    # -----------------------------------------------------------
    # Sphere region annotation
    # -----------------------------------------------------------
    tracks["hp_region"] = assign_healpix_pixels_astropy(tracks, sphere, nside=16)
    added_cols.append("hp_region")

    # Track appearance time
    track_start_times = tracks.groupby("track_id")["t"].min()
    tracks = tracks.merge(track_start_times.rename("t_start"), on="track_id", how="left")
    added_cols.append("t_start")

    # -----------------------------------------------------------
    # Perform smoothing
    # -----------------------------------------------------------
    if smooth_metrics and metrics_to_smooth:
        print("Temporal smoothing...")
        tracks = _temporal_smooth(tracks, metrics_to_smooth, smooth_window)

        print("Spatial KNN smoothing...")
        tracks = _spatial_knn_smooth(tracks, metrics_to_smooth, smooth_knn)
    print("Done.")
    return tracks, added_cols
