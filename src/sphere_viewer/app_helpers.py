from astropy_healpix import HEALPix
from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from src.tracking.track_processing import preprocess_tracks
from sklearn.neighbors import BallTree
from astropy.coordinates import SphericalRepresentation

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


def compute_surface_density(
    df,
    sphere_df=None,
    xcol="x",
    ycol="y",
    zcol="z",
    xcol_sphere="center_x_smooth",
    ycol_sphere="center_y_smooth",
    zcol_sphere="center_z_smooth",
    radius=50.0,
    project_to_sphere=False,
):
    """
    Fixed-radius local density.
    *** IMPORTANT: preserves original df index ***
    """

    # do NOT sort or reset index
    density = np.full(len(df), np.nan)

    # group using the *existing* index values
    groups = df.groupby("t").groups   # this returns correct indices

    for t, idxs in tqdm(groups.items(), desc="Computing fixed-radius densities..."):
        idxs = np.array(idxs)

        pts = df.loc[idxs, [xcol, ycol, zcol]].to_numpy()

        if project_to_sphere:
            if sphere_df is None:
                raise ValueError("sphere_df required when project_to_sphere=True")

            row = sphere_df.loc[sphere_df["t"] == t].iloc[0]
            cx, cy, cz = row[xcol_sphere], row[ycol_sphere], row[zcol_sphere]
            R = row["radius_smooth"]

            v = pts - np.array([cx, cy, cz])
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            unit = v / norm
            pts = unit * R

            cap_area = 2 * np.pi * R * radius
        else:
            cap_area = np.pi * radius * radius

        tree = BallTree(pts)
        # ind = tree.query_radius(pts, r=radius, count_only=False)

        counts = tree.query_radius(pts, r=radius, count_only=True) - 1
        dens = counts / cap_area

        # This now correctly aligns with the original df index
        density[idxs] = dens

    return density


def compute_vecolicy_correlation(
    df,
    sphere_df=None,
    xcol="x",
    ycol="y",
    zcol="z",
    xcol_sphere="center_x_smooth",
    ycol_sphere="center_y_smooth",
    zcol_sphere="center_z_smooth",
    radius=50.0,
    project_to_sphere=False,
):
    """
    Fixed-radius local density.
    *** IMPORTANT: preserves original df index ***
    """

    # do NOT sort or reset index
    density = np.full(len(df), np.nan)

    # group using the *existing* index values
    groups = df.groupby("t").groups   # this returns correct indices

    for t, idxs in tqdm(groups.items(), desc="Computing fixed-radius densities..."):
        idxs = np.array(idxs)

        pts = df.loc[idxs, [xcol, ycol, zcol]].to_numpy()

        if project_to_sphere:
            if sphere_df is None:
                raise ValueError("sphere_df required when project_to_sphere=True")

            row = sphere_df.loc[sphere_df["t"] == t].iloc[0]
            cx, cy, cz = row[xcol_sphere], row[ycol_sphere], row[zcol_sphere]
            R = row["radius_smooth"]

            v = pts - np.array([cx, cy, cz])
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            unit = v / norm
            pts = unit * R

            cap_area = 2 * np.pi * R * radius
        else:
            cap_area = np.pi * radius * radius

        tree = BallTree(pts)
        # ind = tree.query_radius(pts, r=radius, count_only=False)

        counts = tree.query_radius(pts, r=radius, count_only=True) - 1
        dens = counts / cap_area

        # This now correctly aligns with the original df index
        density[idxs] = dens

    return density



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

    # --- 1. Attach sphere center & radius ---
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


def process_tracks(tracks,
                   sphere,
                   deep_cells_only=True,
                   remove_stationary=True,
                   fields_to_smooth=None
                   ):

    # filter for deep cells only
    if deep_cells_only:
        tracks = tracks.loc[tracks["track_class"] == 0].copy().reset_index(drop=True)

    added_cols = []

    # Preprocess tracks
    tracks = preprocess_tracks(tracks)

    if remove_stationary:
        tracks = tracks[~tracks["track_mostly_stationary"]].reset_index(drop=True)

    # calculate metrics
    tracks["nn_density"] = compute_surface_density(tracks)
    added_cols += ["nn_density"]

    # assign to sphere regions
    tracks["hp_region"] = assign_healpix_pixels_astropy(tracks, sphere, nside=16)
    added_cols += ["hp_region"]

    # get appearance time
    track_start_times = tracks.groupby("track_id")["t"].min()
    tracks = tracks.merge(
        track_start_times.rename("t_start"),
        on="track_id",
        how="left"
    )
    added_cols += ["t_start"]

    return tracks, added_cols