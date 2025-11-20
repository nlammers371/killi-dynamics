import napari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull
from astropy_healpix import HEALPix
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from magicgui import magicgui
from pathlib import Path
from src.sphere_viewer.app_helpers import process_tracks, compute_appearance_hp_region
# if you have this in your project, keep it; otherwise stub it
from src.data_io.track_io import _load_track_data, _load_tracks


def add_backdrop_sphere(
    viewer,
    radius: float,
    center: np.ndarray,
    nside: int = 64,
    color=(0, 0, 0, 1.0),  # solid black RGBA
):
    """Add a simple opaque spherical backdrop to the scene."""
    hp = HEALPix(nside=nside, order="nested")
    npix = hp.npix
    x, y, z = hp.healpix_to_xyz(np.arange(npix))
    verts = np.vstack((x, y, z)).T * radius + center
    hull = ConvexHull(verts)
    faces = hull.simplices

    # add as a solid surface with constant color
    viewer.add_surface(
        (verts, faces, np.ones(len(verts))),
        name="embryo surface",
        colormap=None,
        vertex_colors=np.tile(color, (len(verts), 1)),
        opacity=0.95,
        shading="flat",
        blending="opaque",
    )


# =====================================================
# === HEALPix Sphere Construction =====================
# =====================================================
def make_healpix_sphere(nside=32, radius=1.0):
    hp = HEALPix(nside=nside, order="nested")
    npix = hp.npix
    x, y, z = hp.healpix_to_xyz(np.arange(npix))
    verts = np.vstack((x, y, z)).T * radius
    hull = ConvexHull(verts)
    faces = hull.simplices
    return verts, faces


# =====================================================
# === Core mapping logic ==============================
# =====================================================
def assign_vertices_kdtree(verts_world, centers_world, track_ids, cell_radius_max=12.5):
    tree = cKDTree(centers_world)
    dists, idxs = tree.query(verts_world, k=1)
    assigned = np.full(len(verts_world), -1, dtype=np.int32)
    mask = dists < cell_radius_max
    assigned[mask] = track_ids[idxs[mask]]
    return assigned


def _map_frame_ids(
    t,
    tracks,
    sphere_df,
    base_verts,
    cell_radius_max,
    xcol="cx",
    ycol="cy",
    zcol="cz",
    rcol="radius",
):
    # sphere params for this frame
    s_mask = sphere_df["t"].values == t
    if not np.any(s_mask):
        raise ValueError(f"No sphere params for t={t}")

    row = sphere_df.loc[s_mask].iloc[0]
    cx, cy, cz = row[xcol], row[ycol], row[zcol]
    radius = float(row[rcol])

    # move base verts to world
    verts_world = base_verts * radius + np.array([cx, cy, cz], dtype=float)

    # tracks for this frame
    tr_mask = tracks["t"].values == t
    frame_tracks = tracks.loc[tr_mask]
    centers_world = frame_tracks[["x", "y", "z"]].to_numpy()

    # project track centers to sphere surface (stay consistent with surface)
    c_vecs = centers_world - np.array([cx, cy, cz], dtype=float)
    c_norm = np.linalg.norm(c_vecs, axis=1, keepdims=True)
    c_unit = c_vecs / c_norm
    centers_sphere = c_unit * radius + np.array([cx, cy, cz], dtype=float)

    track_ids = frame_tracks["track_id"].to_numpy()

    assigned = assign_vertices_kdtree(
        verts_world,
        centers_sphere,
        track_ids,
        cell_radius_max=cell_radius_max,
    )
    return assigned


def compute_track_id_map(
    tracks,
    sphere_df,
    base_verts,
    cell_radius_max=12.5,
    n_workers=4,
    frame_subset=None,
):
    if frame_subset is None:
        all_t = np.sort(tracks["t"].unique())
    else:
        all_t = np.sort(np.atleast_1d(frame_subset))

    run_mapping = partial(
        _map_frame_ids,
        tracks=tracks,
        sphere_df=sphere_df,
        base_verts=base_verts,
        cell_radius_max=cell_radius_max,
    )

    if n_workers > 1 and len(all_t) > 1:
        results = process_map(
            run_mapping,
            all_t,
            max_workers=n_workers,
            chunksize=1,
            desc=f"Computing track ID map ({len(all_t)} frames)",
        )
    else:
        results = [run_mapping(t) for t in tqdm(all_t, desc="Computing track ID map")]

    return dict(zip(all_t, results))


def get_track_id_array(track_id_map_cache, frames):
    missing = [t for t in frames if t not in track_id_map_cache]
    if missing:
        raise KeyError(f"Missing track_id_map entries for frames: {missing}")
    return np.stack([track_id_map_cache[t] for t in frames])


def map_values_from_track_ids_array(track_id_arr, frames, tracks_sub, quant_var):
    """
    track_id_arr: (F, Nverts) for exactly these 'frames'
    tracks_sub: tracks filtered to those frames
    """
    # map (t, track_id) -> value
    g = tracks_sub.groupby(["t", "track_id"])[quant_var].mean()
    lookup = g.to_dict()

    F, N = track_id_arr.shape
    values = np.zeros((F, N), dtype=np.float32)

    for fi, t in enumerate(tqdm(frames, desc=f"Mapping '{quant_var}' values")):
        tids = track_id_arr[fi]
        vals = np.zeros(N, dtype=np.float32)
        uniq = np.unique(tids[tids >= 0])
        for v in uniq:
            vals[tids == v] = lookup.get((t, v), 0.0)
        values[fi] = vals
    return values


def compute_border_mask(faces, track_id_arr):
    F, N = track_id_arr.shape
    border_mask = np.zeros_like(track_id_arr, dtype=np.float32)

    for fi in range(F):
        tids = track_id_arr[fi]
        face_tids = tids[faces]  # (n_faces, 3)
        border_faces = np.any(face_tids != face_tids[:, [0]], axis=1)
        border_vertices = np.unique(faces[border_faces].ravel())
        border_mask[fi, border_vertices] = 1.0

    return border_mask


# =====================================================
# === Visualization layers ============================
# =====================================================
def add_cell_patches_layer(
    viewer,
    verts,
    faces,
    track_id_arr,
    frames,
    tracks_sub,
    patch_var,
    patch_cmap,
    invert_edges=False,  # new toggle
):
    """Add surface patches colored by track-level variable.

    invert_edges=False → color interiors, transparent edges
    invert_edges=True  → color edges, transparent interiors
    """
    values = map_values_from_track_ids_array(track_id_arr, frames, tracks_sub, patch_var)
    border_mask = compute_border_mask(faces, track_id_arr)

    col = tracks_sub[patch_var]
    is_float = np.issubdtype(col.dtype, np.floating)

    if is_float:
        vmin, vmax = values.min(), values.max()
        normed = (values - vmin) / (vmax - vmin + 1e-9)
    else:
        flat_vals = values.reshape(-1)
        uniq = np.unique(flat_vals[flat_vals >= 0])
        cat_map = {u: i for i, u in enumerate(uniq)}
        K = len(cat_map)
        normed = np.zeros_like(values, dtype=float)
        for u, i in cat_map.items():
            normed[values == u] = float(i) / max(K - 1, 1)

    # === Apply transparency logic ===
    cmap = plt.get_cmap(patch_cmap)
    rgba = cmap(normed)  # (F, N, 4)

    # ---- Apply transparency masks ----
    if invert_edges:
        rgba[border_mask <= 0, :3] = 0  # transparent interior
    else:
        rgba[border_mask > 0, :3] = 0  # transparent edges
    rgba[values == 0, :3] = 0

    # ---- Add surface ----
    viewer.add_surface(
        (verts, faces, normed),
        vertex_colors=rgba,  # key line — use explicit RGBA
        name=f"cells_{patch_var}",
        opacity=1.0,
        shading="none",
    )


def add_timeaware_points_layer(
    viewer,
    tracks_sub,
    sphere_sub,
    point_cmap,
    point_var="mean_fluo",
    size=0.02,
):
    # -------------------------------
    # Merge sphere geom
    # -------------------------------
    m = tracks_sub.merge(
        sphere_sub[["t", "cx", "cy", "cz", "radius", "d_cx", "d_cy", "d_cz"]],
        on="t",
        how="left",
    )

    # Reset time origin
    m["t"] = m["t"] - m["t"].min()

    # Drift-corrected coordinates
    coords = m[["x", "y", "z"]].to_numpy() + m[["d_cx", "d_cy", "d_cz"]].to_numpy()

    # Prepare Napari data: (t, x, y, z)
    pts_data = np.column_stack([m["t"], coords[:, 0], coords[:, 1], coords[:, 2]])

    # --------------------------------
    # Color logic (unchanged)
    # --------------------------------
    vals_raw = m[point_var].to_numpy()

    # 1) Replace NaNs with sentinel
    sentinel = -1000.0
    vals = vals_raw.copy()
    mask_nan = np.isnan(vals)
    vals[mask_nan] = sentinel

    cmap = plt.get_cmap(point_cmap)

    # 2) Floating vs categorical logic unchanged
    if np.issubdtype(vals.dtype, np.floating):
        # IMPORTANT: ignore sentinel when computing min/max
        vmin0 = np.percentile(vals[~mask_nan], 1) if (~mask_nan).any() else 0.0
        vmax0 = np.percentile(vals[~mask_nan], 99) if (~mask_nan).any() else 1.0

        # Normalization only for non-NaN values
        normed = np.empty_like(vals, dtype=float)
        normed[~mask_nan] = (vals[~mask_nan] - vmin0) / (vmax0 - vmin0 + 1e-9)
        normed[mask_nan] = 0.0  # placeholder (we override color anyway)

    else:
        uniq = np.unique(vals[~mask_nan])
        idx_map = {u: i for i, u in enumerate(uniq)}
        idx = np.array([idx_map[v] if v in idx_map else 0 for v in vals], dtype=float)
        normed = idx / max(len(uniq) - 1, 1)
        # vmin0, vmax0 = 0, 1

    # 3) Get the colormap
    face_color = np.asarray(cmap(normed)[:, :4], dtype=np.float32)

    # 4) Override NaN/sentinel points to gray
    face_color[mask_nan] = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

    # --------------------------------
    # Add layer
    # --------------------------------
    layer = viewer.add_points(
        pts_data,
        size=size,
        face_color=face_color,
        name=f"points_{point_var}",
        blending="translucent",

        # raw values including real NaN preserved
        properties={point_var: vals_raw},
    )

    # ---------------------------------------------------------
    # Attach helper method to dynamically rescale the colors
    # ---------------------------------------------------------
    def _rescale(new_vmin, new_vmax):
        vals = layer.properties[point_var]

        # new normalization
        n = (vals - new_vmin) / (new_vmax - new_vmin + 1e-9)
        n = np.clip(n, 0, 1)

        cmap = plt.get_cmap(point_cmap)
        new_fc = cmap(n)[:, :4].astype(np.float32)
        layer.face_color = new_fc

    layer.rescale = _rescale  # attach to layer instance

    # Return the layer in case caller wants to modify it
    return layer



def add_tracks_layer(
    viewer,
    tracks_sub: pd.DataFrame,
    sphere_sub: pd.DataFrame,
    track_var: str = "mean_fluo",
    track_cmap: str = "viridis",
):
    """
    Add a napari Tracks layer from a dataframe with columns:
    ['t', 'track_id', 'x', 'y', 'z', ...].

    Colors tracks by `track_var` using napari's properties/color_by mechanism.
    """
    required = {"t", "track_id", "x", "y", "z"}
    if not required.issubset(tracks_sub.columns):
        missing = required - set(tracks_sub.columns)
        raise ValueError(f"tracks_sub is missing required columns: {missing}")

    # merge sphere geom
    m = tracks_sub.merge(
        sphere_sub[["t", "cx", "cy", "cz", "radius", "d_cx", "d_cy", "d_cz"]],
        on="t",
        how="left",
    )
    # correct for center drift
    m[["x", "y", "z"]] = m[["x", "y", "z"]].to_numpy() + m[["d_cx", "d_cy", "d_cz"]].to_numpy()
    tracks_sub["t"] = tracks_sub["t"] - tracks_sub["t"].min()  # start at t=0
    tracks_data = tracks_sub[["track_id", "t", "x", "y", "z"]].to_numpy()

    # this is the column we want to color by
    if track_var not in tracks_sub.columns:
        # fall back to track_id if the requested var isn’t there
        track_var = "track_id"

    props = {track_var: tracks_sub[track_var].to_numpy()}

    viewer.add_tracks(
        tracks_data,
        name=f"tracks_{track_var}",
        properties=props,
        color_by=track_var,
        colormap=track_cmap,
        tail_length=10,
        head_length=0,
        blending="translucent",
    )






# =====================================================
# === Interactive controls ============================
# =====================================================
def add_interactive_controls(
    viewer,
    tracks,
    sphere_df,
    allowed_vars,
    cell_radius,
    init_frame_range=None,
):
    from magicgui import magicgui
    from napari.utils.notifications import show_info

    # ---------------------------------------------------------
    # Frame bookkeeping
    # ---------------------------------------------------------
    all_frames = np.sort(tracks["t"].unique())
    t_min, t_max = int(all_frames.min()), int(all_frames.max())

    if init_frame_range:
        frame_init_min = init_frame_range[0]
        frame_init_max = init_frame_range[1]
    else:
        frame_init_min = frame_init_max = t_min

    # ---------------------------------------------------------
    # Remove any existing points/tracks layers
    # ---------------------------------------------------------
    def drop_layers(prefix):
        for layer in list(viewer.layers):
            name = layer.name
            # napari sometimes appends suffixes like " [1]"
            if name.startswith(prefix):
                viewer.layers.remove(layer)

    # ---------------------------------------------------------
    # Update POINTS
    # ---------------------------------------------------------
    def update_points(point_var, point_cmap, frame_min, frame_max):
        drop_layers("points_")

        if point_var == "appearance_hp_region":
            tracks["appearance_hp_region"] = compute_appearance_hp_region(
                tracks,
                frame_min=controls.frame_min.value,
                hp_col="hp_region",
            )

        m1 = tracks["t"] >= frame_min
        m2 = tracks["t"] <= frame_max
        tracks_sub = tracks.loc[m1 & m2].copy()

        s1 = sphere_df["t"] >= frame_min
        s2 = sphere_df["t"] <= frame_max
        sphere_sub = sphere_df.loc[s1 & s2].copy()

        layer = add_timeaware_points_layer(
            viewer,
            tracks_sub,
            sphere_sub,
            point_var=point_var,
            point_cmap=point_cmap,
            size=cell_radius,
        )
        if point_var == "t_start":
            layer.rescale(frame_min, frame_max)

    # ---------------------------------------------------------
    # Update TRACKS
    # ---------------------------------------------------------
    def update_tracks(track_var, track_cmap, frame_min, frame_max):
        drop_layers("tracks_")

        m1 = tracks["t"] >= frame_min
        m2 = tracks["t"] <= frame_max
        tracks_sub = tracks.loc[m1 & m2].copy()

        s1 = sphere_df["t"] >= frame_min
        s2 = sphere_df["t"] <= frame_max
        sphere_sub = sphere_df.loc[s1 & s2].copy()

        tracks["appearance_hp_region"] = compute_appearance_hp_region(
            tracks,
            frame_min=controls.frame_min.value,
            hp_col="hp_region",
        )

        add_tracks_layer(
            viewer,
            tracks_sub,
            sphere_sub=sphere_sub,
            track_var=track_var,
            track_cmap=track_cmap,
        )

    # ---------------------------------------------------------
    # MAGICGUI controls — sliders now automatically trigger updates
    # ---------------------------------------------------------
    @magicgui(
        point_var={"widget_type": "ComboBox", "choices": allowed_vars},
        point_cmap={"widget_type": "ComboBox", "choices": plt.colormaps()},
        track_var={"widget_type": "ComboBox", "choices": allowed_vars},
        track_cmap={"widget_type": "ComboBox", "choices": plt.colormaps()},

        frame_min={"widget_type": "Slider", "min": t_min, "max": t_max, "step": 1},
        frame_max={"widget_type": "Slider", "min": t_min, "max": t_max, "step": 1},
    )
    def controls(
        point_var="mean_fluo",
        point_cmap="turbo",
        track_var="mean_fluo",
        track_cmap="viridis",
        frame_min=frame_init_min,
        frame_max=frame_init_max,
    ):
        # Live update
        if frame_min > frame_max:
            show_info("Start frame must be <= end frame")
            return

        update_points(point_var, point_cmap, frame_min, frame_max)
        update_tracks(track_var, track_cmap, frame_min, frame_max)

    # ---------------------------------------------------------
    # Add dock widget
    # ---------------------------------------------------------
    viewer.window.add_dock_widget(controls, area="right")

    return controls





# =====================================================
# === Main launcher ===================================
# =====================================================
def launch_sphere_viewer(
    tracks,
    sphere,
    nside=128,
    patch_var="track_id",
    point_var="mean_fluo",
    track_var="mean_fluo",
    allowed_vars=None,
    show_points=True,
    show_patches=True,
    show_tracks=True,
    cell_radius=12.5,
    n_workers=4,
    patch_cmap=None,
    point_cmap=None,
    track_cmap=None,
    frame_range=None,
):

    if allowed_vars is None:
        allowed_vars = list(tracks.columns)

    if patch_cmap is None and show_patches:
        patch_cmap = (
            "magma" if np.issubdtype(tracks[patch_var].dtype, np.floating) else "tab20"
        )
    if point_cmap is None and show_points:
        point_cmap = (
            "turbo" if np.issubdtype(tracks[point_var].dtype, np.floating) else "tab20"
        )
    if track_cmap is None and show_tracks:
        track_cmap = (
            "viridis" if np.issubdtype(tracks[track_var].dtype, np.floating) else "tab20"
        )

    viewer = napari.Viewer(ndisplay=3)
    base_verts, faces = make_healpix_sphere(nside=nside)  # xyz

    # calculate stable reference frame and adjustent factors
    mean_radius = sphere["radius"].mean()
    mean_center = sphere[["cx", "cy", "cz"]].mean().to_numpy()

    # ---- BACKDROP SPHERE ----
    add_backdrop_sphere(
        viewer,
        radius=mean_radius * 0.85,  # slightly larger than fin mesh
        center=mean_center,
        nside=max(32, nside // 2),  # coarser mesh is fine for background
    )


    verts = base_verts * mean_radius + mean_center
    center_drift_array = mean_center - sphere[["cx", "cy", "cz"]].to_numpy()
    sphere[["d_cx", "d_cy", "d_cz"]] = center_drift_array

    track_id_map_cache = {}

    # decide initial frames
    all_frames = np.sort(tracks["t"].unique())
    if frame_range is None:
        frame_subset = all_frames[:50]  # small initial
    else:
        frame_subset = np.arange(frame_range[0], frame_range[1] + 1)

    # precompute these
    if show_patches:
        track_id_map_cache.update(
            compute_track_id_map(
                tracks,
                sphere,
                base_verts,
                cell_radius_max=cell_radius,
                n_workers=n_workers,
                frame_subset=frame_subset,
            )
        )

    # initial subsets for layers
    fm = frame_subset[0]
    fx = frame_subset[-1]
    m1 = tracks["t"].values >= fm
    m2 = tracks["t"].values <= fx
    tracks_sub = tracks.loc[m1 & m2].copy()

    s1 = sphere["t"].values >= fm
    s2 = sphere["t"].values <= fx
    sphere_sub = sphere.loc[s1 & s2].copy()


    if show_patches:
        # stack initial maps
        track_id_arr = get_track_id_array(track_id_map_cache, frame_subset)

        add_cell_patches_layer(
            viewer,
            verts,
            faces,
            track_id_arr,
            frame_subset,
            tracks_sub,
            patch_var=patch_var,
            patch_cmap=patch_cmap,
        )

    if show_points:
        add_timeaware_points_layer(
            viewer,
            tracks_sub,
            sphere_sub,
            point_var=point_var,
            point_cmap=point_cmap,
            size=cell_radius,
        )

    if show_tracks:
        add_tracks_layer(
            viewer,
            tracks_sub,
            sphere_sub=sphere_sub,
            track_var=track_var,
            track_cmap=track_cmap,
        )

    # allowed_vars = list(tracks.columns)
    tracks["appearance_hp_region"] = compute_appearance_hp_region(
        tracks,
        frame_min=frame_subset[0],
        hp_col="hp_region",
    )

    add_interactive_controls(
        viewer,
        tracks=tracks,
        sphere_df=sphere,
        allowed_vars=allowed_vars + ["appearance_hp_region"],  # <–– this is new
        cell_radius=cell_radius,
        init_frame_range=frame_range,
    )

    napari.run()


# =====================================================
# === Wrapper =========================================
# =====================================================
def launch_sv_wrapper(
    root: Path,
    project_name: str,
    tracking_config: str,
    used_flow: bool = True,
    nside: int = 128,
    patch_var: str = "track_id",
    point_var: str = "mean_fluo",
    track_var: str = "mean_fluo",
    show_points: bool = True,
    show_tracks: bool = True,
    show_patches: bool = True,
    cell_radius: float = 12.5,
    n_workers: int = 4,
    deep_cells_only: bool = True,
    frame_range: tuple[int, int] | None = None,
):
    tracks, sphere = _load_track_data(
        root=root,
        rescale_tracks=True,
        project_name=project_name,
        tracking_config=tracking_config,
        prefer_smoothed=True,
        prefer_flow=used_flow,
    )

    _, tracking_dir = _load_tracks(
        root=root,
        project_name=project_name,
        tracking_config=tracking_config,
        prefer_smoothed=True,
        prefer_flow=used_flow,
    )
    metric_file = tracking_dir / "cell_dynamics_metrics.csv"
    if metric_file.is_file():
        metrics = pd.read_csv(metric_file)
        # tracking_dircks = tracks.merge(
        #     metrics,
        #     on=["track_id", "t"],
        #     how="left",
        # )
    else:
        metrics = None

    candidate_cols = [
        "mean_fluo",
        "track_class",
        "cluster_id_stitched",
        "parent_track_id",
    ]
    # if deep_cells_only:
    #     tracks = tracks.loc[tracks["track_class"] == 0].copy()

    # perform QC and calculate quantities
    tracks, added_cols = process_tracks(
        tracks,
        sphere,
        metrics,
        deep_cells_only=deep_cells_only,
        remove_stationary=True,
    )

    # keep columns
    tracks = tracks[["t", "track_id", "x", "y", "z"] + candidate_cols + added_cols]

    sphere = sphere[
        ["t", "center_x_smooth", "center_y_smooth", "center_z_smooth", "radius_smooth"]
    ].rename(
        columns={
            "center_x_smooth": "cx",
            "center_y_smooth": "cy",
            "center_z_smooth": "cz",
            "radius_smooth": "radius",
        }
    )

    launch_sphere_viewer(
        tracks,
        sphere,
        nside=nside,
        patch_var=patch_var,
        allowed_vars=candidate_cols + added_cols,
        point_var=point_var,
        track_var=track_var,
        show_points=show_points,
        show_tracks=show_tracks,
        show_patches=show_patches,
        cell_radius=cell_radius,
        n_workers=n_workers,
        frame_range=frame_range,
    )


if __name__ == "__main__":
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    tracking_config = "tracking_20251102"

    launch_sv_wrapper(
        root=root,
        project_name=project_name,
        tracking_config=tracking_config,
        nside=256,
        show_points=True,
        show_tracks=True,
        show_patches=False,
        frame_range=(800, 930),
        cell_radius=18,
        n_workers=1,
    )
