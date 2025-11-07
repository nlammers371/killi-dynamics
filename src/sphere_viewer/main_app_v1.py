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

# if you have this in your project, keep it; otherwise stub it
from src.data_io.track_io import _load_track_data


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
    # merge sphere geom
    m = tracks_sub.merge(
        sphere_sub[["t", "cx", "cy", "cz", "radius", "d_cx", "d_cy", "d_cz"]],
        on="t",
        how="left",
    )

    m["t"] = m["t"] - m["t"].min()  # make time start at 0

    # c_vecs = m[["x", "y", "z"]].to_numpy() - m[["cx", "cy", "cz"]].to_numpy()
    # c_unit = c_vecs / np.linalg.norm(c_vecs, axis=1, keepdims=True)
    # coords = c_unit * m["radius"].to_numpy()[:, None] + m[["cx", "cy", "cz"]].to_numpy() + m[["d_cx", "d_cy", "d_cz"]].to_numpy()
    coords = m[["x", "y", "z"]].to_numpy() + m[["d_cx", "d_cy", "d_cz"]].to_numpy()

    # Prepare Napari data (t, z, y, x)
    pts_data = np.column_stack([m["t"], coords[:, 0], coords[:, 1], coords[:, 2]])

    vals = m[point_var].to_numpy()
    cmap = plt.get_cmap(point_cmap)

    if np.issubdtype(vals.dtype, np.floating):
        normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    else:
        # simple categorical
        uniq = np.unique(vals)
        idx_map = {u: i for i, u in enumerate(uniq)}
        idx = np.array([idx_map[v] for v in vals], dtype=float)
        normed = idx / max(len(uniq) - 1, 1)

    face_color = np.asarray(cmap(normed)[:, :4], dtype=np.float32)

    viewer.add_points(
        pts_data,
        size=size,
        face_color=face_color,
        name=f"points_{point_var}",
        blending="translucent",
    )

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
    track_id_map_cache,
    verts,
    faces,
    show_patches,
    sphere_df,
    cell_radius,
    n_workers=4,
    init_frame_range=None,
):
    from napari.utils.notifications import show_info

    all_frames = np.sort(tracks["t"].unique())
    t_min, t_max = int(all_frames.min()), int(all_frames.max())

    if init_frame_range is not None:
        frame_init_min = max(t_min, init_frame_range[0])
        frame_init_max = min(t_max, init_frame_range[1])
    else:
        frame_init_min, frame_init_max = t_min, t_min  # start small

    def ensure_track_id_map(frames):
        missing = [t for t in frames if t not in track_id_map_cache]
        if missing:
            new_maps = compute_track_id_map(
                tracks,
                sphere_df,
                verts,
                cell_radius_max=cell_radius,
                n_workers=n_workers,
                frame_subset=missing,
            )
            track_id_map_cache.update(new_maps)
            show_info(f"Added {len(missing)} frames to cache.")

    def update_patches(patch_var, patch_cmap, frame_min, frame_max):
        frames = list(range(frame_min, frame_max + 1))
        ensure_track_id_map(frames)

        for layer in list(viewer.layers):
            if "cells_" in layer.name:
                viewer.layers.remove(layer)

        # stack maps
        track_id_arr = get_track_id_array(track_id_map_cache, frames)

        # subset tracks to these frames
        m1 = tracks["t"].values >= frame_min
        m2 = tracks["t"].values <= frame_max
        tracks_sub = tracks.loc[m1 & m2].copy()

        add_cell_patches_layer(
            viewer,
            verts,
            faces,
            track_id_arr,
            frames,
            tracks_sub,
            patch_var=patch_var,
            patch_cmap=patch_cmap,
        )

    def update_points(point_var, point_cmap, frame_min, frame_max):
        for layer in list(viewer.layers):
            if "points_" in layer.name:
                viewer.layers.remove(layer)

        m1 = tracks["t"].values >= frame_min
        m2 = tracks["t"].values <= frame_max
        tracks_sub = tracks.loc[m1 & m2].copy()

        s1 = sphere_df["t"].values >= frame_min
        s2 = sphere_df["t"].values <= frame_max
        sphere_sub = sphere_df.loc[s1 & s2].copy()

        add_timeaware_points_layer(
            viewer,
            tracks_sub,
            sphere_sub,
            point_var=point_var,
            point_cmap=point_cmap,
            size=cell_radius * 0.5,
        )

    @magicgui(
        patch_var={"widget_type": "ComboBox", "choices": [], "label": "Patch variable"},
        patch_cmap={"widget_type": "ComboBox", "choices": plt.colormaps(), "label": "Patch colormap"},
        point_var={"widget_type": "ComboBox", "choices": [], "label": "Point variable"},
        point_cmap={"widget_type": "ComboBox", "choices": plt.colormaps(), "label": "Point colormap"},
        frame_min={"widget_type": "Slider", "label": "Start frame", "min": t_min, "max": t_max, "step": 1},
        frame_max={"widget_type": "Slider", "label": "End frame", "min": t_min, "max": t_max, "step": 1},
        update_patches_btn={"widget_type": "PushButton", "text": "Update patches"},
        update_points_btn={"widget_type": "PushButton", "text": "Update points"},
    )
    def controls(
            patch_var=None,
            patch_cmap="magma",
            point_var=None,
            point_cmap="plasma",
            frame_min=frame_init_min,
            frame_max=frame_init_max,
            update_patches_btn=False,
            update_points_btn=False,
    ):
        if frame_min > frame_max:
            show_info("Start frame must be <= end frame")
            return

    controls.update_patches_btn.changed.connect(
        lambda _: update_patches(
            controls.patch_var.value,
            controls.patch_cmap.value,
            controls.frame_min.value,
            controls.frame_max.value,
        )
    )
    controls.update_points_btn.changed.connect(
        lambda _: update_points(
            controls.point_var.value,
            controls.point_cmap.value,
            controls.frame_min.value,
            controls.frame_max.value,
        )
    )

    viewer.window.add_dock_widget(controls, area="right")

    # Now the widgets are fully initialized, so this works
    var_choices = list(tracks.columns)
    controls.patch_var.choices = var_choices
    controls.point_var.choices = var_choices

    if "track_id" in var_choices:
        controls.patch_var.value = "track_id"
    if "mean_fluo" in var_choices:
        controls.point_var.value = "mean_fluo"


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
        radius=mean_radius * 0.95,  # slightly larger than fin mesh
        center=mean_center,
        nside=max(32, nside // 2),  # coarser mesh is fine for background
    )


    verts = 0.99 * base_verts * mean_radius + mean_center
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

    # add_interactive_controls(
    #     viewer,
    #     tracks,
    #     track_id_map_cache,
    #     verts,
    #     faces,
    #     show_patches,
    #     sphere_df=sphere,
    #     cell_radius=cell_radius,
    #     n_workers=n_workers,
    #     init_frame_range=frame_range,
    # )

    napari.run()


# =====================================================
# === Wrapper =========================================
# =====================================================
def launch_sv_wrapper(
    root: Path,
    project_name: str,
    tracking_config: str,
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
    )

    candidate_cols = [
        "mean_fluo",
        "track_class",
        "cluster_id_stitched",
        "parent_track_id",
    ]
    if deep_cells_only:
        tracks = tracks.loc[tracks["track_class"] == 0].copy()

    # keep columns
    tracks = tracks[["t", "track_id", "x", "y", "z"] + candidate_cols]

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
        frame_range=(900, 975),
        cell_radius=12.5,
        n_workers=12,
    )
