import napari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import SphericalVoronoi, ConvexHull
import trimesh
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from scipy.spatial import SphericalVoronoi
import trimesh

def make_intermediate_sphere(subdiv=8, keep_fraction=0.5, radius=1.0):
    base = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
    verts = base.vertices

    # Randomly downsample vertices
    n_keep = int(len(verts) * keep_fraction)
    keep_idx = np.random.choice(len(verts), n_keep, replace=False)
    verts_keep = verts[keep_idx]

    # Normalize back to sphere
    verts_keep /= np.linalg.norm(verts_keep, axis=1, keepdims=True)
    verts_keep *= radius

    # Re-triangulate with convex hull
    hull = ConvexHull(verts_keep)
    faces = hull.simplices
    return verts_keep, faces

# def make_decimated_sphere(subdiv=8, target_fraction=0.5, radius=1.0, smooth_iters=3):
#     mesh = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
#     # n_faces = len(mesh.faces)
#     # target_faces = int(n_faces * target_fraction)
#     if target_fraction == 1.0:
#         return mesh.vertices, mesh.faces
#
#     # Perform simplification
#     mesh_simplified = mesh.simplify_quadric_decimation(1-target_fraction)
#
#     # --- Reproject vertices to sphere ---
#     verts = mesh_simplified.vertices
#     verts = radius * (verts / np.linalg.norm(verts, axis=1, keepdims=True))
#     mesh_simplified.vertices = verts
#
#     # --- Optional Laplacian smoothing for uniformity ---
#     for _ in range(smooth_iters):
#         lap = mesh_simplified.vertex_neighbors
#         v_new = np.stack([
#             verts[i] if not lap[i] else np.mean(verts[lap[i]], axis=0)
#             for i in range(len(verts))
#         ])
#         verts = radius * (v_new / np.linalg.norm(v_new, axis=1, keepdims=True))
#         mesh_simplified.vertices = verts
#
#     return mesh_simplified.vertices, mesh_simplified.faces

def make_dense_sphere_mesh(subdiv=6, radius=1.0):
    """Return (verts, faces) for a dense, uniform icosphere."""
    mesh = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
    return mesh.vertices, mesh.faces

def compute_border_mask(faces, track_id_map):
    """
    Create a border mask per vertex, highlighting edges between different track_ids.
    Returns an array of shape (T, Nverts) with 1 near borders, 0 inside cells.
    """
    n_frames = track_id_map.shape[0]
    border_mask = np.zeros_like(track_id_map, dtype=np.float32)

    for fi in tqdm(range(n_frames), desc="Computing border masks"):
        tids = track_id_map[fi]
        face_tids = tids[faces]
        # mark faces where not all vertices have same ID
        border_faces = np.any(face_tids != face_tids[:, [0]], axis=1)
        border_vertices = np.unique(faces[border_faces].ravel())
        border_mask[fi, border_vertices] = 1.0

    return border_mask

def make_test_sphere_df(n_frames=10,
                        base_radius=1.0,
                        drift_vec=(0.02, 0.0, 0.0),
                        radius_growth=0.005):
    """
    Generate a synthetic sphere_df for testing variable sphere geometry.

    Parameters
    ----------
    n_frames : int
        Number of timepoints.
    base_radius : float
        Starting radius.
    drift_vec : tuple of 3 floats
        Direction and magnitude of center drift per frame (in same units as coords).
    radius_growth : float
        Increment of radius per frame.

    Returns
    -------
    sphere_df : pd.DataFrame
        Columns: ['t', 'cx', 'cy', 'cz', 'radius']
    """
    drift_vec = np.asarray(drift_vec, dtype=float)
    centers = np.outer(np.arange(n_frames), drift_vec)
    radii = base_radius + np.arange(n_frames) * radius_growth

    sphere_df = pd.DataFrame({
        "t": np.arange(n_frames),
        "cx": centers[:, 0],
        "cy": centers[:, 1],
        "cz": centers[:, 2],
        "radius": radii
    })
    return sphere_df

def assign_vertices_kdtree(verts_world, centers_world, track_ids, cell_radius_max=12.5):
    """
    KDTree in world coords; cutoff is angular using this frame's radius.
    verts_world: (Nv, 3) in world space
    centers_world: (Nc, 3) in world space
    radius: float, sphere radius for this frame
    """
    tree = cKDTree(centers_world)
    dists, idxs = tree.query(verts_world, k=1)

    # convert chord length to angular distance on sphere of radius R
    # angles = 2 * np.arcsin(np.clip(dists / (2.0 * radius), 0, 1))

    assigned = np.full(len(verts_world), -1, dtype=np.int32)
    mask = dists < cell_radius_max
    assigned[mask] = track_ids[idxs[mask]]
    return assigned

def _map_frame_ids(t, tracks, sphere_df, base_verts, cell_radius_max, xcol="cx", ycol="cy", zcol="cz", rcol="radius"):

    row = sphere_df.query("t == @t")
    if row.empty:
        raise ValueError(f"No sphere params for t={t}")
    cx, cy, cz = row[[xcol, ycol, zcol]].iloc[0]
    radius = float(row[rcol].iloc[0])

    # put this frame's sphere vertices into world coords
    verts_world = base_verts * radius + np.array([cx, cy, cz])

    frame_tracks = tracks.query("t == @t")
    centers_world = frame_tracks[["x", "y", "z"]].to_numpy()
    c_vecs = centers_world - np.array([cx, cy, cz])
    cu_vecs = np.divide(c_vecs,  np.linalg.norm(c_vecs, axis=1)[:, None])
    centers_sphere = cu_vecs * radius + np.array([cx, cy, cz])
    track_ids = frame_tracks["track_id"].to_numpy()

    assigned = assign_vertices_kdtree(
        verts_world,
        centers_sphere,
        track_ids,
        cell_radius_max=cell_radius_max,
    )
    return assigned

def compute_track_id_map(tracks, sphere_df, base_verts, cell_radius_max=12.5, n_workers=4):
    """
    base_verts: unit icosphere (centered at 0)
    sphere_df: columns ['t', 'cx', 'cy', 'cz', 'radius']
    returns: (T, Nverts) track_id map
    """
    all_t = np.sort(tracks["t"].unique())

    run_mapping = partial(_map_frame_ids,
                          tracks=tracks,
                          sphere_df=sphere_df,
                          base_verts=base_verts,
                          cell_radius_max=cell_radius_max)

    if n_workers > 1:
        track_id_map = process_map(run_mapping,
                                   all_t,
                                   max_workers=n_workers,
                                   chunksize=1,
                                   desc="Computing track ID map over time")
    else:
        track_id_map = []
        for t in tqdm(all_t, desc="Computing track ID map over time"):
            id_map = run_mapping(t)
            track_id_map.append(id_map)


    return np.stack(track_id_map, axis=0)


# def add_cell_patches_layer(viewer, verts, faces, track_id_map, tracks, quant_var):
#     """
#     Create Napari surface layer using cached track_id_map and given variable.
#     """
#     values = map_values_from_track_ids(track_id_map, tracks, quant_var)
#     viewer.add_surface((verts, faces, values),
#                        colormap="viridis",
#                        opacity=0.9,
#                        name=f"cell_patches_{quant_var}")

def _map_frame_vals(fi, track_id_map, track_to_val, n_verts):

    tids = track_id_map[fi]
    vals = np.zeros(n_verts, dtype=np.float32)
    for v in np.unique(tids[tids >= 0]):
        vals[tids == v] = track_to_val.get((fi, v), 0.0)
    return vals

def map_values_from_track_ids(track_id_map, tracks, quant_var):
    """
    Convert track_id_map → vertex values array for a given quantitative variable.
    Returns [n_frames × n_vertices] float array.
    """
    track_to_val = tracks.groupby(["t", "track_id"])[quant_var].mean().to_dict()
    n_frames, n_verts = track_id_map.shape


    run_mapping = partial(_map_frame_vals,
                          track_id_map=track_id_map,
                          track_to_val=track_to_val,
                          n_verts=n_verts)
    # if False: # par logic is actually slower for this
    #     values_list = process_map(run_mapping,
    #                               range(n_frames),
    #                               max_workers=n_workers,
    #                               chunksize=1,
    #                               desc="Mapping vertex values from track IDs")
    #     values = np.stack(values_list, axis=0)
    # else:
    values = np.zeros((n_frames, n_verts), dtype=np.float32)
    for fi in tqdm(range(n_frames), desc="Mapping vertex values from track IDs"):
        values[fi] = run_mapping(fi)

    return values



def make_surface_from_df(sphere_df, frame=None):
    """Return vertices, faces, vertex_values tuple for a Napari Surface layer."""
    if frame is not None and "t" in sphere_df.columns:
        sphere_df = sphere_df.query("t == @frame")

    verts = sphere_df[["x", "y", "z"]].to_numpy()
    hull = ConvexHull(verts)
    faces = hull.simplices
    values = np.zeros(len(verts), dtype=np.float32)
    return verts, faces, values

# =====================================================
# === Visualization Layers ============================
# =====================================================

def add_cell_patches_layer(viewer, verts, faces, track_id_map,
                           tracks, quant_var,
                           edge_var=None,
                           use_dual_layer=False,
                           edge_colormap="magma",
                           n_workers=4):
    """Add main surface layer(s) to Napari."""

    values = map_values_from_track_ids(track_id_map, tracks, quant_var)
    border_mask = compute_border_mask(faces, track_id_map)

    # if border_thickness > 1:
    #     border_mask = thicken_border_mask(border_mask, faces, steps=border_thickness - 1)

    # --- Case 1: Single-layer blended borders ---
    if not use_dual_layer:
        values_with_borders = values.copy()
        values_with_borders[border_mask > 0] = 0  # black borders
        values_with_borders = np.clip(values_with_borders, 0, 1)
        viewer.add_surface(
            (verts, faces, values_with_borders),
            colormap="viridis",
            opacity=0.9,
            name=f"cell_patches_{quant_var}",
        )

    # --- Case 2: Dual-layer (separate edge variable/color) ---
    else:
        # Base interior layer
        viewer.add_surface(
            (verts, faces, values),
            colormap="viridis",
            opacity=0.9,
            name=f"cell_patches_{quant_var}",
        )

        # Edge variable values (default to same as quant_var)
        edge_var = edge_var or quant_var
        edge_values = map_values_from_track_ids(track_id_map, tracks, edge_var)
        edge_overlay = np.zeros_like(edge_values)
        edge_overlay[border_mask > 0] = edge_values[border_mask > 0]
        viewer.add_surface(
            (verts, faces, edge_overlay),
            colormap=edge_colormap,
            opacity=0.7,
            name=f"borders_{edge_var}",
        )


def add_timeaware_points_layer(viewer, tracks, sphere_df, quant_var="quant_var", size=0.02):
    # Merge sphere geometry into tracks
    tracks = tracks.merge(
        sphere_df[["t", "cx", "cy", "cz", "radius"]],
        on="t", how="left"
    )

    # Project points to the sphere surface
    c_vecs = tracks[["x", "y", "z"]].to_numpy() - tracks[["cx", "cy", "cz"]].to_numpy()
    c_unit = c_vecs / np.linalg.norm(c_vecs, axis=1, keepdims=True)
    coords = c_unit * tracks["radius"].to_numpy()[:, None] + tracks[["cx", "cy", "cz"]].to_numpy()

    # Prepare Napari data (t, z, y, x)
    pts_data = np.column_stack([tracks["t"], coords[:, 0], coords[:, 1], coords[:, 2]])

    vals = tracks[quant_var].to_numpy()
    cmap = plt.get_cmap("plasma")
    normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    face_color = np.asarray(cmap(normed)[:, :4], dtype=np.float32)

    viewer.add_points(
        pts_data,
        size=size,
        face_color=face_color,
        name=f"points_{quant_var}",
        blending="translucent",
    )


# ---------------------------------------------------------------------
# --- Main viewer -----------------------------------------------------
# ---------------------------------------------------------------------
def launch_sphere_viewer(tracks,
                         sphere,
                         quant_var="quant_var",
                         edge_var=None,
                         show_points=False,
                         show_tracks=False,
                         cell_radius=12.5,
                         use_dual_layer=False,
                         edge_colormap="magma",
                         color_var_points=None,
                         sphere_subdiv=7,
                         n_workers=4):

    """Main Napari app for spherical cell visualization."""
    viewer = napari.Viewer(ndisplay=3)

    print("Generating sphere mesh...")
    verts, faces = make_dense_sphere_mesh(subdiv=sphere_subdiv)
    # verts, faces = make_intermediate_sphere(subdiv=8, keep_fraction=0.5)

    track_id_map = compute_track_id_map(
                                        tracks=tracks,
                                        sphere_df=sphere,
                                        base_verts=verts,
                                        cell_radius_max=cell_radius,
                                        n_workers=n_workers,
                                    )

    add_cell_patches_layer(viewer, verts, faces, track_id_map,
                           tracks, quant_var,
                           edge_var=edge_var,
                           use_dual_layer=use_dual_layer,
                           edge_colormap=edge_colormap,
                           n_workers=n_workers)

    # Optional points/tracks for context
    if show_points:
        if show_points:
            add_timeaware_points_layer(
                viewer,
                sphere_df=sphere,
                tracks=tracks,
                quant_var=color_var_points or quant_var,
                size=cell_radius*2,
            )

    if show_tracks:
        if "t" not in tracks.columns:
            raise ValueError("tracks must include a 't' column for Tracks layer")
        tracks_data = tracks[["track_id", "t", "z", "y", "x"]].to_numpy()
        viewer.add_tracks(tracks_data, tail_length=10, colormap="turbo", name="tracks")

    napari.run()
# ---------------------------------------------------------------------
# --- Example synthetic data -----------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    n = 500
    n_t = 15
    all_tracks = []
    for ti in range(n_t):
        phi = np.arccos(1 - 2 * np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        quant = np.random.rand(n)
        speed = np.random.rand(n)  # second variable
        df = pd.DataFrame({
            "track_id": np.arange(n),
            "t": ti,
            "x": x, "y": y, "z": z,
            "quant_var": quant,
            "velocity": speed
        })
        all_tracks.append(df)
    tracks = pd.concat(all_tracks, ignore_index=True)

    sphere_df = make_test_sphere_df(
        n_frames=n_t,
        base_radius=1.0,
        drift_vec=(0.002, 0.001, -0.0005),
        radius_growth=0.001
    )

    # ---- choose rendering mode ----
    launch_sphere_viewer(
        tracks,
        sphere=sphere_df,
        quant_var="quant_var",
        edge_var="velocity",
        use_dual_layer=True,
        show_points=True,
        edge_colormap="magma",
        n_workers=1,
        cell_radius=12.5 / 500,
    )