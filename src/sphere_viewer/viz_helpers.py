import numpy as np
import pandas as pd
from scipy.spatial import SphericalVoronoi


def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def compute_spherical_voronoi(track_df, radius=1.0):
    """Compute spherical Voronoi polygons from track xyz positions."""
    points = normalize(track_df[["x", "y", "z"]].to_numpy()) * radius
    sv = SphericalVoronoi(points, radius=radius)
    sv.sort_vertices_of_regions()
    polygons = [sv.vertices[region] for region in sv.regions]
    return polygons


def make_surface_from_df(sphere_df, frame=None):
    """Return vertices, faces, vertex_values tuple for Napari Surface layer."""
    # Assume sphere_df gives a dense mesh already (x,y,z per vertex)
    if frame is not None and "t" in sphere_df.columns:
        sphere_df = sphere_df.query("t == @frame")
    verts = sphere_df[["x", "y", "z"]].to_numpy()
    # Make a simple triangulation using convex hull or marching cubes if desired
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    faces = hull.simplices
    values = np.zeros(len(verts))
    return verts, faces, values
