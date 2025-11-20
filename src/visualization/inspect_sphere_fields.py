"""
Correct + aesthetic visualization of HEALPix scalar fields in napari.

- Samples HEALPix values onto a smooth icosphere
- No displacement, only color
- Uses astropy-healpix for accurate pixel lookup
- Uses trimesh for a beautiful sphere mesh
"""

import numpy as np
import zarr
import napari
import trimesh
from pathlib import Path
from astropy_healpix import HEALPix
from astropy import units as u


# ---------------------------------------------------------------------
# 1. Load HEALPix field (t, npix)
# ---------------------------------------------------------------------
def load_field(store_path, group, dataset):
    store = zarr.open(store_path, mode="r")
    arr = np.asarray(store[group][dataset])

    if arr.ndim != 2:
        raise ValueError(f"Expected (t, pix) array, got {arr.shape}")

    return arr


# ---------------------------------------------------------------------
# 2. Beautiful smooth sphere from trimesh
# ---------------------------------------------------------------------
def make_sphere_mesh(subdivisions=5, radius=500.0):
    """
    Returns:
        verts: (Nverts, 3)
        faces: (Nfaces, 3)
    """
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int32)


# ---------------------------------------------------------------------
# 3. Correct HEALPix → vertex sampling
# ---------------------------------------------------------------------
def sample_healpix(values, verts, nside):
    """
    Given HEALPix pixel values (length npix) and sphere vertices (Nx3),
    return per-vertex values by nearest-pixel lookup.

    This is the *correct* way to paint HEALPix data on arbitrary meshes.
    """

    # Compute lon/lat of each sphere vertex
    x, y, z = verts.T
    r = np.linalg.norm(verts, axis=1)

    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arcsin(z / r))

    hp = HEALPix(nside=nside, order="ring")

    # Critical: use astropy Quantities to avoid TrackedArray issues
    pix_idx = hp.lonlat_to_healpix(lon * u.deg, lat * u.deg)

    return values[pix_idx]


# ---------------------------------------------------------------------
# 4. Main visualization function
# ---------------------------------------------------------------------
def visualize_field(
    store_path: str | Path,
    fields: list[tuple[str, str]],     # <-- NEW: list of (group, dataset)
    nside: int,
    rolling_window: int | None = None,
    start_index: int = 0,
    stop_index: int = None,
    sphere_radius: float = 500.0,
    subdivisions: int = 5,
):
    """
    Display one or more HEALPix scalar fields as multiple surface layers.
    Everything else remains exactly how you originally wrote it.
    """

    # Load first field to get shape + time window
    group0, dataset0 = fields[0]
    field0 = load_field(store_path, group0, dataset0)

    if stop_index is None:
        stop_index = field0.shape[0]

    # T = stop_index - start_index

    # Build smooth sphere ONCE
    verts, faces = make_sphere_mesh(
        subdivisions=subdivisions,
        radius=sphere_radius
    )

    viewer = napari.Viewer()

    # ------------------------------------------------------------
    # Loop over each requested (group, dataset)
    # ------------------------------------------------------------
    for group, dataset in fields:

        field = load_field(store_path, group, dataset)
        if rolling_window is None:
            # original behavior
            field = np.nanmean(field[:-1], axis=0)
        else:
            # rolling average across time dimension
            T, P = field.shape
            k = rolling_window
            field_roll = np.full((T, P), np.nan, dtype=field.dtype)

            for t in range(T):
                start = max(0, t - k + 1)
                end = t + 1
                field_roll[t] = np.nanmean(field[start:end], axis=0)

            field = field_roll
            # Precompute per-vertex sampled values: (T, Nverts)
        vtx_vals_ts = np.empty((T, verts.shape[0]), dtype=np.float32)
        for i, t in enumerate(range(start_index, stop_index)):
            vtx_vals_ts[i] = sample_healpix(field[t], verts, nside)
        # vtx_vals_ts = sample_healpix(field, verts, nside)
        # Add a surface layer for this dataset
        viewer.add_surface(
            (verts, faces, vtx_vals_ts),
            colormap="viridis",
            name=f"{group}/{dataset}",
        )

        print(f"Loaded field: {group}/{dataset}, nside={nside}")

    print(f"Mesh: verts={verts.shape}, faces={faces.shape}")

    napari.run()



# ---------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    store = (
        r"Y:\killi_dynamics\cell_field_dynamics\20251019_BC1-NLS_52-80hpf"
        r"\tracking_20251102\fields_nside0008.zarr"
    )
    visualize_field(
        store_path=store,
        fields=[
            ("density", "field"),
            ("metrics", "path_speed"),
            ("metrics", "diffusivity_total"),
        ],
        nside=8,
        rolling_window=7,
        sphere_radius=500.0,
        subdivisions=6,
    )
