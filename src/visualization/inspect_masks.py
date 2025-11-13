import pandas as pd
import zarr
import napari
import numpy as np
from pathlib import Path
from skimage.measure import regionprops
from matplotlib import cm
import json
import trimesh
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
root = Path(r"Y:\killi_dynamics")
project = "20251019_BC1-NLS_52-80hpf"
seg_type = "li_segmentation"

t_start, t_stop = 950, 1100
scale_vec = np.array([3.0, 0.85, 0.85])  # (Z,Y,X) µm/px

# ----------------------------
# LOAD SPHERE GEOMETRY
# ----------------------------
sphere_path = root / "surf_stats" / f"{project}_surf_stats.zarr" / "surf_fits" / "sphere_fits.csv"
sphere_df = pd.read_csv(sphere_path)

sphere_time_filter = (sphere_df["t"] >= t_start) & (sphere_df["t"] < t_stop)
avg_center = np.mean(
    sphere_df.loc[sphere_time_filter, ["center_x_smooth", "center_y_smooth", "center_z_smooth"]].values,
    axis=0,
)
avg_radius = np.mean(sphere_df.loc[sphere_time_filter, "radius_smooth"].values)

# ----------------------------
# LOAD MASK STACK
# ----------------------------
mpath = root / "segmentation" / seg_type / f"{project}_masks.zarr"
mask_clean = zarr.open(mpath, mode="r")["fused"]["clean"]
mask_p = mask_clean[t_start:t_stop]

# ----------------------------
# COMPUTE DISTANCES + BUILD COLORED VOLUME
# ----------------------------
# timepoints = np.arange(t_start, t_stop)
# colored_stack = np.zeros_like(mask_p, dtype=np.float32)
# distance_records = []
#
# for i, t in enumerate(tqdm(timepoints, desc="Computing distances")):
#     mask_t = mask_p[i]
#     if np.max(mask_t) == 0:
#         continue
#
#     # get frame-specific sphere fit
#     row = sphere_df.loc[sphere_df["t"] == t]
#     if not row.empty:
#         center = row[["center_z", "center_y", "center_x"]].values[0]
#         radius = row["radius"].values[0]
#     else:
#         center, radius = avg_center, avg_radius
#
#     # compute distances and paint efficiently
#     props = regionprops(mask_t, spacing=scale_vec)
#     for prop in props:
#         coords = prop.coords
#         centroid = np.array(prop.centroid)
#         dist = np.linalg.norm(centroid - center ) - radius
#         colored_stack[i, coords[:, 0], coords[:, 1], coords[:, 2]] = dist
#         distance_records.append((t, prop.label, *centroid, dist))
#
# distance_df = pd.DataFrame(
#     distance_records,
#     columns=["t", "label", "z_um", "y_um", "x_um", "dist_to_surface_um"],
# )
#
# print(f"Computed distances for {len(distance_df)} nuclei across {len(timepoints)} frames.")

# ----------------------------
# NORMALIZE + COLORIZE
# ----------------------------
# nonzero = colored_stack[colored_stack != 0]
# vmin, vmax = np.percentile(nonzero, [0.01, 99.9]) if nonzero.size > 0 else (0, 1)
# colored_norm = np.clip((colored_stack - vmin) / (vmax - vmin), 0, 1)
# cmap = cm.get_cmap("coolwarm")
# colored_rgb = (cmap(colored_norm)[..., :3] * 255).astype(np.uint8)

# ----------------------------
# DISPLAY
# ----------------------------
viewer = napari.Viewer(ndisplay=3)
viewer.add_labels(mask_p>0, name="masks", scale=scale_vec, opacity=0.9)
napari.run()
print("why?")