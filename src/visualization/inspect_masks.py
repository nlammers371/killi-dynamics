import pandas as pd
import zarr
import napari
import numpy as np
from pathlib import Path
import os
from src.data_io.zarr_io import open_experiment_array, open_mask_array
from src.geometry.spherical_harmonics import create_sh_mesh
import trimesh 
import json

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = Path(r"Y:\killi_dynamics")
project = "20251019_BC1-NLS_52-80hpf"
seg_type = "li_segmentation"

t_start = 960
t_stop = 965
nucleus_channel = 1

# load surf sphere
sphere_path = root / "surf_stats" / f"{project}_surf_stats.zarr" / "surf_fits" / "sphere_fits.csv"
sphere_df = pd.read_csv(sphere_path)

frames = np.arange(t_start, t_stop)
sphere_time_filter = (sphere_df["t"] >= t_start) & (sphere_df["t"] < t_stop)
avg_sphere_center = np.mean(sphere_df.loc[sphere_time_filter,  ["center_x_smooth", "center_y_smooth", "center_z_smooth"]].values, axis=0)
avf_sphere_radius = np.mean(sphere_df.loc[sphere_time_filter, "radius_smooth"].values)
mesh = trimesh.creation.icosphere(subdivisions=5, radius=avf_sphere_radius)
faces = mesh.faces
verts = mesh.vertices + avg_sphere_center
mesh.vertices = verts

# load SH fit
sh_path = root / "surf_stats" / f"{project}_surf_stats.zarr" / "surf_fits" / "surf_sh_coeffs.json"
with open(sh_path, "r") as f:
    sh_coeffs = json.load(f)

coeffs = np.array(sh_coeffs["960"])
vertices_sh, faces_sh, r_sh = create_sh_mesh(coeffs, sphere_mesh=(verts, faces))


# mpath = root / "built_data" / "mask_stacks" / (project + "_mask_fused.zarr")
mpath = root / "segmentation" / seg_type / (project + "_masks.zarr")
m_store = zarr.open(mpath, mode="r")
mask_clean = m_store["fused"]["clean_pre"]

# mask_raw, _, _ = open_mask_array(root, project, side="virtual_fused",
#                                  seg_type=seg_type, mask_field="stitched",
#                                  use_gpu=True,
#                                  verbose=False)
# mask_clean = m_store["fused"]["clean"]

# im, _store_path, _resolved_side = open_experiment_array(root, project)
# zarr2 = zarr.open(zpath2, mode="r")

# get scale info
scale_vec = tuple([3.0, 0.85, 0.85])

# extract relevant frames
# im_p = np.squeeze(im[t_start:t_stop, nucleus_channel])
mask_p = mask_clean[t_start:t_stop]
# mask_r_p = mask_raw[t_start:t_stop]


viewer = napari.Viewer()

viewer.add_surface(
        (verts, faces),
        name="embryo surface",
        colormap=None,
        opacity=0.95,
        shading="flat",
    )

# viewer.add_surface(
#         (vertices_sh, faces_sh, r_sh),
#         name="embryo surface",
#         colormap=None,
#         opacity=0.95,
#         shading="flat",
#     )
# viewer.add_image(data_full1, scale=scale_vec, colormap="gray", contrast_limits=[0, 2500])

# viewer.add_image(im_p, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
viewer.add_labels(mask_p, scale=scale_vec)
# viewer.add_labels(mask_r_p > 0, scale=scale_vec)

napari.run()
print("wtf")

#
# # generate frame indices
# t_range = np.arange(500, 510)
#
#
# # get scale info
# scale_vec = tuple([mask_full.attrs['PhysicalSizeZ'], mask_full.attrs['PhysicalSizeY'], mask_full.attrs['PhysicalSizeX']])
#
# print("Loading zarr files...")
# # extract relevant frames
# # im = np.squeeze(im_full[t_range, nucleus_channel])
# mask = np.squeeze(mask_full[t_range])
#
# viewer = napari.Viewer()
#
# viewer.add_labels(mask, scale=scale_vec)
#
# napari.run()

print("Check")

