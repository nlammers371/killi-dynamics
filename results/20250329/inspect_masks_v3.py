import zarr
import napari
import numpy as np
from src.build_killi.build_utils import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh
import pandas as pd
import statsmodels.api as sm
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project = "20250311_LCP1-NLSMSC"
# project2 = "20250311_LCP1-NLSMSC_side2"
zpath = os.path.join(root, "built_data", "zarr_image_files", project + ".zarr")
mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_fused.zarr")
# mpath_prev = os.path.join(root, "built_data", "mask_stacks", project + "_side1_mask_aff.zarr")
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")
# load
# mask_o = zarr.open(mpath_prev, mode="r")
mask_full = zarr.open(mpath, mode="a")
#
# for key in list(mask_o.attrs.keys()):
#     mask_full.attrs[key] = mask_o.attrs[key]

# generate frame indices
# t_range = np.arange(2325, 2339)

t_int = 2200

# get scale info
scale_vec = tuple([mask_full.attrs['PhysicalSizeZ'], mask_full.attrs['PhysicalSizeY'], mask_full.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
# im = np.squeeze(im_full[t_range, nucleus_channel])
mask = np.squeeze(mask_full[t_int])

# get mask locations
props = regionprops(mask, spacing=scale_vec)
points = np.array([prop.centroid for prop in props])
label_vec = np.array([prop.label for prop in props])
area_vec = np.array([prop.area for prop in props])

# fit sphere and get SH info
coeffs, fitted_center, fitted_radius = fit_sphere_and_sh(points, L_max=15)
sphere_mesh = create_sphere_mesh(fitted_center, fitted_radius, resolution=100)

# get sh mesh
sh_mesh, r_sh = create_sh_mesh(coeffs, sphere_mesh)

from scipy.spatial import distance_matrix
surf_dist_mat = distance_matrix(points, sh_mesh[0])
surf_dist_vec = np.min(surf_dist_mat, axis=1)

outlier_filter = (surf_dist_vec > 30) & (area_vec < 500)
outlier_labels = label_vec[outlier_filter]  # threshold for outliers
outlier_mask = np.isin(mask, outlier_labels)  # create mask for outliers

viewer = napari.Viewer()

viewer.add_labels(mask, scale=scale_vec)
viewer.add_labels(outlier_mask, scale=scale_vec)

# viewer.add_surface((sphere_mesh[0], sphere_mesh[1], r_sh), name='My Surface', opacity=0.8, colormap='viridis')
viewer.add_surface((sh_mesh[0], sh_mesh[1], r_sh), name='My Surface', opacity=0.8, colormap='viridis')

napari.run()

print("Check")

