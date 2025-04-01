import zarr
import napari
import numpy as np
from src.build_killi.build_utils import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh, fuse_images
import pandas as pd
import statsmodels.api as sm
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

t_int = 2200
# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "20250311_LCP1-NLSMSC"
# project2 = "20250311_LCP1-NLSMSC_side2"

# load mask dataset
mpath = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
mask_full = zarr.open(mpath, mode="a")

# load tracking mask
tracking_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\tracking\\20250311_LCP1-NLSMSC\\tracking_20250328_redux\\well0000\\track_0000_2339_cb\\"
seg_zarr = zarr.open(os.path.join(tracking_path, "segments.zarr"), mode="r")
tracks_df = pd.read_csv(os.path.join(tracking_path, "tracks_fluo.csv"))

# load and fuse images
metadata_path = os.path.join(root, "metadata", project_name + "_side1", "")
half_shift_df = pd.read_csv(os.path.join(metadata_path, project_name + "_side2" + "_to_" + project_name + "_side1" + "_shift_df.csv"))

# generate shift arrays
side2_shifts = half_shift_df.copy() #+ time_shift_df.copy()
side1_shifts = half_shift_df.copy()
side1_shifts[["zs", "ys", "xs"]] = 0  # no shift for side1

# load images
print("Generating fused image...")
zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
image_zarr_path1 = os.path.join(root, "built_data", "zarr_image_files", project_name + "_side1.zarr")
image_zarr_path2 = os.path.join(root, "built_data", "zarr_image_files", project_name + "_side2.zarr")
image_zarr1 = zarr.open(image_zarr_path1, mode='r')
image_zarr2 = zarr.open(image_zarr_path2, mode='r')

# load shifts
# fused_image = np.zeros(image_zarr1.shape[1:], dtype=np.float32)
im0 = fuse_images(t_int, image_zarr1, image_zarr2, side1_shifts, side2_shifts, fuse_channel=0)
im1 = fuse_images(t_int, image_zarr1, image_zarr2, side1_shifts, side2_shifts, fuse_channel=1)

# fused_image[0] = im0
# fused_image[1] = im1
# load fluorescence data
fluo_df = pd.read_csv(f"E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\fluorescence_data\\20250311_LCP1-NLSMSC\\fluorescence_data_frame_{t_int:04}.csv")

# get scale info
scale_vec = tuple([mask_full.attrs['PhysicalSizeZ'], mask_full.attrs['PhysicalSizeY'], mask_full.attrs['PhysicalSizeX']])

# extract relevant frames
mask = np.squeeze(mask_full[t_int])
seg_mask = np.squeeze(seg_zarr[t_int])

viewer = napari.Viewer()

viewer.add_labels(mask, scale=scale_vec)
viewer.add_labels(seg_mask, scale=scale_vec)
viewer.add_image(im0, name="lcp1", scale=scale_vec)
viewer.add_image(im1, name="nls", scale=scale_vec)

# # viewer.add_surface((sphere_mesh[0], sphere_mesh[1], r_sh), name='My Surface', opacity=0.8, colormap='viridis')
# viewer.add_surface((sh_mesh[0], sh_mesh[1], r_sh), name='My Surface', opacity=0.8, colormap='viridis')

napari.run()

print("Check")

