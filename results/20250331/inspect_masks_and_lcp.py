import zarr
import napari
import numpy as np
# from src.build_killi.build_utils import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh, fuse_images
import pandas as pd
# from src.segmentation import calculate_li_thresh
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

t_int = 2201
# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
d_root = "D:\\Nick\\killi_tracker\\"
project_name = "20250311_LCP1-NLSMSC"
# project2 = "20250311_LCP1-NLSMSC_side2"

# load mask dataset
mpath = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
mask_full = zarr.open(mpath, mode="a")

# load tracking mask
tracking_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\tracking\\20250311_LCP1-NLSMSC\\tracking_20250328_da\\well0000\\track_2100_2339\\"
seg_zarr = zarr.open(os.path.join(tracking_path, "segments.zarr"), mode="r")

# load images
zpath = os.path.join(d_root, "built_data", "zarr_image_files", project_name + "_fused.zarr")

fused_image = zarr.open(zpath, mode="r")
im0 = np.squeeze(fused_image[t_int, 0])
im1 = np.squeeze(fused_image[t_int, 1])

# load fluorescence data
fluo_df = pd.read_csv(f"E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\fluorescence_data\\20250311_LCP1-NLSMSC\\fluorescence_data_frame_{t_int:04}.csv")

# get scale info
scale_vec = tuple([mask_full.attrs['PhysicalSizeZ'], mask_full.attrs['PhysicalSizeY'], mask_full.attrs['PhysicalSizeX']])

# extract relevant frames
mask = np.squeeze(mask_full[t_int])
seg_mask = np.squeeze(seg_zarr[t_int-2100])
# det_mask = np.squeeze(det_zarr[t_int])

viewer = napari.Viewer()

viewer.add_labels(mask, scale=scale_vec)
viewer.add_labels(seg_mask, scale=scale_vec)
# viewer.add_labels(det_mask, scale=scale_vec)
viewer.add_image(im0, name="lcp1", scale=scale_vec)
viewer.add_image(im1, name="nls", scale=scale_vec)
# viewer.add_image(data_log_i, name="nls-LoG", scale=scale_vec)

# # viewer.add_surface((sphere_mesh[0], sphere_mesh[1], r_sh), name='My Surface', opacity=0.8, colormap='viridis')
# viewer.add_surface((sh_mesh[0], sh_mesh[1], r_sh), name='My Surface', opacity=0.8, colormap='viridis')

napari.run()

print("Check")

