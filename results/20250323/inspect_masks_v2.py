import zarr
import napari
import numpy as np
from src.build_killi.run02_segment_nuclei import calculate_li_thresh
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
mpath_prev = os.path.join(root, "built_data", "mask_stacks", project + "_side1_mask_aff.zarr")
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")
# load
mask_o = zarr.open(mpath_prev, mode="r")
mask_full = zarr.open(mpath, mode="a")

for key in list(mask_o.attrs.keys()):
    mask_full.attrs[key] = mask_o.attrs[key]

# generate frame indices
t_range = np.arange(2100, 2110)


# get scale info
scale_vec = tuple([mask_full.attrs['PhysicalSizeZ'], mask_full.attrs['PhysicalSizeY'], mask_full.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
# im = np.squeeze(im_full[t_range, nucleus_channel])
mask = np.squeeze(mask_full[t_range])

viewer = napari.Viewer()

viewer.add_labels(mask, scale=scale_vec)

napari.run()

print("Check")

