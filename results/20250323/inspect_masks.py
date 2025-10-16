import zarr
import napari
import os
import numpy as np
from src.segmentation import calculate_li_thresh
import pandas as pd
import statsmodels.api as sm
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project = "20250311_LCP1-NLSMSC_side2"
# project2 = "20250311_LCP1-NLSMSC_side2"
zpath = os.path.join(root, "built_data", "zarr_image_files", project + ".zarr")
mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_stacks.zarr")
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")
# load
im_full = zarr.open(zpath, mode="r")
mask_full = zarr.open(mpath, mode="r")

# generate frame indices
t_int = 2100
nucleus_channel = 1

# get scale info
scale_vec = tuple([im_full.attrs['PhysicalSizeZ'], im_full.attrs['PhysicalSizeY'], im_full.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
im = np.squeeze(im_full[t_int, nucleus_channel])
mask = np.squeeze(mask_full[t_int])

li_thresh = 43.7
thresh2 = li_thresh * 0.75
thresh3 = li_thresh * 0.5
data_log_i, thresh_li = calculate_li_thresh(im, thresh_li=li_thresh)

viewer = napari.Viewer()

viewer.add_image(im, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
viewer.add_image(data_log_i, scale=scale_vec,  colormap="gray", contrast_limits=[0, 250])

viewer.add_labels(label(data_log_i > li_thresh), scale=scale_vec)
viewer.add_labels(label(data_log_i > thresh2), scale=scale_vec)
viewer.add_labels(label(data_log_i > thresh3), scale=scale_vec)

napari.run()

print("Check")

