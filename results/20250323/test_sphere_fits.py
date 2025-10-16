import zarr
import napari
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.registration import phase_cross_correlation
from src.build_killi.process_masks import fit_sphere

import os
os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project = "20250419_BC1-NLSMSC"  # "20250311_LCP1-NLSMSC"  #  "20240611_NLS-Kikume_24hpf_side2"
# project2 = "20250311_LCP1-NLSMSC_side2"
zpath = os.path.join(root, "built_data", "zarr_image_files", project + ".zarr")
mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_clean.zarr")
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")
# load
im_full = zarr.open(zpath, mode="r")
mask_full = zarr.open(mpath, mode="r")

scale_vec = tuple([im_full.attrs['PhysicalSizeZ'], im_full.attrs['PhysicalSizeY'], im_full.attrs['PhysicalSizeX']])

frame_i = 600
nucleus_channel = 1
im = im_full[frame_i, nucleus_channel]
mask = mask_full[frame_i]

viewer = napari.Viewer()

viewer.add_image(im, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
viewer.add_labels(mask, scale=scale_vec)

napari.run()

# viewer = napari.Viewer()
#
# viewer.add_image(im, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
# viewer.add_labels(mask, scale=scale_vec)
# viewer.add_labels(mask01, scale=scale_vec)
# viewer.add_labels(shadow_mask, scale=scale_vec)
# napari.run()

