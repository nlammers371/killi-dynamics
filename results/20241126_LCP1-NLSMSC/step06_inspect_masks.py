import zarr
import napari
import numpy as np
from pathlib import Path
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = Path("E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\")
project = "20241126_LCP1-NLSMSC"
# project2 = "20250311_LCP1-NLSMSC_side2"
# zpath = root / "built_data" / "zarr_image_files" / (project + ".zarr")

# mpath = root / "built_data" / "mask_stacks" / (project + "_mask_fused.zarr")
mpath = root / "built_data" / "mask_stacks" /( project + "_mask_aff.zarr")
# mpath_lcp = root / "built_data" / "mask_stacks" /( project + "_mask_aff_lcp.zarr")

# load
mask = zarr.open(mpath, mode="r")
# mask_lcp = zarr.open(mpath_lcp, mode="r")

# mask_full = zarr.open(mpath, mode="a")

# for key in list(mask_o.attrs.keys()):
#     mask_full.attrs[key] = mask_o.attrs[key]

# get filepaths

zpath = root / "built_data" / "zarr_image_files" / f"{project}.zarr"
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")

# load
im = zarr.open(zpath, mode="r")
# zarr2 = zarr.open(zpath2, mode="r")

# generate frame indices
t_start = 560
t_stop = 562
nucleus_channel = 1
frames = np.arange(t_start, t_stop)
#
mask_p = mask[t_start:t_stop]
# mask_lcp_p = mask_lcp[t_start:t_stop]
# mask_p[mask_lcp_p == 0] = 0

# get scale info
scale_vec = tuple([im.attrs['PhysicalSizeZ'], im.attrs['PhysicalSizeY'], im.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
im_p = np.squeeze(im[t_start:t_stop, nucleus_channel])
im_p2 = np.squeeze(im[t_start:t_stop, 0])



viewer = napari.Viewer()
# viewer.add_image(data_full1, scale=scale_vec, colormap="gray", contrast_limits=[0, 2500])

viewer.add_image(im_p, scale=scale_vec,  colormap="gray", contrast_limits=[0, 5000])
viewer.add_image(im_p2, scale=scale_vec,  colormap="magenta", contrast_limits=[100, 1000])
viewer.add_labels(mask_p, scale=scale_vec)

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

