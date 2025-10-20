import zarr
import napari
import numpy as np
from pathlib import Path
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = Path(r"E:\pipeline_dev\killi_dynamics")
project = "MEM_NLS_test"
seg_type = "li_segmentation"

# mpath = root / "built_data" / "mask_stacks" / (project + "_mask_fused.zarr")
mpath = root / "built_data" / "mask_stacks" / seg_type / (project + "_masks.zarr")
mask = zarr.open(mpath, mode="r")
zpath = root / "built_data" / "zarr_image_files" / f"{project}.zarr"
im = zarr.open(zpath, mode="r")
# zarr2 = zarr.open(zpath2, mode="r")

# generate frame indices
t_start = 12
t_stop = 14
nucleus_channel = 1
frames = np.arange(t_start, t_stop)

# get scale info
scale_vec = tuple(im.attrs["voxel_size_um"])

# extract relevant frames
im_p = np.squeeze(im[t_start:t_stop, nucleus_channel])
mask_p = mask["clean"][t_start:t_stop]


viewer = napari.Viewer()
# viewer.add_image(data_full1, scale=scale_vec, colormap="gray", contrast_limits=[0, 2500])

viewer.add_image(im_p, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
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

