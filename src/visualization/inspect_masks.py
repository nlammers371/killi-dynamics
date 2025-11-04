import zarr
import napari
import numpy as np
from pathlib import Path
import os
from src.data_io.zarr_io import open_experiment_array

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = Path(r"Y:\killi_dynamics")
project = "20251019_BC1-NLS_52-80hpf"
seg_type = "li_segmentation"

# mpath = root / "built_data" / "mask_stacks" / (project + "_mask_fused.zarr")
mpath = root / "segmentation" / seg_type / (project + "_masks.zarr")
m_store = zarr.open(mpath, mode="r")
mask_vf = m_store["fused"]["clean"]

im, _store_path, _resolved_side = open_experiment_array(root, project)
# zarr2 = zarr.open(zpath2, mode="r")

# generate frame indices
t_start = 1200
t_stop = 1201
nucleus_channel = 1
frames = np.arange(t_start, t_stop)

# get scale info
scale_vec = tuple(im.attrs["voxel_size_um"])

# extract relevant frames
im_p = np.squeeze(im[t_start:t_stop, nucleus_channel])
mask_p = mask_vf[t_start:t_stop]


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

