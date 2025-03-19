import zarr
import napari
import os
import numpy as np

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project0 = "20250116_LCP1-NLSMSC_side1"
project1 = "20250116_LCP1-NLSMSC_side1_reset"
zpath0 = os.path.join(root, "built_data", "zarr_image_files", project0 + ".zarr")
zpath1 = os.path.join(root, "built_data", "zarr_image_files", project1 + ".zarr")

# load
zarr0 = zarr.open(zpath0, mode="r")
zarr1 = zarr.open(zpath1, mode="r")

# get scale info
scale_vec = tuple([zarr0.attrs['PhysicalSizeZ'], zarr0.attrs['PhysicalSizeY'], zarr0.attrs['PhysicalSizeX']])

# extract relevant frames
last_5 = np.asarray(zarr0[-5:])
first_5 = np.asarray(zarr1[:5])

viewer = napari.Viewer()
viewer.add_image(last_5, channel_axis=1, scale=scale_vec)
viewer.add_image(first_5, channel_axis=1, scale=scale_vec)

print("wtf")
