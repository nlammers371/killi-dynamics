import zarr
import napari
import os
import numpy as np

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project = "20250116_LCP1-NLSMSC_side2_cb"
zpath = os.path.join(root, "built_data", "zarr_image_files", project + ".zarr")

# load
zarr_image = zarr.open(zpath, mode="r")

# get scale info
scale_vec = tuple(zarr_image.attrs["pixel_res_um"])

# extract relevant frames
# last_5 = np.asarray(zarr0[-5:])

viewer = napari.Viewer()
viewer.add_image(zarr_image[1910:1920], channel_axis=1, scale=scale_vec)
napari.run()

print("wtf")
