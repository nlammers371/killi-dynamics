import zarr
from src.utilities.image_utils import calculate_LoG
import os
from src.utilities.functions import path_leaf
import json
import numpy as np
import napari
import skimage.io as io

project_name = "230425_EXP21_LCP1_D6_1pm_DextranStabWound"
# set read/write paths
root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
# path to zarr files
zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
t=0


save_directory = os.path.join(root, "built_data", "cellpose_output", "training_snips",  project_name, '')
if not os.path.isdir(save_directory):
    os.makedirs(save_directory)

im_name = path_leaf(zarr_path)
print("processing " + im_name)
# read the image data
data_tzyx = zarr.open(zarr_path, mode="r")
# n_wells = len(imObject.scenes)
# well_list = imObject.scenes
n_time_points = data_tzyx.shape[0]

# load metadata
metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
f = open(metadata_file_path)

# returns JSON object as
# a dictionary
metadata = json.load(f)

pixel_res_raw = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
metadata["ProbPhysicalSizeZ"] = pixel_res_raw[0]
metadata["ProbPhysicalSizeY"] = pixel_res_raw[1]
metadata["ProbPhysicalSizeX"] = pixel_res_raw[2]

anisotropy_raw = pixel_res_raw[0] / pixel_res_raw[1]


 # extract image
data_zyx_raw = data_tzyx[0]

dims_orig = data_zyx_raw.shape

dims_new = dims_orig
data_zyx = data_zyx_raw.copy()

im_log = data_zyx#, im_bkg = calculate_LoG(data_zyx=data_zyx, scale_vec=pixel_res_raw)


slice0 = im_log[292, :, :]
io.imsave(os.path.join(save_directory, "slice_xy292.tiff"), slice0, check_contrast=False)

slice1 = im_log[242, :, :]
io.imsave(os.path.join(save_directory, "slice_xy242.tiff"), slice1, check_contrast=False)

slice2 = np.squeeze(im_log[:, :, 170])
io.imsave(os.path.join(save_directory, "slice_yz170.tiff"), slice2, check_contrast=False)