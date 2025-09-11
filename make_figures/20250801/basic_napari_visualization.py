import napari
import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from src.utilities.image_utils import remove_background
import zarr



# raw_data_root = "D:\\Syd\\240611_EXP50_NLS-Kikume_24hpf_2sided_NuclearTracking\\" #"D:\\Syd\\240219_LCP1_67hpf_to_"
# Specify the path to the output OME-Zarr file and metadata file
image_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
side1_name = "20240611_NLS-Kikume_24hpf_side1"
side2_name = "20240611_NLS-Kikume_24hpf_side2"
experiment_date = "20240611"
start_i = 2157
stop_i = 2158
model_name = "LCP-Multiset-v1"
# # set parameters
# full_filepath = "D:\\Syd\\240416_NLS-Kikume_mRNAinjection_LightsheetTest\\E1_timelapse_2024_04_16__17_57_29_105(110).czi"
zarr_path = os.path.join(image_root, "built_data", "cellpose_output", model_name, side2_name, side2_name + '_probs.zarr')
mask_path = os.path.join(image_root, "built_data", "mask_stacks", model_name, side2_name, side2_name + '_mask_aff.zarr')
# imObject = AICSImage(full_filepath)
# scale_vec = tuple(np.asarray(imObject.physical_pixel_sizes))
image_data = zarr.open(zarr_path, mode="r")
mask_data = zarr.open(mask_path, mode="r")

# img_bkg = remove_background(np.squeeze(image_data[start_i:stop_i].copy()))


#
# data_tzyx = zarr.open(image_zarr, mode='r')
# # label_tzyx = zarr.open(label_zarr, mode='r')
#
viewer = napari.view_image(image_data[start_i:stop_i])
# viewer.add_image(img_bkg)
viewer.add_labels(mask_data[start_i:stop_i])

napari.run()

