import napari
import os
from aicsimageio import AICSImage
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import zarr




# # set parameters
full_filepath = "D:\\Syd\\240416_NLS-Kikume_mRNAinjection_LightsheetTest\\E1_timelapse_2024_04_16__17_57_29_105(110).czi"
imObject = AICSImage(full_filepath)
scale_vec = tuple(np.asarray(imObject.physical_pixel_sizes))


#
# data_tzyx = zarr.open(image_zarr, mode='r')
# # label_tzyx = zarr.open(label_zarr, mode='r')
#
viewer = napari.view_image(np.squeeze(imObject.data), channel_axis=0, scale=scale_vec)

