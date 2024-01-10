import napari
import os
from aicsimageio import AICSImage
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import dask.array as da
import glob2 as glob
from skimage.filters import threshold_otsu, threshold_multiotsu
import math


def cart_to_sphere(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2]) # for elevation angle defined from Z-axis down
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew

def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX), 4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residuals, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

if __name__ =="main":

    # # set parameters
    raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    file_prefix = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
    fullfile_name = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
    ds_factor = 4
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\kf_tracking\\built_data\\"
    project_name = "20230108"
    project_path = os.path.join(save_root, project_name, '')
    #
    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
    n_time_points = len(image_list)

    time_point = 10
    readPath = os.path.join(raw_data_root, file_prefix + f"({time_point}).czi")
    da_chunksize = (32, 188, 188)
    imObject = AICSImage(readPath)
    image_data = np.squeeze(imObject.data)

    image_dask = da.from_array(image_data, chunks=da_chunksize)

    scale_vec = np.asarray(imObject.physical_pixel_sizes)
    # viewer = napari.view_image(image_data, scale=tuple(scale_vec))

    # Use otsu's method to get binarized image
    thresh = threshold_otsu(image_data)
    binary = image_data > thresh*0.75
    thresh_dask = da.from_array(binary, chunks=da_chunksize)

    data_shape = image_data.shape
    zg, yg, xg = np.meshgrid(range(data_shape[0]), range(data_shape[1]), range(data_shape[2]), indexing="ij")
    zg = zg * scale_vec[0]
    yg = yg * scale_vec[1]
    xg = xg * scale_vec[2]

    zv = zg[binary == 1]
    yv = yg[binary == 1]
    xv = xg[binary == 1]

    r, xc, yc, zc = sphereFit(xv, yv, zv)
    # center_point = np.asarray([z[0], y[0], x[0]])
    r_array = np.sqrt((zg-zc)**2 + (yg-yc)**2 + (xg-xc)**2)

    surf_dist = np.abs(r_array-r)
    surf_bin = surf_dist <= 50
    surf_dask = da.from_array(surf_bin, chunks=da_chunksize)

    # # label_layer = viewer.add_labels(binary, name='segmentation', scale=tuple(scale_vec))
    block_dims = image_dask.blocks.shape
    chunk_key = np.empty(block_dims, dtype=bool)

    for k in range(block_dims[0]):
        for j in range(block_dims[1]):
            for i in range(block_dims[2]):
                thresh_chunk = surf_dask.blocks[k, j, i]
                chunk_key[k, j, i] = np.any(thresh_chunk > 0)