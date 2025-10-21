import numpy as np
import glob2 as glob
import os
import skimage.io as io
import nd2
from skimage.morphology import label
from skimage.measure import regionprops
import zarr
from skimage.transform import resize
import skimage as ski
import SimpleITK as sitk
import networkx as nx


def remove_background(data_zyx): #, scale_vec, make_isotropic=False):
    # estimate background using blur
    # top1 = np.percentile(data_zyx, 99.9)
    # data_capped = data_zyx.copy()
    # data_capped[data_capped > top1] = top1
    # data_capped = data_capped[:, 500:775, 130:475]

    # shape_orig = np.asarray(data_zyx.shape)
    # shape_iso = shape_orig.copy()
    # iso_factor = scale_vec[0] / scale_vec[1]
    # shape_iso[0] = shape_iso[0] * iso_factor

    gaussian_background = ski.filters.gaussian(data_zyx, sigma=(5, 5, 5)) # NL this should be set dynamically based on image scale
    data_bkg = data_zyx - gaussian_background

    # data_rs = data_zyx.copy() #resize(data_1, shape_iso, preserve_range=True, order=1)
    # image = sitk.GetImageFromArray(data_rs)
    # data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(image, sigma=1.5))
    # if not make_isotropic:
    #     data_log_i = resize(ski.util.invert(data_log), shape_orig, preserve_range=True, order=1)
    #     # data_log_i = ski.util.invert(data_log_i)
    # else:
    #     data_log_i = ski.util.invert(data_log)

    # rescale and convert to 16 bit
    # if make_isotropic:
    #     data_bkg_16 = data_rs.copy()
    # else:

    # data_bkg_16 = data_1.copy()
    # data_bkg_16 = data_bkg_16 - np.min(data_bkg_16)
    # data_bkg_16 = np.round(data_bkg_16 / np.max(data_bkg_16) * 2 ** 16 - 1).astype(np.uint16)

    # log_i_16 = data_log_i.copy()
    # log_i_16 = log_i_16 - np.min(log_i_16)
    # log_i_16 = np.round(log_i_16 / np.max(log_i_16) * 2 ** 16 - 1).astype(np.uint16)

    return data_bkg

# function to process image stacks to account for uneven intensities and get lap-of-gaussian
def calculate_LoG(data_zyx, scale_vec, make_isotropic=False):
    # estimate background using blur
    top1 = np.percentile(data_zyx, 99)
    data_capped = data_zyx.copy()
    # data_capped[data_capped > top1] = top1
    # data_capped = data_capped[:, 500:775, 130:475]

    shape_orig = np.asarray(data_capped.shape)
    shape_iso = shape_orig.copy()
    iso_factor = scale_vec[0] / scale_vec[1]
    shape_iso[0] = shape_iso[0] * iso_factor

    gaussian_background = ski.filters.gaussian(data_capped, sigma=(3, 3, 3)) # NL this should be set dynamically based on image scale
    data_1 = np.divide(data_capped, gaussian_background)

    data_rs = resize(data_1, shape_iso, preserve_range=True, order=1)
    image = sitk.GetImageFromArray(data_rs)
    data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(image, sigma=1.5))
    if not make_isotropic:
        data_log_i = resize(ski.util.invert(data_log), shape_orig, preserve_range=True, order=1)
        # data_log_i = ski.util.invert(data_log_i)
    else:
        data_log_i = ski.util.invert(data_log)

    # rescale and convert to 16 bit
    if make_isotropic:
        data_bkg_16 = data_rs.copy()
    else:
        data_bkg_16 = data_1.copy()
    data_bkg_16 = data_bkg_16 - np.min(data_bkg_16)
    data_bkg_16 = np.round(data_bkg_16 / np.max(data_bkg_16) * 2 ** 16 - 1).astype(np.uint16)

    log_i_16 = data_log_i.copy()
    log_i_16 = log_i_16 - np.min(log_i_16)
    log_i_16 = np.round(log_i_16 / np.max(log_i_16) * 2 ** 16 - 1).astype(np.uint16)

    return log_i_16, data_bkg_16

def process_raw_image(data_zyx, scale_vec):
    # estimate background using blur
    top1 = np.percentile(data_zyx, 99)
    data_capped = data_zyx.copy()
    data_capped[data_capped > top1] = top1
    # data_capped = data_capped[:, 500:775, 130:475]

    shape_orig = np.asarray(data_capped.shape)
    shape_iso = shape_orig.copy()
    iso_factor = scale_vec[0] / scale_vec[1]
    shape_iso[0] = shape_iso[0] * iso_factor

    gaussian_background = ski.filters.gaussian(data_capped, sigma=(2, 8, 8))
    data_1 = np.divide(data_capped, gaussian_background)
    # data_sobel = ski.filters.sobel(data_1)
    # data_sobel_i = ski.util.invert(data_sobel)

    data_rs = resize(data_1, shape_iso, preserve_range=True, order=1)
    image = sitk.GetImageFromArray(data_rs)
    data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(image, sigma=1))
    data_log_i = resize(ski.util.invert(data_log), shape_orig, preserve_range=True, order=1)
    # data_log_i = ski.util.invert(data_log)

    # rescale and convert to 16 bit
    data_bkg_16 = data_1.copy()
    data_bkg_16 = data_bkg_16 - np.min(data_bkg_16)
    data_bkg_16 = np.round(data_bkg_16 / np.max(data_bkg_16) * 2 ** 16 - 1).astype(np.uint16)

    log_i_16 = data_log_i.copy()
    log_i_16 = log_i_16 - np.min(log_i_16)
    log_i_16 = np.round(log_i_16 / np.max(log_i_16) * 2 ** 16 - 1).astype(np.uint16)

    return log_i_16, data_bkg_16