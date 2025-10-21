# import dask
import numpy as np
import json
import os
import glob2 as glob
from aicsimageio import AICSImage
from skimage.transform import resize
from tqdm import tqdm
from skimage import io
from src.utilities.functions import path_leaf
from skimage.filters import threshold_otsu
from src._Archive.utilities.image_utils import calculate_LoG
import zarr
import napari

def make_training_slices(root, project_name, n_samples=2, make_LoG_samples=False):

    write_path = os.path.join(root, "cellpose", "training_snips", project_name, "")
    if not os.path.isdir(write_path):
        os.makedirs(write_path)

    if make_LoG_samples:
        path_log = os.path.join(write_path, 'log')
        path_bkg = os.path.join(write_path, 'bkg')
        if not os.path.isdir(path_log):
            os.makedirs(path_log)
        if not os.path.isdir(path_bkg):
            os.makedirs(path_bkg)

    # n_samples = 50
    suffix_vec = ["_xy", "_zx", "_zy"]

    # specify time points to load
    zarr_path = os.path.join(root, "zarr_image_files", project_name + ".zarr")
    data_zarr = zarr.open(zarr_path, mode="r")

    pixel_res_raw = np.asarray([data_zarr.attrs["PhysicalSizeZ"], data_zarr.attrs["PhysicalSizeY"], data_zarr.attrs["PhysicalSizeX"]])
    anisotropy = pixel_res_raw[0] / pixel_res_raw[1]

    # image_list = sorted(glob.glob(os.path.join(project_path + project_name + "*t0001.tiff")))
    n_time_points = data_zarr.shape[0]

    # Iterate through images
    for im in tqdm(range(n_time_points), "Extracting training slices..."):

        # read the image data
        image_data_raw = data_zarr[im]

        if anisotropy != 1:
            dims_orig = image_data_raw.shape
            dims_new = np.round(
                [dims_orig[0] * anisotropy, dims_orig[1], dims_orig[2]]).astype(int)
            image_data = resize(image_data_raw, dims_new, preserve_range=True, order=1)
        else:
            image_data = image_data_raw

        if inflate_factor > 1:
            dims_curr = image_data.shape
            image_data = resize(image_data, tuple(np.asarray(dims_curr) * 2), preserve_range=True)

        if make_LoG_samples:
            im_log, im_bkg = calculate_LoG(data_zyx=image_data_raw, scale_vec=pixel_res_raw)

        # randomly choose slices along each direction
        dim_vec = image_data.shape
        xy_slice_indices = np.random.choice(range(dim_vec[0]), n_samples, replace=False)
        xy_id_arr = np.zeros(xy_slice_indices.shape)

        if np.random.rand() > 0.5:
            zx_slice_indices = np.random.choice(range(dim_vec[1]), n_samples, replace=False)
            zy_slice_indices = np.asarray([])
        else:
            zy_slice_indices = np.random.choice(range(dim_vec[1]), n_samples, replace=False)
            zx_slice_indices = np.asarray([])
        zx_id_arr = np.ones(zx_slice_indices.shape)
        zy_id_arr = np.ones(zy_slice_indices.shape)

        # combine and shuffle
        slice_num_vec = np.concatenate((xy_slice_indices, zx_slice_indices, zy_slice_indices), axis=0)
        slice_id_vec = np.concatenate((xy_id_arr, zx_id_arr, zy_id_arr), axis=0)
        shuffle_vec = np.random.choice(range(len(slice_id_vec)), len(slice_id_vec), replace=False)
        slice_num_vec = slice_num_vec[shuffle_vec].astype(int)
        slice_id_vec = slice_id_vec[shuffle_vec].astype(int)

        for image_i in range(len(slice_id_vec)):

            # generate save paths for image slice and labels
            slice_id = slice_id_vec[image_i]
            slice_num = slice_num_vec[image_i]
            suffix = suffix_vec[slice_id]


            if not make_LoG_samples:
                out_name = os.path.join(write_path,  f'im_slice' + suffix + f'_t{im:04}_slice{slice_num:04}' + '_' + project_name)
            else:
                out_name_log = os.path.join(write_path, 'LoG',
                                        f'im_slice_' + suffix + f'_t{im:04}_slice{slice_num:04}' + '_' + project_name)
                out_name_bkg = os.path.join(write_path, 'bkg',
                                            f'im_slice_' + suffix + f'_t{im:04}_slice{slice_num:04}' + '_' + project_name)

            # print("Starting with raw label priors...")
            if not make_LoG_samples:
                if slice_id == 0:
                    im_slice = image_data[slice_num, :, :]
                    # im_slice_bin = image_binary[slice_num, :, :]
                elif slice_id == 1:
                    im_slice = np.squeeze(image_data[:, slice_num, :])
                    # im_slice_bin = np.squeeze(image_binary[:, slice_num, :])
                else:  # slice_id == 2:
                    im_slice = np.squeeze(image_data[:, :, slice_num])
                    # im_slice_bin = np.squeeze(image_binary[:, :, slice_num])
            else:
                if slice_id == 0:
                    im_slice_log = im_log[slice_num, :, :]
                    im_slice_bkg = im_bkg[slice_num, :, :]
                elif slice_id == 1:
                    im_slice_log = np.squeeze(im_log[:, slice_num, :])
                    im_slice_bkg = np.squeeze(im_bkg[:, slice_num, :])
                else:  # slice_id == 2:
                    im_slice_log = np.squeeze(im_log[:, :, slice_num])
                    im_slice_bkg = np.squeeze(im_bkg[:, :, slice_num])


            # write to file
            if not make_LoG_samples:
                io.imsave(out_name + ".tiff", im_slice, check_contrast=False)
            else:
                io.imsave(out_name_log + ".tiff", im_slice_log, check_contrast=False)
                io.imsave(out_name_bkg + ".tiff", im_slice_bkg, check_contrast=False)


    print("Done.")

if __name__ == "__main__":

    np.random.seed(253)
    inflate_factor = 1.0

    # Specify the path to the output OME-Zarr file and metadata file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
    project_name = "20240611_NLS-Kikume_24hpf_side2" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
    # project_path = os.path.join(save_root, project_name)

    make_training_slices(root, project_name, n_samples=2, make_LoG_samples=True)


