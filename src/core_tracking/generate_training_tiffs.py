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

if __name__ == "__main__":

    np.random.seed(253)
    inflate_factor = 1.0
    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
    project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
    project_path = os.path.join(save_root, project_name, '')

    write_path = os.path.join(project_path, "cellpose", "")
    if not os.path.isdir(write_path):
        os.makedirs(write_path)

    n_samples = 50
    suffix_vec = ["_xy", "_zx", "_zy"]
    window_size = 512
    overwrite_flag = False
    skip_labeled_flag = False
    if overwrite_flag:
        skip_labeled_flag = False

    # ome_zarr_file_path = project_path + exp_name + ".ome.zarr"
    metadata_file_path = project_path + project_name + "_metadata.json"

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(project_path + project_name + "*t0001.tiff")))
    n_time_points = len(image_list)

    # load metadata
    # Opening JSON file
    f = open(metadata_file_path)

    # returns JSON object as
    # a dictionary
    metadata = json.load(f)

    pixel_res_raw = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
    anisotropy = pixel_res_raw[0] / pixel_res_raw[1]

    # Specify time index and pixel resolution
    print("Extracting training slices...")

    # Iterate through images
    for im, image_path in enumerate(tqdm(image_list)):

        im_name = path_leaf(image_path)
        im_name = im_name.replace(".tiff", "")

        # read the image data
        image_data_raw = io.imread(image_path)

        save_name = im_name #+ f"_w{well_index:03}_t{t:03}"

        if anisotropy != 1:
            dims_orig = image_data_raw.shape
            dims_new = np.round(
                [dims_orig[0] * anisotropy, dims_orig[1], dims_orig[2]]).astype(int)
            image_data = resize(image_data_raw, dims_new, preserve_range=True, order=1)
        else:
            image_data = image_data_raw

        if inflate_factor > 1:
            dims_curr = image_data.shape
            image_data = resize(image_data, tuple(np.asarray(dims_curr)*2), preserve_range=True)

        # generate rudimentary segmentation using otsu's method
        thresh = threshold_otsu(image_data)
        image_binary = image_data >= thresh

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

            slice_path = write_path + save_name + suffix + ".tiff"

            # rand_prefix = np.random.randint(0, 100000, 1)[0]
            # out_name = os.path.join(write_path,
            #                         f'_t01_{rand_prefix:06}' + '_' + save_name + suffix + f'{slice_num:03}')
            out_name = os.path.join(write_path, f'_t01_' + suffix + f'{slice_num:03}' + '_' + save_name)
            # print("Starting with raw label priors...")
            if slice_id == 0:
                im_slice = image_data[slice_num, :, :]
                # im_slice_bin = image_binary[slice_num, :, :]
            elif slice_id == 1:
                im_slice = np.squeeze(image_data[:, slice_num, :])
                # im_slice_bin = np.squeeze(image_binary[:, slice_num, :])
            else:  # slice_id == 2:
                im_slice = np.squeeze(image_data[:, :, slice_num])
                # im_slice_bin = np.squeeze(image_binary[:, :, slice_num])

            thresh = threshold_otsu(im_slice)
            im_slice_bin = im_slice >= thresh

            shape_full = np.asarray(im_slice.shape)
            im_lim = shape_full - window_size

            # find brightest regions
            xb = np.sum(im_slice_bin, axis=0) #np.sum(im_slice_bin[:, window_size:-window_size], axis=0)
            yb = np.sum(im_slice_bin, axis=1) #np.sum(im_slice_bin[window_size:-window_size, :], axis=1)

            if any(xb > 0):
                x_start = np.argmax(xb)
            else:
                x_start = np.random.choice(range(shape_full[1]), 1)[0].astype(int)

            if any(yb > 0):
                y_start = np.argmax(yb)
            else:
                y_start = np.random.choice(range(shape_full[0]), 1)[0].astype(int)

            y_ind = np.arange(np.max([y_start - window_size, 0]), np.min([y_start + window_size, shape_full[0]]))
            x_ind = np.arange(np.max([x_start - window_size, 0]), np.min([x_start + window_size, shape_full[1]]))
            im_slice_chunk = im_slice[y_ind[:, np.newaxis], np.reshape(x_ind, (1, len(x_ind)))]

            if np.max(im_slice_chunk < 1):
                im_slice_chunk = im_slice_chunk * (2 ** 16)
            im_slice_out = im_slice_chunk.astype(np.uint16)
            # write to file
            io.imsave(out_name + ".tiff", im_slice_out)

    print("Done.")
