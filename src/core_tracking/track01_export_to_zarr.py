import numpy as np
import json
import os
import glob2 as glob
from aicsimageio import AICSImage
from skimage.transform import resize
from tqdm import tqdm
from skimage import io
from src.utilities.functions import path_leaf
from tqdm.contrib.concurrent import process_map
from functools import partial
import zarr
import dask
# def create_dummy_data(shape):
#     return da.random.random(shape, chunks=(100, 100, 100))

def initialize_zarr_store(zarr_path, image_list, overwrite_flag=False):

    # im_raw_sample = io.imread(image_list[0])
    # Load image
    imObject = AICSImage(image_list[0])
    image_data = np.squeeze(imObject.data)
    raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)
    if np.max(raw_scale_vec) <= 1e-5:
        raw_scale_vec = raw_scale_vec * 1e6
    dims_orig = image_data.shape
    rs_factors = np.divide(raw_scale_vec, resampling_scale)
    dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)
    # image_data_rs = np.round(resize(image_data, dims_new, preserve_range=True, order=1)).astype(np.uint16)
    shape = tuple(dims_new)
    dtype = np.uint16

    if overwrite_flag:
        zarr_file = zarr.open(zarr_path, mode='w', shape=(len(image_list),) + shape, dtype=dtype, chunks=(1,) + shape)
    else:
        zarr_file = zarr.open(zarr_path, mode='a', shape=(len(image_list),) + shape, dtype=dtype, chunks=(1,) + shape)

    return zarr_file

def write_image(t, image_list, project_path, overwrite_flag, metadata_file_path, file_prefix, tres, resampling_scale):

    f_string = path_leaf(image_list[t])
    time_string = f_string.replace(file_prefix, "")
    time_string = time_string.replace(".czi", "")
    time_point = int(time_string[1:-1]) - 1
    # readPath = os.path.join(raw_data_root, file_prefix + f"({time_point}).czi")
    project_name = path_leaf(project_path)
    save_path = project_path + project_name + f"_t{time_point:04}.tiff"

    if (not os.path.exists(save_path)) | overwrite_flag:
        # Load image
        imObject = AICSImage(image_list[t])
        image_data = np.squeeze(imObject.data)
        raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)

        # Resize
        dims_orig = image_data.shape
        rs_factors = np.divide(raw_scale_vec, resampling_scale)
        dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)
        image_data_rs = np.round(resize(image_data, dims_new, preserve_range=True, order=1)).astype(np.uint16)

        # Export the Dask array to the OME-Zarr file
        if t == 0:
            n_time_points = len(image_list)
            metadata = {
                "DimOrder": "tzyx",
                "Dims": [n_time_points, image_data_rs.shape[0], image_data_rs.shape[1], image_data_rs.shape[2]],
                # Assuming there is only one timepoint per array, adjust if needed
                "TimeRes": tres,
                "PhysicalSizeX": resampling_scale[2],
                "PhysicalSizeY": resampling_scale[1],
                "PhysicalSizeZ": resampling_scale[0],
                "ProjectName": project_name
            }

            # Save metadata to a JSON file
            with open(metadata_file_path, 'w') as json_file:
                json.dump(metadata, json_file)

            # save actual image array

        io.imsave(save_path, image_data_rs)


def write_zarr(t, zarr_file, image_list, overwrite_flag, metadata_file_path, file_prefix, tres, resampling_scale,
               metadata_only_flag=False):

    f_string = path_leaf(image_list[t])
    time_string = f_string.replace(file_prefix, "")
    time_string = time_string.replace(".czi", "")
    time_point = int(time_string[1:-1]) - 1
    # readPath = os.path.join(raw_data_root, file_prefix + f"({time_point}).czi")

    if (not np.any(zarr_file[t] > 0)) | overwrite_flag:
        # Load image
        imObject = AICSImage(image_list[t])
        image_data = np.squeeze(imObject.data)
        raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)
        if np.max(raw_scale_vec) <= 1e-5:
            raw_scale_vec = raw_scale_vec * 1e6
        # Resize
        dims_orig = image_data.shape
        rs_factors = np.divide(raw_scale_vec, resampling_scale)
        dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)
        image_data_rs = np.round(resize(image_data, dims_new, preserve_range=True, order=1)).astype(np.uint16)

        # Export the Dask array to the OME-Zarr file
        if t == 0:
            n_time_points = len(image_list)
            project_name = path_leaf(image_list[t])
            project_name = project_name.replace(".czi", "")
            metadata = {
                "DimOrder": "tzyx",
                "Dims": [n_time_points, image_data_rs.shape[0], image_data_rs.shape[1], image_data_rs.shape[2]],
                # Assuming there is only one timepoint per array, adjust if needed
                "TimeRes": tres,
                "PhysicalSizeX": resampling_scale[2],
                "PhysicalSizeY": resampling_scale[1],
                "PhysicalSizeZ": resampling_scale[0],
                "ProjectName": project_name
            }

            # Save metadata to a JSON file
            with open(metadata_file_path, 'w') as json_file:
                json.dump(metadata, json_file)

            # save actual image array

        zarr_file[time_point] = image_data_rs


def export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, par_flag=True, overwrite_flag=False,
                       resampling_scale=None, n_workers=6, metadata_only_flag=False):

    if resampling_scale is None:
        resampling_scale = np.asarray([1.5, 1.5, 1.5])

    zarr_path = os.path.join(save_root, "built_data", "zarr_image_files", project_name + '.zarr')

    if not os.path.isdir(zarr_path):
        os.makedirs(zarr_path)
    # ome_zarr_file_path = project_path + exp_name + ".ome.zarr"
    metadata_dir = os.path.join(save_root, "metadata", project_name, "")
    if not os.path.isdir(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_file_path = os.path.join(metadata_dir,  "metadata.json")

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))

    # open first image file to get stats

    # Resize
    zarr_file = initialize_zarr_store(zarr_path, image_list, overwrite_flag)

    # Specify time index and pixel resolution
    # print("Exporting image arrays...")
    if par_flag:
        process_map(
            partial(write_zarr, zarr_file=zarr_file, image_list=image_list,
                                 overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
                                 file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale,
                                 metadata_only_flag=metadata_only_flag),
                    range(len(image_list)), max_workers=n_workers)
    else:
        for i in tqdm(range(len(image_list)), "Exporting raw images to zarr..."):
            write_zarr(i, zarr_file=zarr_file, image_list=image_list,
                                 overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
                                 file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale,
                                 metadata_only_flag=metadata_only_flag)

    # for t in range(len(image_list)):
    #     write_image(t, image_list=image_list, project_path=project_path,
    #                          overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
    #                          file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale)

    # Export each Dask array to the same OME-Zarr file one at a time
    # for t in tqdm(range(len(image_list))):


    print("Done.")


if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([1.5, 1.5, 1.5])
    tres = 60  # time resolution in seconds

    # set path parameters
    # raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    raw_data_root = "D:\\Syd\\230425_EXP21_LCP1_D6_1pm_DextranStabWound\\" #"D:\\Syd\\240219_LCP1_67hpf_to_"
    file_prefix = "e2_2023_04_25__20_08_16_426" #"E3_186_TL_start93hpf_2024_02_20__19_13_43_218"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "230425_EXP21_LCP1_D6_1pm_DextranStabWound"
    overwrite = True
    export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres,
                       par_flag=False, overwrite_flag=overwrite)