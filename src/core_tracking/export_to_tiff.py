# import dask
import dask.array as da
import aicsimageio
import numpy as np
import json
import os
import glob2 as glob
from aicsimageio import AICSImage
from skimage.transform import resize
from tqdm import tqdm
from skimage import io
# from src.utilities.io import create_zarr
# def create_dummy_data(shape):
#     return da.random.random(shape, chunks=(100, 100, 100))


if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([1.5, 1.5, 1.5])
    tres = 121.86  # time resolution in seconds

    # set path parameters
    # raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\WT_Timelapse_Raw\\"
    file_prefix = "e2_e3_LCP1_postZap30min_G1UVB_G2WT_Timelapse_2023_10_17__00_05_52_872_G2"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
    project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
    project_path = os.path.join(save_root, project_name, '')

    if not os.path.isdir(project_path):
        os.makedirs(project_path)
    # ome_zarr_file_path = project_path + exp_name + ".ome.zarr"
    metadata_file_path = project_path + project_name + "_metadata.json"

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
    n_time_points = len(image_list)

    # Specify time index and pixel resolution
    print("Resizing image arrays...")
    # Export each Dask array to the same OME-Zarr file one at a time
    for time_point, image_path in enumerate(tqdm(image_list)):

        # readPath = os.path.join(raw_data_root, file_prefix + f"({time_point}).czi")

        # Load image
        imObject = AICSImage(image_path)
        image_data = np.squeeze(imObject.data)
        raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)

        # Resize
        dims_orig = image_data.shape
        rs_factors = np.divide(raw_scale_vec, resampling_scale)
        dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)
        image_data_rs = np.round(resize(image_data, dims_new, preserve_range=True, order=1)).astype(np.uint16)

        # Export the Dask array to the OME-Zarr file
        if time_point == 0:
            metadata = {
                "DimOrder": "tzyx",
                "Dims": [n_time_points, image_data_rs.shape[0], image_data_rs.shape[1], image_data_rs.shape[2]],  # Assuming there is only one timepoint per array, adjust if needed
                "TimeRes": tres,
                "PhysicalSizeX": resampling_scale[2],
                "PhysicalSizeY": resampling_scale[1],
                "PhysicalSizeZ": resampling_scale[0],
            }

            # Save metadata to a JSON file
            with open(metadata_file_path, 'w') as json_file:
                json.dump(metadata, json_file)


            # save actual image array
        save_path = project_path + project_name + f"_t{time_point:04}.tiff"
        io.imsave(save_path, image_data_rs)

    print("Done.")
