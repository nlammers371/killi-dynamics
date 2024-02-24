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
# def create_dummy_data(shape):
#     return da.random.random(shape, chunks=(100, 100, 100))

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
def export_czi_to_tiff(raw_data_root, file_prefix, project_name, save_root, tres, overwrite_flag=False, resampling_scale=None, n_workers=6):

    if resampling_scale is None:
        resampling_scale = np.asarray([1.5, 1.5, 1.5])

    project_path = os.path.join(save_root, "built_data", project_name, '')

    if not os.path.isdir(project_path):
        os.makedirs(project_path)
    # ome_zarr_file_path = project_path + exp_name + ".ome.zarr"
    metadata_file_path = os.path.join(save_root, "metadata",  "metadata.json")

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
    # image_list = image_list
    # n_time_points = len(image_list)

    # Specify time index and pixel resolution
    print("Exporting image arrays...")
    process_map(
        partial(write_image, image_list=image_list, project_path=project_path,
                             overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
                             file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale),
                range(len(image_list)), max_workers=n_workers)
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
    tres = 90  # time resolution in seconds

    # set path parameters
    # raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    raw_data_root = "D:\\Syd\\240219_LCP1_67hpf_to_"
    file_prefix = "E3_186_TL_start93hpf_2024_02_20__19_13_43_218"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf"

    export_czi_to_tiff(raw_data_root, file_prefix, project_name, save_root, tres)