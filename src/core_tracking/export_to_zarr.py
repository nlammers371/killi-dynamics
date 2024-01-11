import dask
import dask.array as da
import aicsimageio
import numpy as np
import json
import os
import glob2 as glob
from aicsimageio import AICSImage
from skimage.transform import resize
from tqdm import tqdm

# def create_dummy_data(shape):
#     return da.random.random(shape, chunks=(100, 100, 100))

def save_metadata(metadata, metadata_file):
    # Save metadata to a JSON file
    with open(metadata_file, 'w') as json_file:
        json.dump(metadata, json_file)

def load_metadata(metadata_file):
    # Load metadata from a JSON file
    with open(metadata_file, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata

def export_dask_array_to_ome_zarr(array, ome_zarr_path, array_name, metadata, metadata_file):
    # Convert Dask array to NumPy array for export to OME-Zarr
    array_np = array.compute()

    # Save metadata to a JSON file
    save_metadata(metadata, metadata_file)

    # Export the array to OME-Zarr
    aicsimageio.writers.ome_zarr.write_ome_zarr(ome_zarr_path, {array_name: array_np}, metadata=metadata_file)

def import_ome_zarr_file(ome_zarr_path):
    # Read OME-Zarr file
    zarr_reader = aicsimageio.readers.ome_zarr.OmeZarrReader(ome_zarr_path)

    # Access a specific dataset (replace 'array_0' with the actual array name)
    array_0 = zarr_reader.get_dataset("array_0")

    # Convert NumPy array to Dask array
    dask_array_0 = da.from_array(array_0, chunks=array_0.shape)

    return dask_array_0

if __name__ == "__main__":

    da_chunksize = (207, 310, 310)
    resampling_scale = np.asarray([2.0, 2.0, 2.0])
    tres = 98.25  # time resolution in seconds

    # set path parameters
    raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    file_prefix = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
    fullfile_name = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
    project_name = "231016_EXP40"
    exp_name = "PreUVB_Timelapse_ds"
    project_path = os.path.join(save_root, project_name, '')

    if not os.path.isdir(project_path):
        os.makedirs(project_path)
    # ome_zarr_file_path = project_path + exp_name + ".ome.zarr"
    metadata_file_path = project_path + exp_name + "_metadata.json"

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
    n_time_points = len(image_list)

    # Specify time index and pixel resolution
    image_list = image_list[:5]
    t_curr = 0
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
        image_data_rs = resize(image_data, dims_new, order=1)

        # Convert to dask array
        image_dask = da.from_array(image_data, chunks=da_chunksize)

        # Create a unique array name for each array


        # Create metadata for each array
        metadata = {
            "SizeT": len(image_list),  # Assuming there is only one timepoint per array, adjust if needed
            "TimeIndex": time_point,
            "Time(s)": t_curr,
            "PhysicalSizeX": resampling_scale[2],
            "PhysicalSizeY": resampling_scale[1],
            "PhysicalSizeZ": resampling_scale[0],
        }

        # Export the Dask array to the OME-Zarr file
        if time_point == 0:
            # Save metadata to a JSON file
            with open(metadata_file_path, 'w') as json_file:
                json.dump(metadata, json_file)

        # save actual image array
        save_path = project_path + exp_name + f"_t{time_point:04}.zarr"
        da.to_zarr(image_dask, save_path, overwrite=True)

        t_curr += tres
        print(f"Array {time_point} exported to {save_path}")

    print("check")
    #
    # # Import the OME-Zarr file and access a Dask array
    # imported_dask_array = import_ome_zarr_file(ome_zarr_file_path)
    #
    # # Load metadata from the JSON file
    # loaded_metadata = load_metadata(metadata_file_path)

    # Perform operations on the imported Dask array and metadata if needed
    # ...

    # print("Imported Dask array shape:", imported_dask_array.shape)
    # print("Loaded metadata:", loaded_metadata)