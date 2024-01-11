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




if __name__ == '__main__':

    da_chunksize = (32, 512, 512)
    resampling_res = np.asarray([2.0, 2.0, 2.0])

    # set path parameters
    raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    file_prefix = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
    fullfile_name = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
    project_name = "20230108"

    project_path = os.path.join(save_root, project_name, '')

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
    n_time_points = len(image_list)

    time_point = 10
    readPath = os.path.join(raw_data_root, file_prefix + f"({time_point}).czi")
    imObject = AICSImage(readPath)
    image_data = np.squeeze(imObject.data)
    raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)

    dims_orig = image_data.shape
    rs_factors = np.divide(raw_scale_vec, resampling_res)
    dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)
    image_data_rs = resize(image_data, dims_new, order=1)

    image_dask = da.from_array(image_data, chunks=da_chunksize)


    # viewer = napari.view_image(image_data, scale=tuple(scale_vec))

    # Use otsu's method to get binarized image
import dask
import dask.array as da
import aicsimageio

def create_dummy_data(shape):
    return da.random.random(shape, chunks=(100, 100, 100))

def export_dask_array_to_ome_zarr(array, ome_zarr_path, array_name):
    # Convert Dask array to NumPy array for export to OME-Zarr
    array_np = array.compute()

    # Export the array to OME-Zarr
    aicsimageio.writers.ome_zarr.write_ome_zarr(ome_zarr_path, {array_name: array_np})

def import_ome_zarr_file(ome_zarr_path):
    # Read OME-Zarr file
    zarr_reader = aicsimageio.readers.ome_zarr.OmeZarrReader(ome_zarr_path)

    # Access a specific dataset (replace 'array_0' with the actual array name)
    array_0 = zarr_reader.get_dataset("array_0")

    # Convert NumPy array to Dask array
    dask_array_0 = da.from_array(array_0, chunks=array_0.shape)

    return dask_array_0

if __name__ == "__main__":
    # Create a list of 3D Dask arrays
    array_list = [create_dummy_data((100, 100, 100)) for _ in range(5)]

    # Specify the path to the output OME-Zarr file
    ome_zarr_file_path = "output.ome.zarr"

    # Export each Dask array to the same OME-Zarr file one at a time
    for i, dask_array in enumerate(array_list):
        # Create a unique array name for each array
        array_name = f"array_{i}"

        # Export the Dask array to the OME-Zarr file
        export_dask_array_to_ome_zarr(dask_array, ome_zarr_file_path, array_name)

        print(f"Array {i} exported to {ome_zarr_file_path} as {array_name}")

    # Import the OME-Zarr file and access a Dask array
    imported_dask_array = import_ome_zarr_file(ome_zarr_file_path)

    # Perform operations on the imported Dask array if needed
    # ...

    print("Imported Dask array shape:", imported_dask_array.shape)