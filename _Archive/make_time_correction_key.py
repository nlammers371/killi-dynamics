
import numpy as np
import pandas as pd
import os
import glob2 as glob
from aicsimageio import AICSImage
from skimage.transform import resize
from tqdm import tqdm
from skimage import io
from src.utilities.functions import path_leaf
# def create_dummy_data(shape):
#     return da.random.random(shape, chunks=(100, 100, 100))

def make_time_key(raw_data_root, file_prefix, project_name, metadata_root):

    out_path = os.path.join(metadata_root, project_name, '')

    # specify time points to load
    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
    # image_list = image_list

    # Specify time index and pixel resolutio
    # Export each Dask array to the same OME-Zarr file one at a time
    time_arr = np.empty((len(image_list), 2), dtype=np.int)
    for t, image_path in enumerate(tqdm(image_list)):
        f_string = path_leaf(image_list[t])
        time_string = f_string.replace(file_prefix, "")
        time_string = time_string.replace(".czi", "")
        time_point = int(time_string[1:-1])

        time_arr[t, 0] = t
        time_arr[t, 1] = time_point

    time_df = pd.DataFrame(time_arr, columns=["time_curr", "time_corrected"])
    time_df.to_csv(os.path.join(out_path, "time_key.csv"), index=False)


    print("Done.")


if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)

    # set path parameters
    raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\WT_Timelapse_Raw\\"
    # raw_data_root = "D:\\Syd\\"
    file_prefix = "e2_e3_LCP1_postZap30min_G1UVB_G2WT_Timelapse_2023_10_17__00_05_52_872_G2"  #"E3_186_TL_start93hpf_2024_02_20__19_13_43_218" #

    # Specify the path to the output OME-Zarr file and metadata file
    metadata_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\metadata\\"
    project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #"240219_LCP1_67hpf_to_"

    make_time_key(raw_data_root, file_prefix, project_name, metadata_root)