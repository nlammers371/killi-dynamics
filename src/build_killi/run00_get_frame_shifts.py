import napari
import numpy as np
import zarr
import scipy.ndimage as ndi
import os
from skimage.registration import phase_cross_correlation
from scipy.ndimage import rotate
from tqdm import tqdm
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial

# Syd had to reposition the FOV early on in the imaging process. As a result the early and late timelapse frames were
# exported separately. This script registers early with late, and generates a combine zarr file

def align_frames(t, data_zyx, interval, nucleus_channel):

    data_zyx0 = np.squeeze(data_zyx[t, nucleus_channel])
    data_zyx1 = np.squeeze(data_zyx[t + interval, nucleus_channel])

    shift, error, _ = phase_cross_correlation(
        data_zyx0,
        data_zyx1,
        normalization=None,
        upsample_factor=1,
        overlap_ratio=0.7,
    )

    return shift

def get_timeseries_shifts(root, project_name, interval=25, nucleus_channel=1, par_flag=False, last_i=None):

    # load zarr files
    zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")

    image_data = zarr.open(zarr_path, mode="r")

    if last_i is None:
        last_i = image_data.shape[0] - interval - 1

    frame_vec = np.arange(last_i)

    frames_to_register = list(np.arange(0, last_i, interval))
    frames_to_register = np.asarray(frames_to_register + [last_i])

    # initialize function
    shift_fun = partial(align_frames, data_zyx=image_data, interval=interval, nucleus_channel=nucleus_channel)

    # run
    shift_vec = process_map(shift_fun, frames_to_register, max_workers=4, chunksize=1)

    # interpolate
    shift_array = np.asarray(shift_vec)
    shift_array_interp = np.empty((len(frame_vec), 3))
    shift_array_interp[:, 0] = np.interp(frame_vec, frames_to_register, shift_array[:, 0])
    shift_array_interp[:, 1] = np.interp(frame_vec, frames_to_register, shift_array[:, 1])
    shift_array_interp[:, 2] = np.interp(frame_vec, frames_to_register, shift_array[:, 2])

    shift_df = pd.DataFrame(frame_vec, columns=["frame"])
    shift_df[["xs", "ys", "zs"]] = shift_array_interp

    out_path = os.path.join(root, "metadata", project_name, "")
    os.makedirs(out_path, exist_ok=True)
    shift_df.to_csv(os.path.join(out_path, "frame_shift_df.csv"), index=False)


if __name__ == '__main__':
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241114_LCP1-NLSMSC"

    # stitch_image_stacks(root=root, project_name=project_name)
