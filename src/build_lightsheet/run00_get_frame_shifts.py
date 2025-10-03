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
from scipy.interpolate import interp1d
import multiprocessing

# Syd had to reposition the FOV early on in the imaging process. As a result the early and late timelapse frames were
# exported separately. This script registers early with late, and generates a combine zarr file

def extract_bottom_right_quadrant(vol):
    """Extract bottom-right quadrant in the YX plane of a ZYX stack."""
    z, y, x = vol.shape
    return vol[:, y//2:, x//2:]

def align_frames(t, data_zyx, interval, nucleus_channel):

    data_zyx0 = np.squeeze(data_zyx[t, nucleus_channel])
    data_zyx1 = np.squeeze(data_zyx[t + interval, nucleus_channel])

    # Extract bottom-right quadrant
    data_zyx0 = extract_bottom_right_quadrant(data_zyx0)
    data_zyx1 = extract_bottom_right_quadrant(data_zyx1)

    shift, error, _ = phase_cross_correlation(
        data_zyx0,
        data_zyx1,
        normalization=None,
        upsample_factor=5,
        overlap_ratio=0.7,
    )

    return shift

def get_timeseries_shifts(root, project_name, interval=1, nucleus_channel=1, calc_chunk=None, n_workers=None, last_i=None):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit to 25% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 4)

    # load zarr files
    zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")

    image_data = zarr.open(zarr_path, mode="r")

    if last_i is None:
        last_i = image_data.shape[0] - interval - 1

    frame_vec = np.arange(0, last_i + interval + 1)

    frames_to_register = list(np.arange(0, last_i, interval))
    frames_to_register = np.asarray(frames_to_register + [last_i])

    if interval > 5:
        raise Warning("Large frame interval specified. Registration may be suboptimal")

    # initialize function
    shift_fun = partial(align_frames, data_zyx=image_data, interval=interval, nucleus_channel=nucleus_channel)

    # run
    shift_vec = process_map(shift_fun, frames_to_register, max_workers=n_workers, chunksize=1)

    # interpolate
    shift_array = np.asarray(shift_vec)
    shift_array_interp = np.empty((len(frame_vec), 3))
    interp0 = interp1d(frames_to_register, shift_array[:, 0], kind="linear", fill_value="extrapolate")
    shift_array_interp[:, 0] = interp0(frame_vec)
    interp1 = interp1d(frames_to_register, shift_array[:, 1], kind="linear", fill_value="extrapolate")
    shift_array_interp[:, 1] = interp1(frame_vec)
    interp2 = interp1d(frames_to_register, shift_array[:, 2], kind="linear", fill_value="extrapolate")
    shift_array_interp[:, 2] = interp2(frame_vec)

    # we need to adjust for frame interval
    shift_frames = np.asarray([0] + list(frames_to_register + interval))
    frame_deltas = np.diff(shift_frames)
    frame_deltas_full = np.repeat(frame_deltas, frame_deltas)

    shift_array_norm = np.divide(shift_array_interp, frame_deltas_full[:, None])
    shift_array_cs = np.cumsum(shift_array_norm, axis=0)

    # make shift data frame
    shift_df = pd.DataFrame(frame_vec, columns=["frame"])
    shift_df[["xs", "ys", "zs"]] = shift_array_cs

    # lets also save raw data
    shift_df_raw = pd.DataFrame(frames_to_register, columns=["frame"])
    shift_df_raw[["xs", "ys", "zs"]] = shift_array

    out_path = os.path.join(root, "metadata", project_name, "")
    os.makedirs(out_path, exist_ok=True)
    shift_df.to_csv(os.path.join(out_path, "frame_shift_df.csv"), index=False)
    shift_df_raw.to_csv(os.path.join(out_path, "frame_shift_df_raw.csv"), index=False)


if __name__ == '__main__':
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241114_LCP1-NLSMSC"

    # stitch_image_stacks(root=root, project_name=project_name)
