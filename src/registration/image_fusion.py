import numpy as np
import zarr
import os
from skimage.registration import phase_cross_correlation
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial
import multiprocessing


AUTO_CSV_SENTINEL = object()


def align_halves(t, image_data1, image_data2, z_align_size=50, nucleus_channel=1):

    multichannel = len(image_data2.shape) > 4
    if multichannel:
        data_zyx1 = np.squeeze(image_data1[t, nucleus_channel])
        data_zyx2 = np.squeeze(image_data2[t, nucleus_channel])
    else:
        data_zyx1 = np.squeeze(image_data1[t])
        data_zyx2 = np.squeeze(image_data2[t])

    # experiment with manual alignment
    data_zyx2_i = data_zyx2[::-1, :, ::-1]

    # ALIGN
    align1 = data_zyx1[:z_align_size, :, :]
    align2 = data_zyx2_i[-z_align_size:, :, :]

    shift, error, _ = phase_cross_correlation(
        align1,
        align2,
        normalization=None,
        upsample_factor=2,
        overlap_ratio=0.05,
    )

    # apply shift
    shift_corrected = shift.copy()
    shift_corrected[0] += z_align_size

    return shift_corrected

def get_hemisphere_shifts(
    root,
    side1_name,
    side2_name,
    interval=25,
    nucleus_channel=1,
    z_align_size=50,
    last_i=None,
    start_i=0,
    n_workers=None,
    *,
    side1_path=None,
    side2_path=None,
    csv_output_path=AUTO_CSV_SENTINEL,
):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit to 25% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 4)

    if side1_path is None or side2_path is None:
        if root is None:
            raise ValueError(
                "Either root must be provided to derive Zarr paths or explicit side paths must be supplied."
            )
        side1_path = os.path.join(root, "built_data", "zarr_image_files", side1_name + ".zarr")
        side2_path = os.path.join(root, "built_data", "zarr_image_files", side2_name + ".zarr")

    if csv_output_path is AUTO_CSV_SENTINEL:
        if root is None:
            csv_output_path = None
        else:
            csv_output_path = os.path.join(
                root,
                "metadata",
                side1_name,
                side2_name + "_to_" + side1_name + "_shift_df.csv",
            )

    image_data1 = zarr.open(side1_path, mode="r")
    image_data2 = zarr.open(side2_path, mode="r")

    if last_i is None:
        last_i = np.min([image_data1.shape[0], image_data2.shape[0]]) - 1

    frame_vec = np.arange(start_i, last_i + 1)

    frames_to_register = list(np.arange(start_i, last_i, interval))
    frames_to_register = np.unique(frames_to_register + [last_i])

    # initialize function
    align_fun = partial(align_halves, image_data1=image_data1, image_data2=image_data2,
                        z_align_size=z_align_size, nucleus_channel=nucleus_channel)
    # apply
    shift_vec = process_map(align_fun, frames_to_register, max_workers=n_workers, chunksize=1)

    # interpolate
    shift_array = np.asarray(shift_vec)
    shift_array_interp = np.empty((len(frame_vec), 3))
    shift_array_interp[:, 0] = np.interp(frame_vec, frames_to_register, shift_array[:, 0], left=shift_array[0, 0])
    shift_array_interp[:, 1] = np.interp(frame_vec, frames_to_register, shift_array[:, 1], left=shift_array[0, 1])
    shift_array_interp[:, 2] = np.interp(frame_vec, frames_to_register, shift_array[:, 2], left=shift_array[0, 2])

    shift_df = pd.DataFrame(frame_vec, columns=["frame"])
    shift_df[["zs", "ys", "xs"]] = shift_array_interp

    if csv_output_path:
        out_path = os.path.dirname(csv_output_path)
        os.makedirs(out_path, exist_ok=True)
        shift_df.to_csv(csv_output_path, index=False)

    return shift_df
