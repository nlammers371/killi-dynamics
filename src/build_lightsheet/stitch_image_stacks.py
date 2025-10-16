import napari
import numpy as np
import zarr
import scipy.ndimage as ndi
import os
from skimage.registration import phase_cross_correlation
from scipy.ndimage import rotate
from tqdm import tqdm
import pandas as pd


def stitch_image_stacks(root, project_name, interval=25, nucleus_channel=1, z_align_size=100, last_i=None):

    # load zarr files
    side1_path = os.path.join(root, "built_data", "zarr_image_file", project_name + "_side1.zarr")
    side2_path = os.path.join(root, "built_data", "zarr_image_file", project_name + "_side2.zarr")

    image_data1 = zarr.open(side1_path, mode="r")
    image_data2 = zarr.open(side2_path, mode="r")

    if last_i is None:
        last_i = np.min([image_data1.shape[0], image_data2.shape[0]]) - 1

    frame_vec = np.arange(last_i)

    frames_to_register = list(np.arange(0, last_i, interval))
    frames_to_register = np.asarray(frames_to_register + [last_i])

    shift_vec = []

    for f, frame in enumerate(tqdm(frames_to_register)):

        data_zyx1 = np.squeeze(image_data1[frame, nucleus_channel])
        data_zyx2 = np.squeeze(image_data2[frame, nucleus_channel])

        # experiment with manual alignment
        data_zyx2_i = data_zyx2[::-1, :, ::-1]

        # ALIGN
        align1 = data_zyx1[:z_align_size, :, :]
        align2 = data_zyx2_i[-z_align_size:, :, :]

        shift, error, _ = phase_cross_correlation(
            align1,
            align2,
            normalization=None,
            upsample_factor=1,
            overlap_ratio=0.05,
        )

        # make array to store full sphere
        # full_shape = list(data_zyx1.shape)
        # full_shape[0] = int(np.ceil((data_zyx1.shape[0] + data_zyx1.shape[0]) / 10) * 10)
        # data_full1 = np.zeros(tuple(full_shape), dtype=np.uint16)
        # data_full2 = data_full1.copy()
        #
        # # add arrays
        # data_full1[-data_zyx1.shape[0]:, :, :] = data_zyx1
        # data_full2[:data_zyx2.shape[0]:, :, :] = data_zyx2_i

        # apply shift
        shift_corrected = shift.copy() + z_align_size
        shift_vec.append(shift_corrected)

        # shift_corrected[0] = shift_corrected[0] + z_align_size
        # data_full2_shift = ndi.shift(data_full2, (shift_corrected), order=1)

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

    stitch_image_stacks(root=root, project_name=project_name)
