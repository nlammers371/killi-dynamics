import zarr
import os
import pandas as pd
import scipy.ndimage as ndi
import numpy as np
from tqdm import tqdm
import napari
from skimage.measure import regionprops, label
from scipy.sparse import csr_matrix, coo_matrix
from tqdm.contrib.concurrent import process_map
from functools import partial
import multiprocessing


if __name__ == "__main__":

    last_i = 2200

    # load masks
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_prefix = "20250311_LCP1-NLSMSC"

    project1 = project_prefix + "_side1"
    project2 = project_prefix + "_side2"

    # load mask zarr file
    mpath1 = os.path.join(root, "built_data", "mask_stacks", project1 + "_mask_aff.zarr")
    mask1 = zarr.open(mpath1, mode="r")

    mpath2 = os.path.join(root, "built_data", "mask_stacks", project2 + "_mask_aff.zarr")
    mask2 = zarr.open(mpath2, mode="r")

    if last_i is None:
        last_i = mask1.shape[0]

    # get scale info
    scale_vec = tuple([mask1.attrs['PhysicalSizeZ'],
                       mask1.attrs['PhysicalSizeY'],
                       mask1.attrs['PhysicalSizeX']])

    # load shift info
    data_path = os.path.join(root, "metadata", project1, "")
    half_shift_df = pd.read_csv(os.path.join(data_path, project2 + "_to_" + project1 + "_shift_df.csv"))
    time_shift_df = pd.read_csv(os.path.join(data_path, "frame_shift_df.csv"))

    # generate shift arrays
    side1_shifts = time_shift_df.copy()
    side2_shifts = time_shift_df.copy() + half_shift_df.copy()

    # subset frames
    frames_to_align = np.arange(2180, last_i)

    # extract ID dictionary
    keep_dict1 = mask1.attrs["mask_keep_ids"]
    keep_dict2 = mask2.attrs["mask_keep_ids"]

    # mask1 = mask1[frames_to_align]
    # mask2 = mask2[frames_to_align]

    # initialze 'full' array
    zdim1_orig = mask1.shape[1]
    zdim2_orig = mask2.shape[1]
    full_z = zdim1_orig + zdim2_orig  # int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
    full_shape = tuple([full_z]) + tuple(mask1.shape[2:])

    masks_fused = np.zeros(tuple([len(frames_to_align)]) + full_shape, dtype=np.uint16)

    # generate fused mask frames
    frame = 2180
    frame_i = frames_to_align == frame

    shift1 = side1_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()
    shift2 = side2_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()

    # data_zyx1 = np.squeeze(mask1[frame_i])
    # data_zyx2 = np.squeeze(mask2[frame_i][::-1, :, ::-1])

    # extract masks
    m1 = np.squeeze(mask1[frame])
    m2 = np.squeeze(mask2[frame])

    # filter
    keep_ids1 = keep_dict1[str(frame)]
    m1_binary = np.isin(m1, keep_ids1)
    keep_ids2 = keep_dict2[str(frame)]
    m2_binary = np.isin(m2, keep_ids2)

    # assign to full array
    m1_full = np.zeros(full_shape, dtype=np.uint16)
    m1_full[zdim2_orig:, :, :] = m1_binary[:, :, :]
    m2_full = np.zeros(full_shape, dtype=np.uint16)
    m2_full[:zdim2_orig, :, :] = m2_binary[::-1, :, ::-1]

    mask1_shifted = ndi.shift(m1_full, (shift1), order=0)
    mask2_shifted = ndi.shift(m2_full, (shift2), order=0)

    mask_fused = label((mask1_shifted + mask2_shifted) > 0)

    viewer = napari.Viewer()

    # viewer.add_image(im, scale=scale_vec, colormap="gray", contrast_limits=[0, 2500])
    viewer.add_labels(mask_fused, scale=scale_vec)
    # viewer.add_labels(m1_full, scale=scale_vec)
    # viewer.add_labels(m2_full, scale=scale_vec)