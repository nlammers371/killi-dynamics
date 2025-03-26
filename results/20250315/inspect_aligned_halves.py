import zarr
import napari
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.registration import phase_cross_correlation

import os
os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project1 = "20250311_LCP1-NLSMSC_side1"
project2 = "20250311_LCP1-NLSMSC_side2"
zpath1 = os.path.join(root, "built_data", "zarr_image_files", project1 + ".zarr")
zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")

# load
zarr1 = zarr.open(zpath1, mode="r")
zarr2 = zarr.open(zpath2, mode="r")

# generate frame indices
t_start = 550
t_stop = 552
nucleus_channel = 1
frames = np.arange(t_start, t_stop)

# get scale info
scale_vec = tuple([zarr1.attrs['PhysicalSizeZ'], zarr1.attrs['PhysicalSizeY'], zarr1.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
im1 = np.squeeze(zarr1[frames])
im2 = np.squeeze(zarr2[frames])

# load shift info
data_path = os.path.join(root, "metadata", project1, "")
shift_df = pd.read_csv(os.path.join(data_path, project2 + "_to_" + project1 + "_shift_df.csv"))
# 20250311_LCP1-NLSMSC_side2_to_20250311_LCP1-NLSMSC_side1_shift_df
# make array to store full sphere
print("initializing giant arrays...")
zdim1_orig = im1.shape[2]
zdim2_orig = im2.shape[2]
full_z = zdim1_orig + zdim2_orig #int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
full_shape = tuple([t_stop-t_start, full_z]) + tuple(im1.shape[3:])
data_full1 = np.zeros(full_shape, dtype=np.uint16)
data_full2 = data_full1.copy()
print("Done.")

# add arrays
data_full1[:, zdim2_orig:, :, :] = im1[:, nucleus_channel, :, :, :]
im2_temp = im2[:, nucleus_channel]
data_full2[:, :zdim2_orig, :, :] = im2_temp[:, ::-1, :, ::-1]
data_full2_shift = np.zeros_like(data_full2)

z_align_size = 50
overlap_guess = 30

shift_list_new = []
# apply shift
for t, time in enumerate(tqdm(range(t_start, t_start+1), "Applying shifts")):
    shift_calc = shift_df.loc[time, ["zs", "ys", "xs"]].to_numpy()

    data_zyx1 = np.squeeze(im1[t, nucleus_channel])
    data_zyx2 = np.squeeze(im2[t, nucleus_channel])

    # experiment with manual alignment
    data_zyx2_i = data_zyx2[::-1, :, ::-1]

    # ALIGN
    align1 = data_zyx1[:z_align_size, :, :]
    align2 = data_zyx2_i[-z_align_size:, :, :]
    # align1 = np.squeeze(data_full1[t, i1a:i1b, :, :])
    # align2 = np.squeeze(data_full2[t, i2a:i2b, :, :])  #[-z_align_size:, :, :]

    # side1_mask = np.zeros_like(align1, dtype=bool)
    # side1_mask[z_align_size:] = True
    # side2_mask = np.zeros_like(align2, dtype=bool)
    # side2_mask[:z_align_size + overlap_guess] = True

    # shift, error, _ = phase_cross_correlation(
    #     align1,
    #     align2,
    #     normalization=None,
    #     # reference_mask=side1_mask,
    #     # moving_mask=side2_mask,
    #     upsample_factor=1,
    #     overlap_ratio=0.05,
    # )
    # shift_list_new.append(shift)
    # shift_corrected = shift.copy()
    # shift_corrected[0] = shift_corrected[0] + z_align_size

    # experiment with manual alignment
    im_slice = data_full2[t, :, :, :]
    data_full2_shift[t] = ndi.shift(im_slice, (shift_calc), order=1)
    # align2_shifted = ndi.shift(align2, (shift), order=1)
#
print(shift_list_new)

viewer = napari.Viewer()

# viewer.add_image(align1, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
# viewer.add_image(align2, scale=scale_vec,  colormap="cyan", contrast_limits=[0, 2500])
# viewer.add_image(align2_shifted, scale=scale_vec,  colormap="cyan", contrast_limits=[0, 2500])
viewer.add_image(data_full1, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
# viewer.add_image(im1, scale=scale_vec, channel_axis=1)#,  contrast_limits=[0, 2500])
# viewer.add_image(data_full2, scale=scale_vec, colormap="cyan", contrast_limits=[0, 2500])
viewer.add_image(data_full2_shift, scale=scale_vec,  colormap="magma", contrast_limits=[0, 2500])

napari.run()

