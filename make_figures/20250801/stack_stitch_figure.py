import os
import pandas as pd
import scipy.ndimage as ndi
import zarr
import napari
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    # shift info
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241114_LCP1-NLSMSC"

    df_path = os.path.join(root, "metadata", project_name, "")
    shift_df = pd.read_csv(os.path.join(df_path, "frame_shift_df.csv"))

    # load image stacks
    zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\zarr_image_files\\"
    side1_name = "20241114_LCP1-NLSMSC_side1.zarr"
    side2_name = "20241114_LCP1-NLSMSC_side2.zarr"

    # open zar files
    image_data1 = zarr.open(os.path.join(zarr_path, side1_name), mode="r")
    image_data2 = zarr.open(os.path.join(zarr_path, side2_name), mode="r")

    # get voxel size
    scale_vec = tuple([image_data1.attrs['PhysicalSizeZ'],
                       image_data1.attrs['PhysicalSizeY'],
                       image_data1.attrs['PhysicalSizeX']])

    t_start = 200
    t_stop = 202
    channel_i = 1

    print("Initializing shifted array...")
    # image_data2_shift = np.zeros_like(image_data2[t_start:t_stop, channel_i])

    # make array to store full sphere
    print("initializing giant arrays...")
    zdim1_orig = image_data1.shape[2]
    zdim2_orig = image_data2.shape[2]
    full_z = zdim1_orig + zdim2_orig #int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
    full_shape = tuple([t_stop-t_start, full_z]) + tuple(image_data1.shape[3:])
    data_full1 = np.zeros(full_shape, dtype=np.uint16)
    data_full2 = data_full1.copy()
    print("Done.")

    # add arrays
    data_full1[:, -zdim1_orig:, :, :] = image_data1[t_start:t_stop, channel_i]
    im2 = image_data2[t_start:t_stop, channel_i]
    data_full2[:, :zdim2_orig, :, :] = im2[:, ::-1, :, ::-1]
    data_full2_shift = np.zeros_like(data_full2)

    for t, time in enumerate(tqdm(range(t_start, t_stop), "Applying shifts")):
        shift = shift_df.loc[time, ["zs", "ys", "xs"]].to_numpy()
        # experiment with manual alignment
        im_slice = data_full2[t, :, :, :]
        data_full2_shift[t] = ndi.shift(im_slice, (shift), order=1)


    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(data_full1, scale=scale_vec, opacity=0.7, colormap="gray",
                     contrast_limits=[0, 2500])
    viewer.add_image(data_full2_shift, scale=scale_vec, opacity=0.7, colormap="cyan",
                     contrast_limits=[0, 2500])
    # viewer.add_image(data_full2_shift, scale=scale_vec, opacity=0.7, colormap="cyan", contrast_limits=[0, 2500])