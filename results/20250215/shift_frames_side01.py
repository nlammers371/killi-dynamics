import zarr
import napari
import os
import numpy as np
from skimage.registration import phase_cross_correlation
import scipy.ndimage as ndi
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

# Syd had to reposition the FOV early on in the imaging process. As a result the early and late timelapse frames were
# exported separately. This script registers early with late, and generates a combine zarr file

def write_chunk(t, zarr_to, zarr_from, shift_flag=False, shift=None, offset=0):

    for c in range(zarr_from.shape[1]):
        # shift
        chunk = np.squeeze(zarr_from[t, c])
        if shift_flag:
            chunk_shifted = ndi.shift(chunk, (shift), order=1)
            zarr_to[offset + t, c] = chunk_shifted
        else:
            zarr_to[offset + t, c] = chunk



if __name__ == '__main__':
    # get filepaths
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project0 = "20250116_LCP1-NLSMSC_side1"
    project1 = "20250116_LCP1-NLSMSC_side1_reset"

    zpath0 = os.path.join(root, "built_data", "zarr_image_files", project0 + ".zarr")
    zpath1 = os.path.join(root, "built_data", "zarr_image_files", project1 + ".zarr")

    nc_index = 1 # index of NLS channel (used for registration)

    # load
    zarr0 = zarr.open(zpath0, mode="r")
    zarr1 = zarr.open(zpath1, mode="r")

    print("Initializing zarr file...")
    # initialize new zarr file to store combined files
    new_shape = (zarr0.shape[0] + zarr1.shape[0],) + zarr1.shape[1:]

    # Create a new zarr array for output with the desired shape and same chunking & dtype.
    zarr_cb = zarr.open(os.path.join(root, "built_data", "zarr_image_files", project0 + "_cb.zarr"), mode='w', shape=new_shape, chunks=zarr1.chunks, dtype=zarr1.dtype)

    # add attributes
    for key in list(zarr1.attrs.keys()):
        zarr_cb.attrs[key] = zarr1.attrs[key]

    # get scale info
    scale_vec = tuple([zarr0.attrs['PhysicalSizeZ'], zarr0.attrs['PhysicalSizeY'], zarr0.attrs['PhysicalSizeX']])
    zarr_cb.attrs["pixel_res_um"] = [zarr0.attrs['PhysicalSizeZ'], zarr0.attrs['PhysicalSizeY'], zarr0.attrs['PhysicalSizeX']]

    print("Performing registration...")
    # extract relevant frames
    last_f = np.squeeze(zarr0[-1, nc_index])
    first_f = np.squeeze(zarr1[0, nc_index])

    # calculate shift
    shift, error, _ = phase_cross_correlation(
                first_f,
                last_f,
                normalization=None,
                upsample_factor=1,
                overlap_ratio=0.05,
            )

    # apply shift to early frames. Store shifted frames in new zarr object
    # Compute the number of chunks along each axis.
    # grid_shape = [int(np.ceil(s / c)) for s, c in zip(zarr1.shape, zarr1.chunks)]
    print("Writing series 0...")
    map_fun0 = partial(write_chunk, zarr_from=zarr0, zarr_to=zarr_cb, shift_flag=True, shift=shift, offset=0)
    process_map(map_fun0, range(zarr0.shape[0]), max_workers=16, chunksize=1)

    print("Writing series 1...")
    map_fun1 = partial(write_chunk, zarr_from=zarr1, zarr_to=zarr_cb, shift_flag=False, shift=None, offset=zarr0.shape[0])
    process_map(map_fun1, range(zarr1.shape[0]), max_workers=16, chunksize=1)

    print("Done.")
