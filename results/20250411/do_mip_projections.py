import numpy as np
import os
from tqdm import tqdm
import zarr
from dask.diagnostics import ProgressBar
import dask.array as da
from dask.distributed import Client
client = Client(n_workers=24, threads_per_worker=1)


def write_zarr(t, in_data, out_zarr, dim1=1):
    """
    Write a single timepoint to the zarr array.
    """
    # Copy the data from in_data to out_zarr at the specified timepoint t
    out_zarr[t, dim1] = in_data[t]
    return True

def calculate_mip(t, full_zarr, mip_zarr, dual_sided=True, axis=1):
    """
    Calculate the maximum intensity projection (MIP) of a zarr array along a specified axis.
    """
    zarr_frame = full_zarr[t]
    zarr_shape = zarr_frame.shape
    div_lim = zarr_shape[axis] // 2

    start_array0 = np.zeros((len(zarr_shape),), dtype=np.int32)
    start_array1 = start_array0.copy()
    start_array1[axis] = div_lim
    stop_array1 = np.asarray(zarr_shape)
    stop_array0 = stop_array1.copy()
    stop_array0[axis] = div_lim

    slices0 = tuple(slice(start_array0[l], stop_array0[l]) for l in range(len(zarr_shape)))
    slices1 = tuple(slice(start_array1[l], stop_array1[l]) for l in range(len(zarr_shape)))

    mip0 = np.max(zarr_frame[slices0], axis=axis)
    mip1 = np.max(zarr_frame[slices1], axis=axis)

    mip_zarr[t, 0] = mip0
    mip_zarr[t, 1] = mip1

    return True

if __name__ == "__main__":

    # script to stitch tracks after initial tracking. Also updates corresponding seg_zarr's
    # At this point, should have tracked all relevant experiments

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    project_name_list = ["20250311_LCP1-NLSMSC"]

    # set gap closing parameters
    p_axis = 2
    n_workers = 24
    overwrite = False
    par_flag = True

    for i in tqdm(range(len(project_name_list)), desc="Processing projects", unit="project"):

        project_name = project_name_list[i]

        # open image zarr
        zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
        fused_image_zarr = zarr.open(zpath, mode="r")
        image_da = da.from_zarr(zpath)

        # get shape info
        full_shape = fused_image_zarr.shape
        mip_shape = tuple([full_shape[0], full_shape[1], 2, full_shape[3], full_shape[4]])


        # split between front and back
        div_lim = full_shape[p_axis] // 2

        start_array0 = np.zeros((len(full_shape),), dtype=np.int32)
        start_array1 = start_array0.copy()
        start_array1[p_axis] = div_lim
        stop_array1 = np.asarray(full_shape)
        stop_array0 = stop_array1.copy()
        stop_array0[p_axis] = div_lim

        slices0 = tuple(slice(start_array0[l], stop_array0[l]) for l in range(len(full_shape)))
        slices1 = tuple(slice(start_array1[l], stop_array1[l]) for l in range(len(full_shape)))

        # do projections
        print("Projecting front and back halves.")
        front_half = image_da[slices0]
        back_half = image_da[slices1]

        front_proj = front_half.max(axis=p_axis)
        back_proj = back_half.max(axis=p_axis)

        # stack
        combined = da.stack([front_proj, back_proj], axis=2)

        # initialize  zarr store
        mip_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_mip.zarr")
        mip_zarr = zarr.open(mip_path, mode='a', shape=mip_shape, chunks=(1, 1) + mip_shape[-3:], dtype=combined.dtype)
        mip_zarr.attrs.update(fused_image_zarr.attrs)

        print("Saving...")
        with ProgressBar():
            combined.to_zarr(mip_zarr, overwrite=True)
        print("Done.")


