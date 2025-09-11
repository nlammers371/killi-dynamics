import numpy as np
import os
from tqdm import tqdm
import zarr
from dask.diagnostics import ProgressBar
import dask.array as da
from dask.distributed import Client


def write_zarr(t, in_data, out_zarr, dim1=1):
    """
    Write a single timepoint to the zarr array.
    """
    # Copy the data from in_data to out_zarr at the specified timepoint t
    out_zarr[t, dim1] = in_data[t]
    return True

def project_dual_sided(image_da, full_shape, p_axis=2):

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

    return combined

def project_single_sided(image_da, p_axis):
    """
    Project a single-sided image along the specified axis.
    """
    print("Projecting single-sided image.")
    proj = image_da.max(axis=p_axis)

    return proj

def calculate_mip_projection(root, project_name, dual_sided=True):


    # open image zarr
    if dual_sided:
        zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    else:
        zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
    image_zarr = zarr.open(zpath, mode="r")
    image_da = da.from_zarr(zpath)


    # get shape info
    full_shape = image_zarr.shape
    multchannel_flag = len(image_zarr.attrs["Channels"]) > 1

    rechunk_map = {0: 1}
    if multchannel_flag:
        rechunk_map[1] = 1  # C
        rechunk_map[2] = -1  # Z
    else:
        rechunk_map[1] = -1  # Z
    image_da = image_da.rechunk(rechunk_map)

    # check if multichannel
    p_axis = 2 if multchannel_flag else 1

    # get projection
    proj_da = project_dual_sided(image_da, full_shape, p_axis) if dual_sided else project_single_sided(image_da, p_axis)

    # rechunk
    proj_da = proj_da.rechunk({0: 1})
    if multchannel_flag:
        proj_da = proj_da.rechunk({1: 1})

    # initialize  zarr store
    # if dual_sided:
    #     mip_shape = tuple(full_shape[:-3] + (2,) + full_shape[-2:])
    # else:
    #     mip_shape = tuple(full_shape[:-3] + full_shape[-2:])
    # mip_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_mip.zarr")
    # if dual_sided:
    #     mip_zarr = zarr.open(mip_path, mode='w', shape=mip_shape, chunks=(1, 1,) + mip_shape[-3:], dtype=proj_da.dtype)
    # else:
    #     mip_zarr = zarr.open(mip_path, mode='w', shape=mip_shape, chunks=(1, 1) + mip_shape[-2:], dtype=proj_da.dtype)

    mip_path = os.path.join(root, "built_data", "zarr_image_files", f"{project_name}_mip.zarr")

    # Always write array at root
    with ProgressBar():
        da.to_zarr(proj_da, mip_path, overwrite=True)

    # Always update attrs on the root array
    arr = zarr.open(mip_path, mode="a")
    arr.attrs.update(image_zarr.attrs)

    print("Done.")


