import numpy as np
import os
import glob2 as glob
from skimage.transform import resize
from tqdm import tqdm
from src.utilities.functions import path_leaf
from tqdm.contrib.concurrent import process_map
from functools import partial
import zarr
import re
# from bioio import BioImage
from pathlib import Path
import dask
import numpy as np
import zarr
from bioio import BioImage
from pathlib import Path
from typing import List, Optional, Tuple


_CHUNK_KEY_RE = re.compile(r'^(\d+)\..*$')  # capture leading time index

def _existing_time_indices(zarr_dir):
    """Return a set of time indices that already have at least one chunk written."""
    seen = set()
    try:
        for name in os.listdir(zarr_dir):
            m = _CHUNK_KEY_RE.match(name)
            if m:
                seen.add(int(m.group(1)))
    except FileNotFoundError:
        # No store yet
        pass
    return seen

def get_prefix_list(raw_data_root):
    image_list = sorted(glob.glob(os.path.join(raw_data_root, f"*.czi")))
    stripped_names = [re.sub(r"\(.*\).czi", "", os.path.basename(p)) for p in image_list]
    prefix_list = np.unique(stripped_names).tolist()
    prefix_list = [p for p in prefix_list if p != ""]
    return prefix_list



def initialize_zarr_store(
    zarr_path: Path,
    image_list: List[Path],
    resampling_scale: Tuple[float, float, float],
    channel_to_keep: List[bool] | None,
    overwrite_flag: bool = False,
    last_i: Optional[int] = None,
):
    """
    Initialize or reopen a Zarr file for resampled image writing.

    Automatically handles both multi-timepoint CZIs and "list export" (one file per timepoint) cases.

    Parameters
    ----------
    zarr_path : Path
        Output path for the Zarr store.
    image_list : list of Path
        List of image file paths. For multi-timepoint CZIs, only one element is expected.
    resampling_scale : tuple
        Target physical spacing (ZYX) in microns.
    channel_to_use : int, optional
        If provided, extract only this channel index.
    channels_to_keep : list[int], optional
        Boolean mask or list of channel indices to retain.
    overwrite_flag : bool
        If True, overwrite existing data.
    last_i : int, optional
        Optionally truncate to a fixed number of timepoints.

    Returns
    -------
    zarr_file : zarr.Array
        Opened (and possibly initialized) Zarr dataset ready for writing.
    indices_to_write : list[int]
        Indices of timepoints still needing to be written.
    """

    # ---------------------------------------------------------
    # 1. Inspect first image for metadata
    # ---------------------------------------------------------
    imObject = BioImage(image_list[0])
    dims = imObject.dims
    dims_str = dims.order.upper()
    channel_names = getattr(imObject, "channel_names", None)

    # Determine whether this file contains multiple timepoints
    if "T" in dims_str:
        n_timepoints = dims["T"][0]
        time_stack_flag = True
        n_files = 1
    else:
        n_timepoints = len(image_list)
        time_stack_flag = False
        n_files = n_timepoints

    if last_i is not None:
        n_timepoints = min(n_timepoints, int(last_i))

    # ---------------------------------------------------------
    # 2. Determine channel behavior
    # ---------------------------------------------------------
    # Load one representative volume lazily
    multichannel_flag = False
    nCh = 1
    if "C" in dims_str:
        nCh = dims["C"][0]
        if channel_to_keep is not None:
            nCh = sum(channel_to_keep)
        multichannel_flag = nCh > 1  # (C, Z, Y, X)

    space_shape = tuple([dims[k][0] for k in ["Z", "Y", "X"]])
    # ---------------------------------------------------------
    # 3. Handle spatial scaling and output shape
    # ---------------------------------------------------------
    raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)

    # Convert from meters to microns if necessary
    if np.max(raw_scale_vec) <= 1e-5:
        raw_scale_vec *= 1e6

    # Resampling factors
    if raw_scale_vec[0] != resampling_scale[0]:
        raise ValueError("Z resampling not supported; input data Z spacing differs from target.")

    rs_factors = np.divide(raw_scale_vec, resampling_scale)
    out_spatial = tuple(np.round(np.multiply(space_shape, rs_factors)).astype(int))
    if multichannel_flag:
        inner_shape = (nCh,) + out_spatial  # (C, Z, Y, X)
        chunks = (1, 1) + inner_shape[1:]
    else:
        inner_shape = out_spatial  # (Z, Y, X)
        chunks = (1,) + inner_shape

    shape_out = (n_timepoints,) + inner_shape
    dtype = np.uint16
    mode = "w" if overwrite_flag else "a"

    # ---------------------------------------------------------
    # 4. Initialize or reopen Zarr
    # ---------------------------------------------------------
    zarr_file = zarr.open(
        zarr_path,
        mode=mode,
        shape=shape_out,
        dtype=dtype,
        chunks=chunks,
    )

    # ---------------------------------------------------------
    # 5. Identify which timepoints still need writing
    # ---------------------------------------------------------
    if overwrite_flag:
        indices_to_write = list(range(n_timepoints))
    else:
        already = _existing_time_indices(zarr_path)
        already = {i for i in already if 0 <= i < n_timepoints}
        indices_to_write = sorted(set(range(n_timepoints)) - already)

    # ---------------------------------------------------------
    # 6. Metadata summary
    # ---------------------------------------------------------
    summary = dict(
        dims=dims,
        shape=shape_out,
        pixel_size_um=resampling_scale.tolist(),
        channels=channel_names,
        time_stack=time_stack_flag,
        n_files=n_files,
    )
    print(f"[initialize_zarr_store] Summary:\n{summary}")

    return zarr_file, indices_to_write, time_stack_flag


def write_zarr(t,
               zarr_file,
               image_list,
               time_stack_flag,
               file_prefix,
               resampling_scale,
               tres=None,
               channel_names=None,
               channels_to_keep=None):

    if not time_stack_flag:
        im_path = Path(image_list[t])
        f_string = im_path.name
        time_string = f_string.replace(file_prefix, "")
        time_string = time_string.replace(".czi", "")
        time_point = int(time_string[1:-1]) - 1
        imObject = BioImage(im_path)
        arr = imObject.get_image_dask_data("CZYX")
    else:
        time_point = t
        im_path = Path(image_list[0])
        imObject = BioImage(im_path)
        arr = imObject.get_image_dask_data("CZYX", T=t)

    # ---------------------------------------------------------
    # 2) Channel handling
    # ---------------------------------------------------------
    Cdim = arr.shape[0]
    if channels_to_keep is not None:
        keep = np.asarray(channels_to_keep, dtype=bool)
        arr = arr[keep, ...]
        Cdim = arr.shape[0]

    multichannel = arr.ndim == 4 and Cdim > 1
    if not multichannel:
        arr = np.squeeze(arr)

    # ---------------------------------------------------------
    # 3) Spatial resampling
    # ---------------------------------------------------------
    target_shape = zarr_file.shape[1:]  # (C,Z,Y,X) or (Z,Y,X)
    zdim = target_shape[1] if multichannel else target_shape[0]
    yx_shape = target_shape[-2:]

    image_data_rs = np.empty(target_shape, dtype=np.uint16)

    for c in range(Cdim):
        channel_data = arr[c].compute()  # load only this channel lazily
        nz = channel_data.shape[0]

        # Resample each Z plane individually
        resized_planes = np.empty((zdim, *yx_shape), dtype=np.uint16)
        for z in range(min(zdim, nz)):
            resized_planes[z] = np.round(
                resize(
                    channel_data[z],
                    yx_shape,
                    preserve_range=True,
                    order=1,
                    anti_aliasing=True,
                )
            ).astype(np.uint16)

        if multichannel:
            image_data_rs[c] = resized_planes
        else:
            image_data_rs = resized_planes

    # ---------------------------------------------------------
    # 4) Metadata extraction and first-frame write
    # ---------------------------------------------------------
    if t == 0:
        px_um = np.asarray(imObject.physical_pixel_sizes)
        if np.max(px_um) <= 1e-5:
            px_um *= 1e6
        pixel_size_um = tuple(px_um.tolist())  # (Z, Y, X)

        if channel_names is None:
            if multichannel:
                channel_names = getattr(imObject, "channel_names", None)
                if channel_names is None:
                    channel_names = [f"channel{c:02}" for c in range(Cdim)]
            else:
                channel_names = ["channel00"]

        metadata = {
            "dim_order": "TCZYX" if multichannel else "TZYX",
            "n_timepoints": int(zarr_file.shape[0]),
            "voxel_size_um": tuple(map(float, resampling_scale)),  # tuple ok
            "time_resolution_s": float(tres) if tres is not None else None,
            "raw_voxel_scale_um": list(map(float, pixel_size_um)),
            "channels": list(map(str, channel_names)),
            "source_file": str(im_path),
        }
        zarr_file.attrs.update(metadata)

        print(f"[write_zarr] Metadata: {metadata}")

    # ---------------------------------------------------------
    # 5) Write data to Zarr
    # ---------------------------------------------------------
    zarr_file[time_point] = image_data_rs


def export_czi_to_zarr(raw_data_root: Path | str,
                       file_prefix: str,
                       project_name: str,
                       tres: float = None,
                       save_root: Path | str | None = None,
                       last_i=None, overwrite_flag=False,
                       resampling_scale=None, channel_names=None,
                       channels_to_keep=None,
                       n_workers=8):

    par_flag = n_workers > 1

    # clean up paths
    raw_data_root = Path(raw_data_root)
    if save_root is None:
        save_root = raw_data_root.root
    else:
        save_root = Path(save_root)

    if resampling_scale is None:
        resampling_scale = np.asarray([3, 0.85, 0.85])

    if channels_to_keep is not None:
        if channel_names is None:
            raise ValueError("channel_names must be provided if channels_to_keep is used.")
        if len(channels_to_keep) != len(channel_names):
            raise ValueError("channels_to_keep must match length of channel_names.")
        channel_names = [ch for ch, keep in zip(channel_names, channels_to_keep) if keep]

    zarr_path = save_root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"
    if not os.path.isdir(zarr_path):
        os.makedirs(zarr_path)

    # determine what kind of export we are dealing with
    raw_path = raw_data_root / "raw_image_data"/ project_name
    image_list = sorted(raw_path.glob(f"{file_prefix}(*).czi"))
    if len(image_list) == 0:
        image_list = sorted(raw_path.glob(f"{file_prefix}*.czi"))

    # initialize
    zarr_file, times_to_write, time_stack_flag = initialize_zarr_store(zarr_path,
                                                      image_list=image_list,
                                                      resampling_scale=resampling_scale,
                                                      channel_to_keep=channels_to_keep,
                                                      overwrite_flag=overwrite_flag,
                                                      last_i=last_i)

    times_to_write = np.asarray(times_to_write)
    if last_i is not None:
        times_to_write = times_to_write[times_to_write <= last_i]

    # get list of image timestamps and use this to figure out which indices to write
    if not time_stack_flag:
        image_time_stamps = []
        for _, image_path in enumerate(image_list):
            f_string = path_leaf(image_path)
            time_string = f_string.replace(file_prefix, "")
            time_string = time_string.replace(".czi", "")
            image_time_stamps.append(int(time_string[1:-1]) - 1)
    else:
        image_time_stamps = np.arange(zarr_file.shape[0])

    indices_to_write = np.where(np.isin(np.asarray(image_time_stamps), times_to_write))[0]

    run_write_zarr = partial(write_zarr,
                            zarr_file=zarr_file,
                            image_list=image_list,
                            time_stack_flag=time_stack_flag,
                            channel_names=channel_names,
                            channels_to_keep=channels_to_keep,
                            file_prefix=file_prefix, tres=tres,
                            resampling_scale=resampling_scale)
    if par_flag:
        process_map(run_write_zarr,
            indices_to_write, max_workers=n_workers, chunksize=1)
    else:
        for i in tqdm(indices_to_write, "Exporting raw images to zarr..."):
            run_write_zarr(i)

    print("Done.")



if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([1.5, 1.5, 1.5])
    tres = 123.11  # time resolution in seconds

    # set path parameters
    raw_data_root = "D:\\Syd\\240611_EXP50_NLS-Kikume_24hpf_2sided_NuclearTracking\\" #"D:\\Syd\\240219_LCP1_67hpf_to_"
    file_prefix_vec = ["E2_2024_11_14__20_21_18_968_G1", "E2_Timelapse_2024_06_11__22_51_41_085_G2"] #"E3_186_TL_start93hpf_2024_02_20__19_13_43_218"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20240611_NLS-Kikume_24hpf_side1", "20240611_NLS-Kikume_24hpf_side2"]
    overwrite = False

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(
            raw_data_root,
            file_prefix,
            project_name,
            save_root,
            tres,
            par_flag=True,
            channel_to_use=0,
            overwrite_flag=overwrite,
        )


__all__ = [
    "get_prefix_list",
    "initialize_zarr_store",
    "write_zarr",
    "export_czi_to_zarr",
]
