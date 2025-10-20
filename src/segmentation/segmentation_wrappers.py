from __future__ import annotations

import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.segmentation.mask_builders import perform_li_segmentation

def reset_dataset(store, name, shape, dtype, chunks, overwrite=False):
    if name in store:
        if overwrite:
            del store[name]              # remove old array (regardless of shape)
        else:
            return store[name]           # reuse existing one if allowed
    return store.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)

def segment_nuclei_thresh(
    root: Path | str,
    project_name: str,
    nuclear_channel: int | None = None,
    n_workers: int = 1,
    n_thresh: int = 5,
    overwrite: bool = False,
    last_i: int | None = None,
    preproc_flag: bool = True,
    thresh_factors: Sequence[float] | None = None,
):

    par_flag = n_workers is not None and n_workers > 1
    """Segment a full timeseries into hierarchical mask stacks."""
    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        n_workers = max(1, total_cpus // 3)

    root = Path(root)
    zarr_path = root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"

    out_directory = root / "built_data" / "mask_stacks" / "li_segmentation"
    out_directory.mkdir(exist_ok=True)

    li_df = pd.read_csv(out_directory / f"{project_name}_li_thresh_trend.csv")

    image_zarr = zarr.open(zarr_path.as_posix(), mode="r")
    channel_list = image_zarr.attrs["channels"]
    multichannel_flag = len(channel_list) > 1
    if nuclear_channel is None:
        if multichannel_flag:
            nuclear_channel = [
                i
                for i in range(len(channel_list))
                if ("H2B" in channel_list[i]) or ("nls" in channel_list[i])
            ][0]
        else:
            nuclear_channel = 0

    if last_i is None:
        last_i = image_zarr.shape[0]

    mask_store_path = out_directory / f"{project_name}_masks.zarr"
    mode = "a" if not overwrite else "w"
    mask_store = zarr.open(mask_store_path.as_posix(), mode=mode)

    # transfer all metadata from image_zarr to mask store
    for meta_key in image_zarr.attrs.keys():
        mask_store.attrs[meta_key] = image_zarr.attrs[meta_key]

    if multichannel_flag:
        dim_shape = image_zarr.shape[2:]
    else:
        dim_shape = image_zarr.shape[1:]

    stack_zarr = reset_dataset(
        mask_store,
        "thresh_stack",
        shape=(image_zarr.shape[0], n_thresh, *dim_shape),
        dtype=np.uint16,
        chunks=(1, 1, *dim_shape),
        overwrite=overwrite,
    )

    aff_zarr = reset_dataset(
        mask_store,
        "stitched",
        shape=(image_zarr.shape[0], *dim_shape),
        dtype=np.uint16,
        chunks=(1, *dim_shape),
        overwrite=overwrite,
    )

    all_indices = set(range(last_i))
    existing_chunks = os.listdir(Path(mask_store_path) / "stitched")
    written_indices = set(
        int(fname.split(".")[0]) for fname in existing_chunks if fname and fname[0].isdigit()
    )

    empty_indices = np.asarray(sorted(all_indices - written_indices))
    write_indices = np.asarray(list(all_indices)) if overwrite else empty_indices

    for meta_key in image_zarr.attrs.keys():
        stack_zarr.attrs[meta_key] = image_zarr.attrs[meta_key]
        aff_zarr.attrs[meta_key] = image_zarr.attrs[meta_key]

    stack_zarr.attrs["thresh_levels"] = dict({})
    aff_zarr.attrs["thresh_levels"] = dict({})

    li_thresh_call = partial(
        perform_li_segmentation,
        li_df=li_df,
        image_zarr=image_zarr,
        nuclear_channel=nuclear_channel,
        multichannel_flag=multichannel_flag,
        mask_zarr=mask_store,
        n_thresh=n_thresh,
        preproc_flag=preproc_flag,
        thresh_factors=thresh_factors,
    )

    if par_flag:
        seg_flags = process_map(li_thresh_call, write_indices, max_workers=n_workers, chunksize=1)
    else:
        seg_flags = []
        for time_int in tqdm(write_indices, "Conducting segmentation serially..."):
            seg_flag = li_thresh_call(time_int=time_int)
            seg_flags.append(seg_flag)

    return seg_flags


__all__ = ["segment_nuclei_thresh"]