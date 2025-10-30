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
from src.data_io.zarr_utils import open_experiment_array

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
    segment_sides_separately: bool = True,
    seg_type: str = "li_segmentation",
    n_workers: int = 1,
    n_thresh: int = 5,
    overwrite: bool = False,
    last_i: int | None = None,
    preproc_flag: bool = True,
    thresh_factors: Sequence[float] | None = None,
):
    par_flag = n_workers is not None and n_workers > 1
    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        n_workers = max(1, total_cpus // 3)

    root = Path(root)
    mask_store_path = root / "segmentation" / seg_type / f"{project_name}_masks.zarr"
    mask_store = zarr.open_group(mask_store_path, mode="a")

    # ---- Load smoothed Li thresholds once ----
    li_df = pd.read_csv(mask_store_path / "thresholds" / "li_thresh_trend.csv")

    # ---- Define threshold range table ----
    if thresh_factors is None:
        thresh_factors = [0.9, 1.1]
    n_thresh = (n_thresh // 2) * 2 + 1
    thresh_range = np.linspace(thresh_factors[0], thresh_factors[1], n_thresh)
    thresh_range_table = li_df.copy()
    for i, factor in enumerate(thresh_range):
        thresh_range_table[f"thresh_{i:02}"] = thresh_range_table["li_thresh"] * factor
    thresh_range_table.to_csv(mask_store_path / "thresholds" / "li_thresh_levels.csv")

    # ---- Load image arrays ----
    if segment_sides_separately:
        # discover all sides in the store
        image_zarr, image_store_path, _resolved_side = open_experiment_array(root, project_name)
        side_keys = sorted([k["name"] for k in image_zarr.attrs["sides"]])
        # arr_list = []
        # for key in side_keys:
        #     arr, _, _ = open_experiment_array(root, project_name, side=key)
        #     path_list.append(arr)
    else:
        image_zarr, image_store_path, resolved_side = open_experiment_array(root, project_name)
        # arr_list = [arr]
        side_keys = [resolved_side]

    # ---- Process each side ----
    for side_key in side_keys:
        print(f"Processing side: {side_key}")

        # create or open the subgroup for this side
        side_group = mask_store.require_group(side_key)

        # determine nuclear channel
        channel_list = image_zarr.attrs["channels"]
        multichannel_flag = len(channel_list) > 1
        if nuclear_channel is None:
            if multichannel_flag:
                nuclear_channel = [
                    i for i, ch in enumerate(channel_list)
                    if ("H2B" in ch.upper()) or ("NLS" in ch.upper())
                ][0]
            else:
                nuclear_channel = 0

        if last_i is None:
            last_i = image_zarr.shape[0]

        # copy image metadata to side group
        for meta_key, val in image_zarr.attrs.items():
            side_group.attrs[meta_key] = val

        if multichannel_flag:
            dim_shape = image_zarr.shape[2:]
        else:
            dim_shape = image_zarr.shape[1:]

        # --- define datasets within side group ---
        stack_zarr = reset_dataset(
            side_group,
            "thresh_stack",
            shape=(image_zarr.shape[0], n_thresh, *dim_shape),
            dtype=np.uint16,
            chunks=(1, 1, *dim_shape),
            overwrite=overwrite,
        )

        aff_zarr = reset_dataset(
            side_group,
            "stitched",
            shape=(image_zarr.shape[0], *dim_shape),
            dtype=np.uint16,
            chunks=(1, *dim_shape),
            overwrite=overwrite,
        )

        # figure out which frames need to be written
        all_indices = set(range(last_i))
        stitched_dir = Path(mask_store_path) / side_key / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        existing_chunks = os.listdir(stitched_dir)
        written_indices = {
            int(fname.split(".")[0])
            for fname in existing_chunks
            if fname and fname[0].isdigit()
        }

        empty_indices = np.asarray(sorted(all_indices - written_indices))
        write_indices = np.asarray(list(all_indices)) if overwrite else empty_indices

        # propagate attrs
        for meta_key, val in image_zarr.attrs.items():
            stack_zarr.attrs[meta_key] = val
            aff_zarr.attrs[meta_key] = val

        # segmentation call
        li_thresh_call = partial(
                                perform_li_segmentation,
                                image_zarr_path=image_store_path,
                                thresh_range_table=thresh_range_table,
                                mask_zarr_path=mask_store_path,
                                group_key=side_key,
                                nuclear_channel=nuclear_channel,
                                preproc_flag=preproc_flag
                            )

        if par_flag:
            seg_flags = process_map(li_thresh_call, write_indices, max_workers=n_workers, chunksize=1)
        else:
            seg_flags = []
            for time_int in tqdm(write_indices, desc=f"Segmenting {side_key} serially..."):
                seg_flag = li_thresh_call(time_int=time_int)
                seg_flags.append(seg_flag)

    return seg_flags


__all__ = ["segment_nuclei_thresh"]