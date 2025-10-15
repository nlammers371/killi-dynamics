"""High-level segmentation orchestration helpers."""
from __future__ import annotations

import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import zarr
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.segmentation.mask_builders import perform_li_segmentation
from src.segmentation.thresholding import calculate_li_thresh


def segment_nuclei(
    root: Path | str,
    project_name: str,
    nuclear_channel: int | None = None,
    n_workers: int | None = None,
    par_flag: bool = False,
    n_thresh: int = 5,
    overwrite: bool = False,
    last_i: int | None = None,
    preproc_flag: bool = True,
    thresh_factors: Sequence[float] | None = None,
):
    """Segment a full timeseries into hierarchical mask stacks."""
    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        n_workers = max(1, total_cpus // 3)

    root = Path(root)
    zarr_path = root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"

    out_directory = root / "built_data" / "mask_stacks"
    out_directory.mkdir(exist_ok=True)

    li_df = pd.read_csv(out_directory / f"{project_name}_li_thresh_trend.csv")

    image_zarr = zarr.open(zarr_path.as_posix(), mode="r")
    channel_list = image_zarr.attrs["Channels"]
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

    multi_mask_zarr_path = out_directory / f"{project_name}_mask_stacks.zarr"
    aff_mask_zarr_path = out_directory / f"{project_name}_mask_aff.zarr"

    if multichannel_flag:
        dim_shape = image_zarr.shape[2:]
    else:
        dim_shape = image_zarr.shape[1:]

    mode = "w" if overwrite else "a"
    stack_zarr = zarr.open(
        multi_mask_zarr_path.as_posix(),
        mode=mode,
        shape=(image_zarr.shape[0],) + (n_thresh,) + dim_shape,
        dtype=np.uint16,
        chunks=(1, 1) + dim_shape,
    )
    aff_zarr = zarr.open(
        aff_mask_zarr_path.as_posix(),
        mode=mode,
        shape=(image_zarr.shape[0],) + dim_shape,
        dtype=np.uint16,
        chunks=(1,) + dim_shape,
    )

    all_indices = set(range(last_i))
    existing_chunks = os.listdir(aff_mask_zarr_path)
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
        stack_zarr=stack_zarr,
        aff_zarr=aff_zarr,
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


def estimate_li_thresh(
    root: Path | str,
    project_name: str,
    interval: int = 125,
    nuclear_channel: int | None = None,
    tol: float | None = None,
    initial_guess: float | None = None,
    start_i: int = 0,
    last_i: int | None = None,
    timeout: float = 60 * 6,
    use_subsample: bool = False,
):
    """Estimate Li thresholds on a coarse grid for a project."""
    root = Path(root)
    zarr_path = root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"

    out_directory = root / "built_data" / "mask_stacks"
    out_directory.mkdir(exist_ok=True)

    image_zarr = zarr.open(zarr_path.as_posix(), mode="r")
    channel_list = image_zarr.attrs["Channels"]
    multichannel_flag = len(channel_list) > 1
    if nuclear_channel is None:
        if multichannel_flag:
            nuclear_channel = [
                i for i in range(len(channel_list)) if ("H2B" in channel_list[i]) or ("nls" in channel_list[i])
            ][0]
        else:
            nuclear_channel = 0

    if last_i is None:
        last_i = image_zarr.shape[0]

    thresh_frames = np.arange(start_i, last_i, interval)
    if len(thresh_frames) > 0:
        thresh_frames[-1] = last_i - 1

    li_vec = []
    frame_vec = []
    for time_int in tqdm(thresh_frames, "Estimating Li thresholds..."):
        if multichannel_flag:
            image_array = np.squeeze(image_zarr[time_int, nuclear_channel, :, :, :]).copy()
        else:
            image_array = np.squeeze(image_zarr[time_int, :, :, :]).copy()
        try:
            _, li_thresh = func_timeout(
                timeout,
                calculate_li_thresh,
                args=(image_array, use_subsample, tol, initial_guess),
            )
            li_vec.append(li_thresh)
            frame_vec.append(time_int)
        except FunctionTimedOut:
            print(f"Function timed out for time: {time_int}")

    li_df_raw = pd.DataFrame(frame_vec, columns=["frame"])
    li_df_raw["li_thresh"] = li_vec

    li_df_raw.to_csv(out_directory / f"{project_name}_li_df.csv", index=False)

    return li_df_raw


__all__ = ["segment_nuclei", "estimate_li_thresh"]

