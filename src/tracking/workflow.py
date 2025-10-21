"""Tracking orchestration and helper utilities."""
from __future__ import annotations

import multiprocessing
import os
from functools import partial
from typing import Iterable, Sequence

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from ultrack import load_config, to_tracks_layer, track, tracks_to_zarr
from zarr.errors import ContainsArrayError
from src.tracking.track_utils import labels_to_contours_nl
from pathlib import Path


def copy_zarr(frame: int, src, dst) -> None:
    """Copy a single frame from ``src`` to ``dst`` Zarr arrays."""
    dst[frame] = np.copy(src[frame])


def reindex_mask(frame: int, seg_zarr, lookup: np.ndarray, track_df: pd.DataFrame | None = None) -> None:
    """Relabel ``seg_zarr[frame]`` according to ``lookup`` in-place."""
    if track_df is not None:
        frame_tracks = track_df.loc[track_df["t"] == frame, "track_id"].values
        props = regionprops(seg_zarr[frame])
        labels = np.array([p.label for p in props])
        if labels.size and np.all(np.isin(labels, frame_tracks)):
            return
    seg_zarr[frame] = lookup[seg_zarr[frame]]


def write_dask_to_zarr(frame: int, out_zarr, dask_array: da.Array, frames_to_reindex: Sequence[int], lookup: np.ndarray) -> None:
    """Write ``dask_array[frame]`` to ``out_zarr`` applying ``lookup`` when needed."""
    arr = dask_array[frame].compute()
    if frame in frames_to_reindex:
        out_zarr[frame] = lookup[arr]
    else:
        out_zarr[frame] = arr


#
#
# def check_tracking(
#     root: str,
#     project_name: str,
#     tracking_config: str,
#     seg_model: str = "",
#     suffix: str = "",
#     well_num: int | None = None,
#     start_i: int = 0,
#     stop_i: int | None = None,
#     use_marker_masks: bool = False,
#     use_stack_flag: bool = False,
#     use_fused: bool = True,
#     view_range: Iterable[int] | None = None,
#     tracks_only: bool = False,
# ):
#     """Launch a napari viewer showing tracking results."""
#     import napari  # lazy import to avoid dependency during tests
#
#     if well_num is not None:
#         file_prefix = f"{project_name}_well{well_num:04}"
#         subfolder = project_name
#     else:
#         file_prefix = project_name
#         subfolder = ""
#         well_num = 0
#
#     tracking_name = tracking_config.replace(".txt", "")
#
#     if use_stack_flag:
#         mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_stacks.zarr")
#     elif use_fused:
#         mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_fused.zarr")
#     elif use_marker_masks:
#         mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_marker_masks.zarr")
#     else:
#         mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, project_name, f"{file_prefix}_mask_aff.zarr")
#
#     if use_marker_masks:
#         project_name += "_marker"
#
#     project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
#     project_sub_path = os.path.join(project_path, f"track_{start_i:04}_{stop_i:04}{suffix}", "")
#
#     mask_zarr = zarr.open(mask_path, mode="r")
#     if "voxel_size_um" in mask_zarr.attrs:
#         scale_vec = mask_zarr.attrs["voxel_size_um"]
#     else:
#         scale_vec = [mask_zarr.attrs[k] for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")]
#
#     seg_zarr = zarr.open(os.path.join(project_sub_path, "segments.zarr"), mode="r")
#     if view_range is None:
#         view_range = np.arange(start_i, stop_i)
#
#     tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))
#     tracks_df_plot = tracks_df.loc[np.isin(tracks_df["t"], view_range), :].copy()
#     tracks_df_plot.loc[:, "t"] = tracks_df_plot.loc[:, "t"] - view_range[0]
#
#     viewer = napari.Viewer(ndisplay=3)
#     viewer.add_tracks(
#         tracks_df_plot[["track_id", "t", "z", "y", "x"]],
#         name="tracks",
#         scale=tuple(scale_vec),
#         visible=False,
#     )
#     viewer.scale_bar.visible = True
#     viewer.scale_bar.unit = "um"
#
#     if not tracks_only:
#         seg_data = seg_zarr[view_range]
#         viewer.add_labels(seg_data, name="segments", scale=tuple(scale_vec)).contour = 2
#
#     return viewer


