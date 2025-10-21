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



def perform_tracking(
    root: Path,
    project_name: str,
    tracking_config: str,
    seg_model: str,
    well_num: int | None = None,
    start_i: int = 0,
    stop_i: int | None = None,
    use_stack_flag: bool = False,
    use_marker_masks: bool = False,
    use_fused: bool = True,
    suffix: str = "",
    par_seg_flag: bool = True,
    last_filter_start_i: int | None = None,
):
    """
    Execute Ultrack on a mask time series.
    """
    # --- setup ---
    tracking_name: str = tracking_config.replace(".txt", "")

    if well_num is not None:
        file_prefix = f"{project_name}_well{well_num:04}"
        subfolder = Path(project_name)
    else:
        file_prefix = project_name
        subfolder = Path()

    # --- mask path selection ---
    mask_dir = root / "built_data" / "mask_stacks" / seg_model / subfolder
    mask_path = mask_dir / f"{file_prefix}_masks.zarr"
    mask_store = zarr.open(mask_path, mode="r")
    # Choose which group or dataset to load
    mask_field = "thresh_stacks" if use_stack_flag else "clean"
    # Access that group
    if mask_field not in mask_store:
        raise KeyError(f"'{mask_field}' not found in {mask_path}")
    # Load the group as a zarr array
    mask_group = mask_store[mask_field]
    mask_da = da.from_zarr(mask_group)


    if stop_i is None:
        stop_i = mask_store.shape[0]

    if use_marker_masks:
        project_name += "_marker"

    # --- define output directories ---
    seg_path = root / "tracking" / project_name / "segmentation"
    project_path = root / "tracking" / project_name / tracking_name
    if well_num is not None:
        seg_path = seg_path / f"well{well_num:04}"
        project_path = project_path / f"well{well_num:04}"

    project_sub_path = project_path / f"track_{start_i:04}_{stop_i:04}{suffix}"
    project_sub_path.mkdir(parents=True, exist_ok=True)

    if "voxel_size_um" in mask_store.attrs:
        scale_vec: list[float] = mask_store.attrs["voxel_size_um"]
    else:  # for backwards compatibility
        scale_vec = [mask_store.attrs[k] for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")]

    # --- load configuration ---
    metadata_path = root / "metadata" / "tracking"
    cfg = load_config(metadata_path / f"{tracking_config}.txt")
    cfg.data_config.working_dir = str(project_sub_path)

    # --- initialize segmentation stores ---
    dstore_path: Path = seg_path / "detection.zarr"
    bstore_path: Path = seg_path / "boundaries.zarr"

    dstore = zarr.DirectoryStore(str(dstore_path))
    bstore = zarr.DirectoryStore(str(bstore_path))

    detection = zarr.open(
        store=dstore,
        mode="a",
        shape=mask_store.shape,
        dtype=bool,
        chunks=(1,) + mask_store.shape[1:],
    )
    boundaries = zarr.open(
        store=bstore,
        mode="a",
        shape=mask_store.shape,
        dtype=np.uint16,
        chunks=(1,) + mask_store.shape[1:],
    )

    # --- find missing frames to process ---
    segment_indices = np.arange(start_i, stop_i)
    existing_files = list(dstore_path.iterdir()) if dstore_path.exists() else []
    written = {int(p.stem) for p in existing_files if p.stem.isdigit()}
    segment_indices = np.array(sorted(set(segment_indices) - written))

    # --- segmentation and boundary extraction ---
    if segment_indices.size > 0:
        detection_da, boundaries_da = labels_to_contours_nl(
            mask_da,
            segment_indices,
            par_flag=par_seg_flag,
            last_filter_start_i=last_filter_start_i,
            scale_vec=scale_vec,
        )
        for t, frame in enumerate(segment_indices):
            detection[frame] = detection_da[t]
            boundaries[frame] = boundaries_da[t]

    # --- tracking ---
    detection_da = da.from_zarr(detection)[start_i:stop_i]
    boundaries_da = da.from_zarr(boundaries)[start_i:stop_i]

    print("Performing tracking...")
    track(cfg, detection=detection_da, edges=boundaries_da, scale=scale_vec)

    # --- save results ---
    print("Saving results...")
    tracks_df, graph = to_tracks_layer(cfg)
    tracks_csv_path: Path = project_sub_path / "tracks.csv"
    tracks_df.to_csv(tracks_csv_path, index=False)

    segments_path: Path = project_sub_path / "segments.zarr"
    segments = tracks_to_zarr(cfg, tracks_df, store_or_path=str(segments_path), overwrite=True)

    print("Done.")
    return segments
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


