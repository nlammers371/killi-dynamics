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

from src.segmentation import labels_to_contours_nl


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


def combine_tracking_results(
    root: str,
    project_name: str,
    tracking_config: str,
    track_range1: Sequence[int],
    track_range2: Sequence[int],
    handoff_index: int | None = None,
    suffix: str = "",
    well_num: int | None = None,
    par_flag: bool = False,
    n_workers: int | None = None,
    overwrite_flag: bool = False,
):
    """Merge two Ultrack runs that share a hand-off frame."""
    if well_num is None:
        well_num = 0
    if handoff_index is None:
        handoff_index = track_range2[0]

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        n_workers = max(1, total_cpus // 3)

    tracking_name = tracking_config.replace(".txt", "")
    start_i1, stop_i1 = track_range1[0], track_range2[1]
    start_i2, stop_i2 = track_range2

    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path1 = os.path.join(project_path, f"track_{start_i1:04}_{stop_i1:04}{suffix}", "")
    project_sub_path2 = os.path.join(project_path, f"track_{start_i2:04}_{stop_i2:04}{suffix}", "")

    seg_zarr1 = zarr.open(os.path.join(project_sub_path1, "segments.zarr"), mode="r")
    seg_zarr2 = zarr.open(os.path.join(project_sub_path2, "segments.zarr"), mode="r")
    tracks_df1 = pd.read_csv(os.path.join(project_sub_path1, "tracks.csv"))
    tracks_df2 = pd.read_csv(os.path.join(project_sub_path2, "tracks.csv"))

    combined_sub_path = os.path.join(project_path, f"track_{start_i1:04}_{stop_i2:04}_cb", "")
    os.makedirs(combined_sub_path, exist_ok=True)

    hi_rel1 = handoff_index - start_i1
    hi_rel2 = handoff_index - start_i2

    ref_frame1 = seg_zarr1[hi_rel1]
    ref_frame2 = seg_zarr2[hi_rel2]

    props1 = regionprops(ref_frame1)
    props2 = regionprops(ref_frame2)
    label_vec1 = np.array([p.label for p in props1])
    label_vec2 = np.array([p.label for p in props2])

    mask_overlap = np.zeros((len(props1), len(props2)), dtype=np.int32)
    for j, prop2 in enumerate(tqdm(props2, desc="Overlap")):
        coords = prop2.coords
        ref_pixels = ref_frame1[coords[:, 0], coords[:, 1], coords[:, 2]]
        ref_pixels = ref_pixels[ref_pixels > 0]
        lbu, lbc = np.unique(ref_pixels, return_counts=True)
        for label, count in zip(lbu, lbc):
            mask_overlap[label_vec1 == label, j] = count

    row_ind, col_ind = linear_sum_assignment(-mask_overlap)

    track2_ids = np.unique(tracks_df2.loc[tracks_df2["t"] >= hi_rel2, "track_id"].values)
    mapping = {label_vec2[col_ind[i]]: label_vec1[row_ind[i]] for i in range(len(row_ind))}

    start_id = int(ref_frame1.max()) + 1
    new_labels = np.sort(track2_ids[~np.isin(track2_ids, list(mapping.keys()))])
    mapping.update({old: start_id + i for i, old in enumerate(new_labels)})

    tracks_df1_new = tracks_df1.loc[tracks_df1["t"] < hi_rel1].reset_index(drop=True)
    tracks_df2_new = tracks_df2.loc[tracks_df2["t"] >= hi_rel2].reset_index(drop=True)
    tracks_df2_new["track_id_orig"] = tracks_df2_new["track_id"]
    tracks_df2_new["parent_track_id_orig"] = tracks_df2_new["parent_track_id"]

    handoff_tracks = tracks_df1.loc[tracks_df1["t"] == hi_rel1, :]
    mapped_parents = np.zeros(len(tracks_df2_new), dtype=bool)
    for label in label_vec2[col_ind]:
        new_id = mapping[label]
        parent_id = handoff_tracks.loc[handoff_tracks["track_id"] == new_id, "parent_track_id"].to_numpy()[0]
        mask = tracks_df2_new["track_id"] == label
        tracks_df2_new.loc[mask, "parent_track_id"] = parent_id
        mapped_parents |= mask.values

    tracks_df2_new["track_id"] = tracks_df2_new["track_id"].map(mapping).astype(int)
    new_parent_ids = tracks_df2_new.loc[~mapped_parents, "parent_track_id"].map(mapping).astype(int)
    tracks_df2_new.loc[~mapped_parents, "parent_track_id"] = new_parent_ids
    tracks_df2_new["t"] = tracks_df2_new["t"] - tracks_df2_new["t"].min() + hi_rel1

    tracks_df_cb = pd.concat([tracks_df1_new, tracks_df2_new], ignore_index=True)
    tracks_df_cb.to_csv(os.path.join(combined_sub_path, "tracks.csv"), index=False)

    da1 = da.from_array(seg_zarr1, seg_zarr1.chunks)
    da2 = da.from_array(seg_zarr2, seg_zarr1.chunks)
    combined = da.concatenate([da1[:hi_rel1], da2[hi_rel2:]], axis=0)

    max_label = int(da2[-1].max())
    lookup = np.arange(max_label + 1, dtype=da1.dtype)
    for old_label, new_label in mapping.items():
        if old_label <= max_label:
            lookup[old_label] = new_label

    combined_seg_path = os.path.join(combined_sub_path, "segments.zarr")
    try:
        combined.to_zarr(combined_seg_path, overwrite=overwrite_flag)
    except ContainsArrayError:
        print(f"An array already exists at {combined_seg_path}. Skipping write.")

    combined_seg_zarr = zarr.open(combined_seg_path, mode="a")

    if par_flag:
        reindex_run = partial(reindex_mask, seg_zarr=combined_seg_zarr, lookup=lookup, track_df=tracks_df_cb)
        process_map(reindex_run, range(hi_rel1, combined.shape[0]), max_workers=n_workers, chunksize=1)
    else:
        for frame in tqdm(range(hi_rel1, combined.shape[0]), desc="Reindex"):
            reindex_mask(frame, combined_seg_zarr, lookup, track_df=tracks_df_cb)

    return combined_seg_zarr


def check_tracking(
    root: str,
    project_name: str,
    tracking_config: str,
    seg_model: str = "",
    suffix: str = "",
    well_num: int | None = None,
    start_i: int = 0,
    stop_i: int | None = None,
    use_marker_masks: bool = False,
    use_stack_flag: bool = False,
    use_fused: bool = True,
    view_range: Iterable[int] | None = None,
    tracks_only: bool = False,
):
    """Launch a napari viewer showing tracking results."""
    import napari  # lazy import to avoid dependency during tests

    if well_num is not None:
        file_prefix = f"{project_name}_well{well_num:04}"
        subfolder = project_name
    else:
        file_prefix = project_name
        subfolder = ""
        well_num = 0

    tracking_name = tracking_config.replace(".txt", "")

    if use_stack_flag:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_stacks.zarr")
    elif use_fused:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_fused.zarr")
    elif use_marker_masks:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_marker_masks.zarr")
    else:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, project_name, f"{file_prefix}_mask_aff.zarr")

    if use_marker_masks:
        project_name += "_marker"

    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}_{stop_i:04}{suffix}", "")

    mask_zarr = zarr.open(mask_path, mode="r")
    if "voxel_size_um" in mask_zarr.attrs:
        scale_vec = mask_zarr.attrs["voxel_size_um"]
    else:
        scale_vec = [mask_zarr.attrs[k] for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")]

    seg_zarr = zarr.open(os.path.join(project_sub_path, "segments.zarr"), mode="r")
    if view_range is None:
        view_range = np.arange(start_i, stop_i)

    tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))
    tracks_df_plot = tracks_df.loc[np.isin(tracks_df["t"], view_range), :].copy()
    tracks_df_plot.loc[:, "t"] = tracks_df_plot.loc[:, "t"] - view_range[0]

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_tracks(
        tracks_df_plot[["track_id", "t", "z", "y", "x"]],
        name="tracks",
        scale=tuple(scale_vec),
        visible=False,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    if not tracks_only:
        seg_data = seg_zarr[view_range]
        viewer.add_labels(seg_data, name="segments", scale=tuple(scale_vec)).contour = 2

    return viewer


def perform_tracking(
    root: str,
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
    """Execute Ultrack on a mask time series."""
    tracking_name = tracking_config.replace(".txt", "")

    if well_num is not None:
        file_prefix = f"{project_name}_well{well_num:04}"
        subfolder = project_name
    else:
        file_prefix = project_name
        subfolder = ""
        well_num = 0

    if use_stack_flag:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_stacks.zarr")
    elif use_fused:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_fused.zarr")
    elif use_marker_masks:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_marker_masks.zarr")
    else:
        mask_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, f"{file_prefix}_mask_aff.zarr")

    metadata_path = os.path.join(root, "metadata", "tracking")
    mask_store = zarr.open(mask_path, mode="a")
    mask_da = da.from_zarr(mask_store)

    if stop_i is None:
        stop_i = mask_store.shape[0]

    if use_marker_masks:
        project_name += "_marker"

    seg_path = os.path.join(root, "tracking", project_name, "segmentation", f"well{well_num:04}", "")
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}_{stop_i:04}{suffix}", "")
    os.makedirs(project_sub_path, exist_ok=True)

    if not mask_store.attrs and use_fused:
        ref_path = os.path.join(root, "built_data", "mask_stacks", seg_model, f"{file_prefix}_side1_mask_aff.zarr")
        ref_store = zarr.open(ref_path, mode="r")
        for key, value in ref_store.attrs.items():
            mask_store.attrs[key] = value

    if "voxel_size_um" in mask_store.attrs:
        scale_vec = mask_store.attrs["voxel_size_um"]
    else:
        scale_vec = [mask_store.attrs[k] for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")]

    cfg = load_config(os.path.join(metadata_path, tracking_config + ".txt"))
    cfg.data_config.working_dir = project_sub_path

    dstore = zarr.DirectoryStore(os.path.join(seg_path, "detection.zarr"))
    bstore = zarr.DirectoryStore(os.path.join(seg_path, "boundaries.zarr"))
    detection = zarr.open(store=dstore, mode="a", shape=mask_store.shape, dtype=bool, chunks=(1,) + mask_store.shape[1:])
    boundaries = zarr.open(store=bstore, mode="a", shape=mask_store.shape, dtype=np.uint16, chunks=(1,) + mask_store.shape[1:])

    segment_indices = np.arange(start_i, stop_i)
    existing = os.listdir(dstore.path) if os.path.isdir(dstore.path) else []
    written = set(int(fname.split(".")[0]) for fname in existing if fname and fname[0].isdigit())
    segment_indices = np.array(sorted(set(segment_indices) - written))

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

    detection = da.from_zarr(detection)[start_i:stop_i]
    boundaries = da.from_zarr(boundaries)[start_i:stop_i]

    print("Performing tracking...")
    track(cfg, detection=detection, edges=boundaries, scale=scale_vec)

    print("Saving results...")
    tracks_df, graph = to_tracks_layer(cfg)
    tracks_df.to_csv(os.path.join(project_sub_path, "tracks.csv"), index=False)

    segments = tracks_to_zarr(cfg, tracks_df, store_or_path=os.path.join(project_sub_path, "segments.zarr"), overwrite=True)
    print("Done.")
    return segments
