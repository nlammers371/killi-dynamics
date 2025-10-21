import os
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
from zarr.errors import ContainsArrayError
import zarr
from skimage.measure import regionprops, regionprops_table
import napari
import dask.array as da
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from functools import partial
from tqdm.contrib.concurrent import process_map
import multiprocessing
from dask.diagnostics import ProgressBar

from src.build_lightsheet.build_utils import labels_to_contours_nl
from src.tracking.workflow import reindex_mask

def props_integrate(frame, mask_zarr, im_zarr, fluo_channel, tracks_df, start_i, scale_vec):

    mask_frame = mask_zarr[frame]
    im_frame = im_zarr[frame + start_i, fluo_channel]

    props = regionprops_table(mask_frame, intensity_image=im_frame, spacing=scale_vec,
        properties=('label', 'intensity_mean', "area"),
    )
    props_df = pd.DataFrame(props).rename(columns={'label':'track_id', 'intensity_mean': 'fluo_mean', 'area': 'nucleus_volume'})
    frame_df = tracks_df.loc[tracks_df["t"] == frame, :]
    frame_df = frame_df.merge(props_df, how="left", on="track_id")

    return frame_df

# define wrapper function for parallel processing
def track_fluorescence_wrapper(root, project_name, tracking_config, suffix="", well_num=0, start_i=0, fluo_channel=None,
                               overwrite=False, par_flag=False, stop_i=None, use_marker_masks=False, use_fused=True,
                               n_workers=None, use_stitched_flag=False):

    # get name
    tracking_name = tracking_config.replace(".txt", "")

    if use_fused:
        image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    else:
        image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
    image_zarr = zarr.open(image_zarr_path, mode='r')

    if use_marker_masks:
        project_name += "_marker"

    scale_vec = tuple(
        [image_zarr.attrs['PhysicalSizeZ'],image_zarr.attrs['PhysicalSizeY'], image_zarr.attrs['PhysicalSizeX']])

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    if fluo_channel is None:
        channels = image_zarr.attrs["Channels"]
        nls_flags = np.asarray(["nls" in name for name in channels])
        if np.sum(nls_flags == 0) == 1:
            fluo_channel = np.where(nls_flags == 0)[0][0]
        elif np.sum(nls_flags == 0) > 1:
            raise Exception("Multiple non-nuclear channels found")
        else:
            raise Exception("No non-nuclear channels found")

    # load tracking masks
    label_path = os.path.join(project_sub_path, "segments.zarr")
    label_path_s = os.path.join(project_sub_path, "segments_stitched.zarr")
    if use_stitched_flag:
        label_path = label_path_s
    seg_zarr = zarr.open(label_path, mode="r")

    if use_stitched_flag:   # use stitched version if present
        tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks_stitched.csv"))
        fluo_df_path = os.path.join(project_sub_path, "tracks_stitched_fluo.csv")
    else:
        tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))
        fluo_df_path = os.path.join(project_sub_path, "tracks_fluo.csv")

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    cat_flag = False
    if os.path.isfile(fluo_df_path) & (~overwrite):
        fluo_df_prev = pd.read_csv(fluo_df_path)
        extant_indices = set(np.unique(fluo_df_prev["t"].values))
        write_indices = np.asarray(set(np.arange(seg_zarr.shape[0])) - extant_indices)
        cat_flag = True
    else:
        write_indices = np.arange(seg_zarr.shape[0])

    fluo_run = partial(props_integrate, mask_zarr=seg_zarr, im_zarr=image_zarr, fluo_channel=fluo_channel,
                       tracks_df=tracks_df, start_i=start_i, scale_vec=scale_vec)
    if par_flag:
        print("Using parallel processing")
        # Use process_map for parallel processing
        df_list = process_map(fluo_run,
            write_indices,
            max_workers=n_workers,
            chunksize=1
        )
    else:
        print("Using sequential processing")
        # Sequential processing
        df_list = []
        for t in tqdm(write_indices):
            df = fluo_run(t)
            df_list.append(df)

    if cat_flag:
        df_list += [fluo_df_prev]
    tracks_df_fluo = pd.concat(df_list, ignore_index=True)
    tracks_df_fluo.to_csv(fluo_df_path, index=False)

    return True


# Note: this assumes that the two tracking results contain predominanty the same objects. Will not work well otherwise
def concatenate_tracking_results(track_folder1, track_folder2, out_folder, track_range1, track_range2,
                                 handoff_index=None, par_flag=False, n_workers=None, overwrite_flag=False,
                                 stitch_suffix=""):

    if handoff_index is None:
        handoff_index = track_range2[0]  # default to end of first range

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    # make out directory
    os.makedirs(out_folder, exist_ok=True)
    # tracking_name = tracking_config.replace(".txt", "")

    start_i1, stop_i1 = track_range1[0], track_range2[1]
    start_i2, stop_i2 = track_range2[0], track_range2[1]

    # stitch_suffix = ""
    # if os.path.isdir(os.path.join(track_folder1, "segments_stitched.zarr")) & \
    #     os.path.isdir(os.path.join(track_folder1, "segments_stitched.zarr")):
        # stitch_suffix = "_stitched"

    # load tracking masks
    seg_zarr1 = zarr.open(os.path.join(track_folder1, "segments" + stitch_suffix + ".zarr"), mode="r")
    seg_zarr2 = zarr.open(os.path.join(track_folder2, "segments" + stitch_suffix + ".zarr"), mode="r")

    # load tracking DFs
    fluo_suffix = ""
    if os.path.isfile(os.path.join(track_folder1, f"tracks{stitch_suffix}_fluo.csv")) & \
            os.path.isfile(os.path.join(track_folder1, f"tracks{stitch_suffix}_fluo.csv")):
        fluo_suffix = "_fluo"
    tracks_df1 = pd.read_csv(os.path.join(track_folder1, "tracks" + stitch_suffix + fluo_suffix + ".csv"))
    tracks_df2 = pd.read_csv(os.path.join(track_folder2, "tracks" + stitch_suffix + fluo_suffix + ".csv"))

    # figure out track id mappings
    hi_rel1 = handoff_index - start_i1
    hi_rel2 = handoff_index - start_i2
    ref_frame1 = seg_zarr1[hi_rel1]  # get the reference frame from the first run02_segment
    ref_frame2 = seg_zarr2[hi_rel2]  # get the reference frame from the second run02_segment

    # tracks_df1_s = pd.read_csv(os.path.join(track_folder1, "tracks" + stitch_suffix + ".csv"))
    # tracks_df2_s = pd.read_csv(os.path.join(track_folder2, "tracks" + stitch_suffix + ".csv"))

    ###########
    # First look for masks that can be merged
    ###########
    # generate array of track mask overlaps
    props1 = regionprops(ref_frame1)
    props2 = regionprops(ref_frame2)

    label_vec1 = np.array([prop.label for prop in props1])  # labels in the first frame
    label_vec2 = np.array([prop.label for prop in props2])

    mask_overlap_array = np.zeros((len(props1), len(props2)), dtype=np.int32)
    for i, prop2 in enumerate(tqdm(props2)):
        coords2 = prop2.coords
        ref1_pixels = ref_frame1[coords2[:, 0], coords2[:, 1], coords2[:, 2]]  # get the pixels in the first frame
        ref1_pixels = ref1_pixels[ref1_pixels > 0]  # only consider valid pixels (greater than 0)
        lbu, lbc = np.unique(ref1_pixels,  return_counts=True)
        for l, lb in enumerate(lbu):
            # iou = lbc[l] / (area_vec2[i] + area_vec1[label_vec1 == lb])  # calculate the intersection over union (IoU) for this label
            mask_overlap_array[label_vec1 == lb, i] = lbc[l]  # fill the overlap array

    # remove rows and cols with no overlap
    col_filter = np.max(mask_overlap_array, axis=0) > 0  # filter for columns with overlap
    row_filter = np.max(mask_overlap_array, axis=1) > 0  # filter for rows with overlap
    if np.mean(col_filter) < 0.9:
        print("Warning: less than 90% of the labels in the second frame overlap with the first frame. Concatenation may not be appropriate.")

    label_vec1 = label_vec1[row_filter]  # filter the labels in the first frame
    label_vec2 = label_vec2[col_filter]  # filter the labels in the second frame

    mask_overlap_array = mask_overlap_array[row_filter, :][:, col_filter]  # filter the overlap array

    # solve for optimal mapping
    cost_matrix = -mask_overlap_array

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # first, set mapping for those tracks that connect directly to prior tracks
    track2_label_index = np.unique(tracks_df2.loc[tracks_df2["t"] >=hi_rel2, "track_id"].values)   # unique track ids in the second frame
    track1_label_map = dict({})
    for i in range(len(row_ind)):
        track1_label_map[label_vec2[col_ind[i]]] = label_vec1[row_ind[i]]  # map the track ids from the first frame to the second

    # now, assign new labels to tracks that are new to the second piece
    start_id = np.max(ref_frame1) + 1
    ids_sorted = np.sort(track2_label_index[~np.isin(track2_label_index, list(track1_label_map.keys()))])  # find the track ids in the second frame that are not in the first
    new_ids = start_id + np.arange(0, len(ids_sorted))
    for i in range(len(ids_sorted)):
        track1_label_map[ids_sorted[i]] = new_ids[i]

    # make new track ID data frames
    tracks_df1_new = tracks_df1.copy()
    tracks_df1_new = tracks_df1_new.loc[tracks_df1_new["t"] < hi_rel1, :].reset_index(drop=True)  # only keep tracks before the handoff index
    tracks_df2_new = tracks_df2.copy()
    tracks_df2_new = tracks_df2_new.loc[tracks_df2_new["t"] >= hi_rel2, :].reset_index(drop=True) # only keep tracks after the handoff index
    tracks_df2_new["track_id_orig"] = tracks_df2_new["track_id"].copy()  # keep a copy of the original track ids
    tracks_df2_new["parent_track_id_orig"] = tracks_df2_new["parent_track_id"] .copy()  # keep a copy of the original parent track ids
    tracks_df1_ho = tracks_df1.loc[tracks_df1["t"] == hi_rel1, :]

    # First let's update track info for the continuation tracks. Specificaly, we need to adjust parent info
    ct_labels = label_vec2[col_ind]
    mapped_parent_filter = np.zeros((tracks_df2_new.shape[0],), dtype=bool)  # filter for mapped parent tracks
    for _, lb in enumerate(tqdm(ct_labels)):
        new_id = track1_label_map[lb]  # get the new track id for this label
        parent_id = tracks_df1_ho.loc[tracks_df1_ho["track_id"] == new_id, "parent_track_id"].to_numpy()[0]
        track_ft = (tracks_df2_new["track_id"] == lb)
        tracks_df2_new.loc[track_ft, "parent_track_id"] = parent_id  # update the parent id for the new track
        mapped_parent_filter = mapped_parent_filter | track_ft  # update the filter for mapped parent tracks

    # Now update all other track labels
    new_track_ids = tracks_df2_new["track_id"].map(track1_label_map).values.astype(int)   # [track1_label_map[tri] for tri in tracks_df2_new["track_id"].values]  # get the new track ids for the second frame
    new_track_ids[new_track_ids < 0] = -1
    tracks_df2_new["track_id"] = new_track_ids

    new_parent_ids = tracks_df2_new.loc[~mapped_parent_filter, "parent_track_id"].map(track1_label_map).values.astype(int)
    tracks_df2_new.loc[~mapped_parent_filter, "parent_track_id"] = new_parent_ids

    # update frame index in tracks2
    tracks_df2_new["t"] = tracks_df2_new["t"] - np.min(tracks_df2_new["t"]) + hi_rel1

    # combine tracks
    tracks_df_cb = pd.concat([tracks_df1_new, tracks_df2_new], ignore_index=True)
    tracks_df_cb.to_csv(os.path.join(out_folder,  "tracks" + stitch_suffix + fluo_suffix + ".csv"), index=False)

    # create new segments zarr for combined results
    combined_seg_zarr_path = os.path.join(out_folder, "segments" + stitch_suffix + ".zarr")

    # Wrap them as Dask arrays. (Assuming their chunk sizes are already appropriate.)
    da1 = da.from_array(seg_zarr1, seg_zarr1.chunks)
    da2 = da.from_array(seg_zarr2, seg_zarr1.chunks)

    # Concatenate them along the desired axis (for example, axis 0).
    combined = da.concatenate([da1[:hi_rel1], da2[hi_rel2:]], axis=0)

    # relabel frames that come from tracks2
    max_label = np.max(da2[-1])
    # Create an identity lookup table (i.e. each label maps to itself)
    lookup = np.arange(max_label + 1, dtype=da1.dtype)

    # Update the lookup table with your mapping.
    for old_label, new_label in track1_label_map.items():
        # It's assumed that old_label is within [0, max_label]
        lookup[old_label] = new_label

    try:
        with ProgressBar():
            combined.to_zarr(combined_seg_zarr_path, overwrite=overwrite_flag)
    except ContainsArrayError:
        print(f"An array already exists at {combined_seg_zarr_path}. Skipping writing and continuing.")

    # reopen
    combined_seg_zarr = zarr.open(combined_seg_zarr_path, mode='a')

    if par_flag:
        reindex_run = partial(reindex_mask, seg_zarr=combined_seg_zarr, lookup=lookup, track_df=tracks_df_cb)
        process_map(reindex_run, range(hi_rel1, combined.shape[0]), max_workers=n_workers, chunksize=1)
    else:
        # Apply the lookup table to relabel the combined array.
        for frame in tqdm(range(hi_rel1, combined.shape[0]), "Reindexing run02_segment labels"):
            reindex_mask(frame=frame, seg_zarr=combined_seg_zarr, lookup=lookup, track_df=tracks_df_cb)

    return {}