import numpy as np
from pathlib import Path
import pandas as pd
from skimage.measure import regionprops_table
from src.data_io.zarr_io import open_mask_array, open_experiment_array
from functools import partial
import re
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def integrate_fluorescence(t, root, project, seg_type, side_key, tracked_mask_path, fluo_channel):
    """
    Integrates fluorescence values for a given time point and mask.

    Parameters:
    - t: Time index.
    - image_zarr: Zarr array of images (4D: time, z, y, x).
    - mask_zarr: Zarr array of masks (4D: time, z, y, x).
    - fluo_channel: Index of the fluorescence channel to use.

    Returns:
    - Integrated fluorescence value for the given time point.
    """

    # get zarr arrays
    image_zarr, _, _ = open_experiment_array(root=root, project_name=project)
    mask_zarr, mask_store_path, _ = open_mask_array(root=root, project_name=project, side=side_key)
    out_folder = mask_store_path / "fluorescence_data"
    out_folder.mkdir(out_folder, exist_ok=True)
    mask_frame = np.squeeze(mask_zarr[t])
    im_frame = np.squeeze(image_zarr[t, fluo_channel])
    meta = image_zarr.attrs

    if "voxel_size_um" not in meta.keys():
        # get scale info
        scale_vec = tuple([meta['PhysicalSizeZ'],
                           meta['PhysicalSizeY'],
                           meta['PhysicalSizeX']])
    else:
        scale_vec = meta["voxel_size_um"]

    mask_vec = mask_frame[mask_frame > 0]
    i_vec = im_frame[mask_frame > 0]
    # get average by label


    # initialize array to store fluorescence values
    props = regionprops_table(mask_frame, im_frame,
                              properties=["label", "centroid", "coords", "intensity_mean"],
                              spacing=scale_vec)
    label_vec = np.array([pr.label for pr in props])
    fluo_df = pd.DataFrame(label_vec, columns=["nucleus_id"])
    fluo_df["frame"] = t
    centroids = np.array([pr.centroid for pr in props])
    if centroids.size > 0:
        fluo_df[["z", "y", "x"]] = centroids
        # fluo_mean = ndi_mean(im_frame, labels=mask_frame, index=label_vec)
        # fluo_df["mean_fluo"] = fluo_mean
        for p, pr in enumerate(props):
            # get array indices corresponding to nucleus mask from props
            coords = pr.coords
            # get the mean fluorescence value for each nucleus using coordinates
            fluo_df.loc[fluo_df["nucleus_id"] == pr.label, "mean_fluo"] = np.mean(im_frame[coords[:, 0], coords[:, 1], coords[:, 2]])
    else:
        fluo_df = pd.DataFrame(columns=["nucleus_id", "frame", "z", "y", "x", "mean_fluo"])
    # save
    output_file = out_folder / f"fluorescence_data_frame{t:04}.csv"
    fluo_df.to_csv(output_file, index=False)

    return fluo_df

# def transfer_fluorescence(frame_i, root, project_name, seg_type, tracked_mask_path, start_i):
#
#     # load
#     mask_zarr, mask_store_path, _ = open_mask_array(root=root, project_name=project_name, seg_type=seg_type)
#     fluo_df_frame = pd.read_csv(mask_store_path / "fluorescence_data" / f"fluorescence_data_frame{frame_i:04}.csv")
#     tracked_mask_zarr = zarr.open(tracked_mask_path, mode='r')
#
#     # do the label transfer
#     # fluo_df_frame = fluo_df.loc[fluo_df["frame"] == frame_i, :]
#     tracked_frame = tracked_mask_zarr[frame_i - start_i]
#     # props = regionprops(tracked_mask_zarr[frame_i - start_i])
#     # label_vec = np.array([pr.label for pr in props])
#     # fluo_vec = np.zeros(len(label_vec), dtype=np.float32)  # initialize fluorescence vector
#     mask_frame = mask_zarr[frame_i]
#
#     fluo_lookup = fluo_df_frame.set_index("nucleus_id")["mean_fluo"]
#
#     map_to_vec = tracked_mask_zarr[tra]
#     # # for pr in props:
#     # for p, pr in enumerate(props):
#     #     coords = pr.coords  # get array indices corresponding to nucleus mask from props
#     #     lb, counts = np.unique(mask_frame[coords[:, 0], coords[:, 1], coords[:, 2]], return_counts=True)  # get labels in the mask
#     #     counts = counts[lb > 0]
#     #     weights = counts / np.sum(counts)
#     #     lb = lb[lb > 0]  # remove background label
#     #
#     #     fluo_vals = fluo_lookup.loc[lb].values
#     #     fluo_vec[p] = np.dot(weights, fluo_vals)
#
#     temp_df = pd.DataFrame(label_vec, columns=["track_id"])
#     temp_df["t"] = frame_i - start_i
#     temp_df["mean_fluo"] = fluo_vec
#
#     return temp_df

# def transfer_fluorescence_wrapper(root, project_name, fused_flag=True, tracking_config=None, tracking_range=None,
#                                   suffix="", well_num=0, overwrite=False, use_markers_flag=False, n_workers=None, par_flag=False):
#
#     """
#     :param root:
#     :param project_name:
#     :param fused_flag:
#     :param tracking_config:
#     :param tracking_range:
#     :param suffix:
#     :param well_num:
#     :param overwrite:
#     :return:
#     """
#
#     if n_workers is None:
#         total_cpus = multiprocessing.cpu_count()
#         # Limit yourself to 33% of CPUs (rounded down, at least 1)
#         n_workers = max(1, total_cpus // 3)
#
#     # get path to fluo files
#     if use_markers_flag:
#         fluo_path = os.path.join(root, "built_data", "fluorescence_data", project_name + "_markers", "")
#     else:
#         fluo_path = os.path.join(root, "built_data", "fluorescence_data", project_name, "")
#     fluo_df_path_list = sorted(glob(fluo_path + "*.csv"))
#     fluo_df_list = []
#     for fluo_file in tqdm(fluo_df_path_list, "Loading fluorescence data files..."):
#         fluo_df = pd.read_csv(fluo_file)
#         fluo_df_list.append(fluo_df)
#
#     # combine all fluorescence dataframes
#     # fluo_df = pd.concat(fluo_df_list, ignore_index=True)
#
#     # path to raw masks
#     if use_markers_flag:
#         mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_marker_masks.zarr")
#     elif fused_flag:
#         mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
#     else:
#         mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_aff.zarr")
#
#     # load raw mask zarr
#     mask_zarr = zarr.open(mask_zarr_path, mode='r')
#
#     #############################
#     # load tracked zarr mask
#     # get name
#     tracking_name = tracking_config.replace(".txt", "")
#     start_i, stop_i = tracking_range[0], tracking_range[1]
#
#     fluo_df_full = pd.concat(fluo_df_list, ignore_index=True)
#
#     if use_markers_flag:
#         project_name += "_marker"
#
#     # set output path for tracking results
#     project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
#     project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")
#
#     # load tracking masks
#     label_path = os.path.join(project_sub_path, "segments.zarr")
#     tracked_mask_zarr = zarr.open(label_path, mode='r')
#
#     # load dataframe of track info
#     tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))
#     # tracks_df_fluo = tracks_df.copy()
#
#     transfer_run = partial(transfer_fluorescence, start_i=start_i, fluo_df=fluo_df_full,
#                            tracked_mask_zarr=tracked_mask_zarr, mask_zarr=mask_zarr)
#
#     if par_flag:
#         tr_df_list = process_map(transfer_run, range(start_i, stop_i), max_workers=n_workers, chunksize=1)
#     else:
#         tr_df_list = []
#         for f in tqdm(range(start_i, stop_i), desc="Processing frames for fluorescence transfer..."):
#             tr_df = transfer_run(f)
#             tr_df_list.append(tr_df)
#
#     transfer_df = pd.concat(tr_df_list, ignore_index=0)
#     tracks_df = tracks_df.merge(transfer_df, on=["track_id", "t"], how="left")  # merge fluorescence values into the tracks dataframe
#
#     tracks_df.to_csv(os.path.join(project_sub_path, "tracks_fluo.csv"), index=False)  # save the updated tracks dataframe


# define wrapper function for parallel processing
def nuclear_fluorescence_wrapper(root: Path,
                                 project_name: str,
                                 fluo_channel: int,
                                 group_key: str | None = None,
                                 tracking_config: str | None = None,
                                 seg_type: str = "li_segmentation",
                                 overwrite: bool = False,
                                 track_start_i: int = 0,
                                 track_stop_i: int = None,
                                 n_workers: int = 1):


    # open dummy dataset
    mask_zarr, mask_store_path, _ = open_mask_array(root=root, project_name=project_name, side="side_00")
    if track_stop_i is None:
        track_stop_i = mask_zarr.shape[0]

    # determine which frames need to be analyzed
    all_indices = set(range(track_start_i, track_stop_i))

    # List files directly within output directory:
    fluo_dir = mask_store_path / "fluorescence_data"
    existing_frames = set(int(re.search(r"(\d{4})", f.name).group()) for f in fluo_dir.glob("fluorescence_data_frame*.csv"))

    # Extract time indices from chunk filenames:
    if overwrite:
        write_indices = all_indices
    else:
        write_indices = all_indices - existing_frames

    # get path to tracking masks if needed
    if tracking_config is not None:
        track_path = root / "tracking" / project_name / tracking_config / f"track_{track_start_i:04}_{track_stop_i:04}"
        seg_path = track_path / "segments.zarr"
    else:
        seg_path = None
    # build function call
    run_integration = partial(integrate_fluorescence,
                              root=root,
                              project=project_name,
                              seg_type=seg_type,
                              side_key=group_key,
                              fluo_channel=fluo_channel,
                              tracked_mask_path=seg_path)

    if n_workers > 1:
        print("Using parallel processing")
        # Use process_map for parallel processing
        results = process_map(run_integration,
                              write_indices,
                              max_workers=n_workers,
                              chunksize=1
                             )
    else:
        print("Using sequential processing")
        # Sequential processing
        results = []
        for t in tqdm(write_indices):
            result = run_integration(t)
            results.append(result)

    return True