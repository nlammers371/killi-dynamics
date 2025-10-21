import napari
import os
from tqdm.contrib.concurrent import process_map
from functools import partial
import pandas as pd
import scipy.interpolate
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
from skimage.filters import gaussian
import skimage.io as io
from skimage.transform import resize
from src._Archive.utilities.functions import sphereFit
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import binary_closing, disk, binary_dilation, ball
import json
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian


def stitch_tracks(root, project_name, config_name, overwrite_flag=False, cell_radius=25, min_track_len=10,
                  max_frame_gap=1, min_um_per_frame=2.75, rolling_window=5):

    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")
    # image_dir = os.path.join(root, "built_data", project_name, "")

    print("loading track and mask info...")
    read_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)

    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)
    metadata = json.load(f)
    scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

    # tracks_df, graph = to_tracks_layer(cfg)
    tracks_df = pd.read_csv(os.path.join(read_directory, "tracks.csv"))
    segments = zarr.open(os.path.join(read_directory, "segments.zarr"), mode='r')

    # initialize new track df
    tracks_df_new = tracks_df.copy()
    tracks_df_new["track_id_qc"] = tracks_df_new["track_id"]

    # path to raw cell labels
    raw_mask_zarr = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
    raw_masks = zarr.open(raw_mask_zarr, mode='r')
    # mask_list = sorted(glob.glob(raw_mask_directory + "*.tif"))

    # path to raw image data
    raw_data_zarr = os.path.join(root, "built_data", "exported_image_files", project_name + ".zarr")
    raw_data = zarr.open(raw_data_zarr, mode='r')
    # load sphere fit info
    sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

    # get list of elligible tracks
    track_index, track_lengths = np.unique(tracks_df["track_id"], return_counts=True)

    print("linking 1-to-1 cases...")
    #####
    # first scan for cases where one track disappears and another appears immediately after
    #####

    # calculate start and end times for each track
    start_time_vec = np.empty(track_index.shape)
    start_zyx = np.empty((track_index.size, 3))
    stop_time_vec = np.empty(track_index.shape)
    stop_zyx = np.empty((track_index.size, 3))
    parent_track_id_vec = np.empty(track_index.shape)
    stop_id_vec = np.empty(track_index.shape, dtype=np.uint64)
    for t, track_id in enumerate(track_index):
        cell_df = tracks_df_new.loc[tracks_df_new["track_id_qc"] == track_id, :]
        cell_df.loc[:, "frame_id"] = cell_df.loc[:, "id"]
        cell_df.reset_index(drop=True, inplace=True)
        start_time_vec[t] = np.min(cell_df.loc[:, "t"])
        start_i = np.argmin(cell_df.loc[:, "t"].to_numpy())
        start_zyx[t, :] = np.multiply(cell_df.loc[start_i, ["z", "y", "x"]].to_numpy(), scale_vec)
        stop_time_vec[t] = np.max(cell_df.loc[:, "t"])
        stop_i = np.argmax(cell_df.loc[:, "t"].to_numpy())
        stop_zyx[t, :] = np.multiply(cell_df.loc[stop_i, ["z", "y", "x"]].to_numpy(), scale_vec)

        parent_track_id_vec[t] = cell_df.loc[0, "parent_track_id"]
        stop_id_vec[t] = cell_df.loc[cell_df.shape[0]-1, "frame_id"]

    label_conversion_dict = dict({})
    # get ordering by stop time
    stop_so = np.argsort(stop_time_vec)
    childless_filter = np.asarray([ti not in parent_track_id_vec for ti in track_index])
    parent_filter = parent_track_id_vec == -1
    counter = 0
    counter2 = 0
    for si in stop_so:

        if childless_filter[si]:
            track_id = track_index[si]
            stop_time = stop_time_vec[si]
            # find ids that start in the next frame
            time_filter = (start_time_vec <= (stop_time + 1 + max_frame_gap)) & (start_time_vec > stop_time)

            candidate_filter = time_filter & parent_filter
            candidate_indices = np.where(candidate_filter)[0]
            if np.any(candidate_filter):
                stop_pos = stop_zyx[si, :]
                start_pos = start_zyx[candidate_filter, :]
                distance_vec = np.sqrt(np.sum((start_pos - stop_pos)**2, axis=1))
                distance_filter = distance_vec <= (2*cell_radius)
                counter2 += 1
                if np.sum(distance_filter) == 1:   # for this case I want 1-to-1 cases only
                    track_to_link = track_index[candidate_indices[distance_filter]][0]
                    link_indices = tracks_df_new.index[tracks_df_new["track_id_qc"] == track_to_link]
                    tracks_df_new.loc[link_indices, "track_id_qc"] = track_id
                    tracks_df_new.loc[link_indices[0], "parent_id"] = stop_id_vec[si]
                    tracks_df_new.loc[link_indices, "parent_track_id"] = parent_track_id_vec[si]

                    # update track_index
                    track_index[track_index == track_to_link] = track_id
                    label_conversion_dict[track_to_link] = track_id
                    counter += 1

    # scan for updated labels with children
    from_labels = np.asarray(list(label_conversion_dict.keys()))
    for i in range(len(from_labels)):
        from_id = from_labels[i]
        child_filter = tracks_df_new["parent_track_id"] == from_id
        tracks_df_new.loc[child_filter, "parent_track_id"] = label_conversion_dict[from_id]


    ##############
    # now filter for tracks that are too short
    track_index_new, track_lengths_new = np.unique(tracks_df_new["track_id_qc"], return_counts=True)
    fragment_flags = track_lengths_new <= min_track_len

    # tracks_df_new["fragment_flag"] = fragment_flags

    ##############
    # now filter for tracks that are too sstationary
    track_speed_vec = np.empty(track_index_new.shape)
    for t, track_id in enumerate(track_index_new):
        zyx_array = np.multiply(tracks_df_new.loc[tracks_df_new["track_id_qc"] == track_id, ["z", "y", "x"]].to_numpy(), scale_vec)
        if zyx_array.shape[0] > 1:
            diff_array = np.diff(zyx_array, 1, axis=0)
            dr_vec = np.sqrt(np.sum(diff_array**2, axis=1))
            dr_mean = np.mean(dr_vec)
            track_speed_vec[t] = dr_mean
        else:
            track_speed_vec[t] = 0

    for t, track_id in enumerate(track_index_new):
        tracks_df_new.loc[tracks_df_new["track_id_qc"] == track_id, "avg_disp_um"] = track_speed_vec[t]
        tracks_df_new.loc[tracks_df_new["track_id_qc"] == track_id, "track_len"] = track_lengths_new[t]
        tracks_df_new.loc[tracks_df_new["track_id_qc"] == track_id, "slow_flag"] = track_speed_vec[t] < min_um_per_frame
        tracks_df_new.loc[tracks_df_new["track_id_qc"] == track_id, "fragment_flag"] = fragment_flags[t]


    # rename fields and save
    tracks_df_new["track_id_orig"] = tracks_df_new["track_id"]
    tracks_df_new = tracks_df_new.drop(labels="track_id", axis=1)
    tracks_df_new = tracks_df_new.rename(columns={"track_id_qc": "track_id"})
    
    # add spherical coordinates
    sphere_df["ZM"] = sphere_df["Z"].rolling(rolling_window, center=True, min_periods=1).mean()  # / scale_vec[0]
    sphere_df["YM"] = sphere_df["Y"].rolling(rolling_window, center=True, min_periods=1).mean()  # / scale_vec[1]
    sphere_df["XM"] = sphere_df["X"].rolling(rolling_window, center=True, min_periods=1).mean()  # / scale_vec[2]
    sphere_df["rm"] = sphere_df["r"].rolling(rolling_window, center=True, min_periods=1).mean()  # / scale_vec[0]

    tracks_df_new["zum"] = tracks_df["z"] * scale_vec[0]
    tracks_df_new["yum"] = tracks_df["y"] * scale_vec[1]
    tracks_df_new["xum"] = tracks_df["x"] * scale_vec[2]

    time_index = np.arange(np.max(tracks_df_new["t"]))
    for time_id in time_index:
        sphere_center = sphere_df.loc[sphere_df["t"] == time_id, ["XM", "YM", "ZM"]].to_numpy()
        track_time_ft = tracks_df_new["t"] == time_id
        track_coords_cart = tracks_df_new.loc[track_time_ft, ["xum", "yum", "zum"]].to_numpy()
        track_coords_centered = track_coords_cart - sphere_center

        track_coords_sph = cartesian_to_spherical(track_coords_centered[:, 0], track_coords_centered[:, 1],
                                                  track_coords_centered[:, 2])
        track_array = np.asarray([c.value for c in track_coords_sph]).T
    
        tracks_df_new.loc[track_time_ft, "r"] = track_array[:, 0]
        tracks_df_new.loc[track_time_ft, "phi"] = track_array[:, 1]
        tracks_df_new.loc[track_time_ft, "theta"] = track_array[:, 2]


    # save
    tracks_df_new.to_csv(os.path.join(read_directory, "tracks_cleaned.csv"), index=False)
    sphere_df.to_csv(os.path.join(read_directory, "sphere_df_smoothed.csv"), index=False)

    # update segments

    return {}



if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #

    overwrite_flag = False
    config_name = "tracking_cell.txt" #"tracking_v17.txt"

    stitch_tracks(root, project_name, config_name, overwrite_flag=overwrite_flag, cell_radius=25, min_track_len=10)

    # process_map(
    #     partial(extract_cell_masks, image_list=image_list, project_path=project_path,
    #             overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
    #             file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale),
    #     range(len(image_list)), max_workers=n_workers)
