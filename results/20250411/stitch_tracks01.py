import pandas as pd
import numpy as np
import os
from ultrack.tracks.gap_closing import close_tracks_gaps
from tqdm import tqdm
from src.tracking import reindex_mask, copy_zarr
import zarr
from functools import partial
from tqdm.contrib.concurrent import process_map
from zarr.sync import ProcessSynchronizer

if __name__ == "__main__":

    # script to stitch tracks after initial tracking. Also updates corresponding seg_zarr's
    # At this point, should have tracked all relevant experiments

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    project_name_list = ["20250311_LCP1-NLSMSC_marker"]#, "20250311_LCP1-NLSMSC", "20250311_LCP1-NLSMSC_marker"]
    track_range_list = ["track_1200_2339"]#, "track_2000_2339", "track_1200_2339"]
    track_config_list = ["tracking_20250328_redux"]#, "tracking_20250328_redux", "tracking_20250328_redux"]

    # set gap closing parameters
    max_gap = 3
    max_radius = 25 * np.sqrt(max_gap)
    scale_vec = np.asarray([3.0, 1.0, 1.0])
    n_workers = 12
    overwrite = True

    for i in tqdm(range(len(project_name_list)), desc="Processing projects", unit="project"):

        project_name = project_name_list[i]
        tracking_config = track_config_list[i]
        tracking_name = track_range_list[i]

        # load mask zarr
        seg_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000", tracking_name, "segments.zarr")
        seg_zarr = zarr.open(seg_path, mode='a')

        # create second zarr file for stitched tracks
        new_seg_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000", tracking_name, "segments_stitched.zarr")

        synchronizer = ProcessSynchronizer('pipeline_org/my_lock')
        seg_zarr_stitched = zarr.open(new_seg_path, mode='a', shape=seg_zarr.shape, chunks=seg_zarr.chunks,
                                      dtype=seg_zarr.dtype, synchronizer=synchronizer)

        # check for previous seg zarr
        all_indices = set(np.arange(seg_zarr.shape[0]))

        # List files directly within zarr directory (recursive search):
        existing_chunks = os.listdir(new_seg_path)

        # Extract time indices from chunk filenames:
        written_indices = set(int(fname.split('.')[0])
                              for fname in existing_chunks if fname[0].isdigit())

        empty_indices = np.asarray(sorted(all_indices - written_indices))

        if overwrite:
            write_indices = np.asarray(list(all_indices))
        else:
            write_indices = empty_indices

        print("Copying indices.") #, write_indices)
        run_copy = partial(copy_zarr, src=seg_zarr, dst=seg_zarr_stitched)
        process_map(run_copy, write_indices, max_workers=n_workers, chunksize=1)

        # load tracks
        track_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000", tracking_name, "tracks.csv")
        tracks_df = pd.read_csv(track_path)

        # call track stitching
        print("Stitching tracks...")
        tracks_df_stitched = close_tracks_gaps(tracks_df, max_gap=max_gap, max_radius=max_radius, scale=scale_vec)

        # get map from old IDS to new IDS
        map_df = tracks_df[["t", "track_id", "id"]].copy()
        map_df = map_df.rename(columns={"track_id": "old_track_id"})
        map_df = map_df.merge(tracks_df_stitched[["t", "track_id", "id"]].drop_duplicates(), on=["t", "id"], how="left")
        check_df = map_df.loc[:, ["old_track_id"]].drop_duplicates()
        map_df = map_df.loc[:, ["track_id", "old_track_id"]].drop_duplicates().reset_index(drop=True)
        if map_df.shape[0] != check_df.shape[0]:
            raise ValueError("Degenerate old -> new mappings found in map_df")

        # relabel frames that come from tracks2
        max_label = np.max(map_df["old_track_id"])
        # Create an identity lookup table (i.e. each label maps to itself)
        lookup = np.zeros((max_label+1,))  #np.arange(max_label + 1, dtype=seg_zarr.dtype)

        # Update the lookup table with your mapping.
        for _, row in map_df.iterrows():
            # It's assumed that old_label is within [0, max_label]
            lookup[row["old_track_id"]] = row["track_id"]

        # reindex
        print("Reindexing segments...")
        run_reindex = partial(reindex_mask, seg_zarr=seg_zarr_stitched, lookup=lookup, track_df=tracks_df_stitched)
        process_map(run_reindex, write_indices, max_workers=n_workers, chunksize=1)

        # save stitched tracks
        print("Saving stitched tracks...")
        tracks_df_stitched.to_csv(
            os.path.join(root, "tracking", project_name, tracking_config, "well0000", tracking_name, "tracks_stitched.csv"),
            index=False)
