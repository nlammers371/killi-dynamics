import pandas as pd
import numpy as np
import os
from ultrack.tracks.gap_closing import close_tracks_gaps
from tqdm import tqdm
from src.nucleus_dynamics.tracking.perform_tracking import reindex_mask, copy_zarr
import zarr
from functools import partial
from tqdm.contrib.concurrent import process_map
from zarr.sync import ProcessSynchronizer



def preprocess_tracks(df, ROLL_W=5, MOVE_THRESH=1.0, OVERLAP_MIN=5, MERGE_DIST=5.0):
    df = df.sort_values(["track_id", "t"]).copy()

    # ============================================================
    # (ii) Add track length (frames)
    # ============================================================
    df["track_len"] = df.groupby("track_id")["t"].transform("count")

    # ============================================================
    # (i) Flag stationary frames using rolling mean step length
    # ============================================================
    # frame-to-frame displacement per track
    df["dx"] = df.groupby("track_id")["x"].diff()
    df["dy"] = df.groupby("track_id")["y"].diff()
    df["dz"] = df.groupby("track_id")["z"].diff()
    df["step_len"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)

    # rolling mean step length (ignore NaNs)
    df["roll_move"] = (
        df.groupby("track_id")["step_len"]
          .rolling(ROLL_W, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["is_stationary"] = df["roll_move"] < MOVE_THRESH

    # track-level summary (optional)
    df["track_mostly_stationary"] = (
        df.groupby("track_id")["is_stationary"]
          .transform(lambda x: x.mean() > 0.7)
    )

    # ============================================================
    # (iii) Fuse duplicate / parallel tracks
    # ============================================================
    # Identify overlapping + spatially co-located tracks
    track_ids = df["track_id"].unique()
    track_bounds = (
        df.groupby("track_id")["t"]
        .agg(["min", "max"])
        .rename(columns={"min": "t0", "max": "t1"})
    )

    # build adjacency list for track-merge graph
    merge_edges = []

    for i, tid1 in enumerate(track_ids):
        t0_1, t1_1 = track_bounds.loc[tid1, ["t0", "t1"]]
        df1 = df[df.track_id == tid1]

        for tid2 in track_ids[i+1:]:
            t0_2, t1_2 = track_bounds.loc[tid2, ["t0", "t1"]]

            # temporal overlap
            overlap_start = max(t0_1, t0_2)
            overlap_end   = min(t1_1, t1_2)
            if overlap_end - overlap_start < OVERLAP_MIN:
                continue

            # spatial proximity check
            df2 = df[df.track_id == tid2]
            merged = df1.merge(df2, on="t", suffixes=("_1","_2"))
            if len(merged) < OVERLAP_MIN:
                continue

            dist = np.sqrt(
                (merged["x_1"]-merged["x_2"])**2
                + (merged["y_1"]-merged["y_2"])**2
                + (merged["z_1"]-merged["z_2"])**2
            )

            if dist.median() < MERGE_DIST:
                merge_edges.append((tid1, tid2))

    # build connected components of merge graph
    # (each component = tracks that belong to one cell)
    adj = {}
    for a, b in merge_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # DFS to get connected components
    visited = set()
    components = []

    for tid in track_ids:
        if tid in visited:
            continue
        stack = [tid]
        comp = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    stack.append(v)
        components.append(sorted(comp))

    # ============================================================
    # Fuse tracks inside each component
    # ============================================================
    fused_rows = []

    for comp in components:
        if len(comp) == 0:
            continue
        elif len(comp) == 1:
            fused_rows.append(df[df.track_id == comp[0]])
            continue

        # choose earliest track ID as representative
        rep = comp[0]

        # pull all rows for all tracks in this component
        sub = df[df.track_id.isin(comp)]

        # fuse by averaging positions per frame
        fused = (
            sub.groupby("t")
               .agg({"x": "mean", "y": "mean", "z": "mean",
                     "step_len": "mean",
                     "roll_move": "mean",
                     "parent_track_id": "max",  # keep highest parent ID
                     "is_stationary": "all",
                     "track_mostly_stationary": "all"})  # any = true if any frame stationary
               .reset_index()
        )
        fused["track_id"] = rep
        fused_rows.append(fused)

    clean_df = pd.concat(fused_rows, ignore_index=True)
    clean_df = clean_df.sort_values(["track_id", "t"]).reset_index(drop=True)

    # recompute track_len after fusion
    clean_df["track_len"] = clean_df.groupby("track_id")["t"].transform("count")

    # clean_df["track_id"] = clean_df["track_id"].astype(int)
    # clean_df.loc[np.isnan(clean_df["parent_track_id"]), "parent_track_id"] = -1
    # clean_df["parent_track_id"] = clean_df["parent_track_id"].astype(int)

    return clean_df

if __name__ == "__main__":

    # script to stitch tracks after initial tracking. Also updates corresponding seg_zarr's
    # At this point, should have tracked all relevant experiments

    # load zarr image file
    root = "E:\\Nick\\killi_immuno_paper\\"

    project_name_list = ["20241126_LCP1-NLSMSC"]#, "20250311_LCP1-NLSMSC", "20250311_LCP1-NLSMSC_marker"]
    track_range_list = ["track_0000_0719"]#, "track_2000_2339", "track_1200_2339"]
    track_config_list = ["tracking_lcp_nuclei"]#, "tracking_20250328_redux", "tracking_20250328_redux"]

    # set gap closing parameters
    max_gap = 3
    max_radius = 35 * np.sqrt(max_gap)
    scale_vec = np.asarray([3.0, 0.8, 0.8])
    n_workers = 12
    overwrite = True

    for i in tqdm(range(len(project_name_list)), desc="Processing projects", unit="project"):

        project_name = project_name_list[i]
        tracking_config = track_config_list[i]
        tracking_name = track_range_list[i]

        # load mask zarr
        seg_path = os.path.join(root, "tracking", project_name, tracking_config, tracking_name, "segments.zarr")
        seg_zarr = zarr.open(seg_path, mode='a')

        # create second zarr file for stitched tracks
        # new_seg_path = os.path.join(root, "tracking", project_name, tracking_config, tracking_name, "segments_stitched.zarr")

        # synchronizer = ProcessSynchronizer('pipeline_org/my_lock')
        # seg_zarr_stitched = zarr.open(new_seg_path, mode='a', shape=seg_zarr.shape, chunks=seg_zarr.chunks,
        #                               dtype=seg_zarr.dtype, synchronizer=synchronizer)

        # check for previous seg zarr
        all_indices = set(np.arange(seg_zarr.shape[0]))

        # List files directly within zarr directory (recursive search):
        # existing_chunks = os.listdir(new_seg_path)

        # Extract time indices from chunk filenames:
        # written_indices = set(int(fname.split('.')[0])
        #                       for fname in existing_chunks if fname[0].isdigit())
        #
        # empty_indices = np.asarray(sorted(all_indices - written_indices))

        # if overwrite:
        write_indices = np.asarray(list(all_indices))
        # else:
        #     write_indices = empty_indices

        print("Copying indices.") #, write_indices)
        # run_copy = partial(copy_zarr, src=seg_zarr, dst=seg_zarr_stitched)
        # process_map(run_copy, write_indices, max_workers=n_workers, chunksize=1)

        # load tracks
        track_path = os.path.join(root, "tracking", project_name, tracking_config, tracking_name, "tracks.csv")
        tracks_df = pd.read_csv(track_path)

        # clean up tracks
        print("Preprocessing tracks...")
        tracks_df = preprocess_tracks(tracks_df, ROLL_W=5, MOVE_THRESH=1.0, OVERLAP_MIN=3, MERGE_DIST=7.5)

        # call track stitching
        print("Stitching tracks...")
        tracks_df_stitched = close_tracks_gaps(tracks_df, max_gap=max_gap, max_radius=max_radius, scale=scale_vec)

        # get map from old IDS to new IDS
        # map_df = tracks_df[["t", "track_id", "id"]].copy()
        # map_df = map_df.rename(columns={"track_id": "old_track_id"})
        # map_df = map_df.merge(tracks_df_stitched[["t", "track_id", "id"]].drop_duplicates(), on=["t", "id"], how="left")
        # check_df = map_df.loc[:, ["old_track_id"]].drop_duplicates()
        # map_df = map_df.loc[:, ["track_id", "old_track_id"]].drop_duplicates().reset_index(drop=True)
        # if map_df.shape[0] != check_df.shape[0]:
        #     raise ValueError("Degenerate old -> new mappings found in map_df")
        #
        # # relabel frames that come from tracks2
        # max_label = np.max(map_df["old_track_id"])
        # # Create an identity lookup table (i.e. each label maps to itself)
        # lookup = np.zeros((max_label+1,))  #np.arange(max_label + 1, dtype=seg_zarr.dtype)
        #
        # # Update the lookup table with your mapping.
        # for _, row in map_df.iterrows():
        #     # It's assumed that old_label is within [0, max_label]
        #     lookup[row["old_track_id"]] = row["track_id"]

        # reindex
        print("Reindexing segments...")
        # run_reindex = partial(reindex_mask, seg_zarr=seg_zarr_stitched, lookup=lookup, track_df=tracks_df_stitched)
        # process_map(run_reindex, write_indices, max_workers=n_workers, chunksize=1)

        # save stitched tracks
        print("Saving stitched tracks...")
        tracks_df_stitched.to_csv(
            os.path.join(root, "tracking", project_name, tracking_config, tracking_name, "tracks_stitched.csv"),
            index=False)
