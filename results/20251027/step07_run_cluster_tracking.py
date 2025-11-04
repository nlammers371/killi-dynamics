import numpy as np
import pandas as pd
from src.data_io.track_io import _load_track_data
from pathlib import Path
from src.symmetry_breaking.cell_cluster_tracking import (find_clusters_per_timepoint,
                                                        track_clusters_over_time,
                                                        stitch_tracklets)
from time import time

if __name__ == "__main__":
    # --- parameters ---
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    tracking_config = "tracking_20251102"

    tracks_df, sphere_df = _load_track_data(root=root,
                                         project_name=project_name,
                                         tracking_config=tracking_config)

    # filter for deep cells only and remove unnecessary columns
    tracks_df = tracks_df.loc[tracks_df["track_class"] == 0].copy()
    tracks_df = tracks_df[["t", "track_id", "x", "y", "z", "mean_fluo"]]

    start = time()
    # print("Finding clusters and tracking...")
    # 1) clusters per frame
    clusters_by_t = find_clusters_per_timepoint(
        tracks_df,
        sphere_df,
        d_thresh=30.0,
        min_size=5,  # tune
        fluo_col="mean_fluo",
        time_col="t",
        sphere_radius_col="radius_smooth",
        sphere_center_cols=("center_z_smooth", "center_y_smooth", "center_x_smooth")
    )
    print(f"Found clusters in {len(clusters_by_t)} timepoints, took {time()-start:.1f} s")

    start = time()
    # 2) link across time (motion/feature-aware, with merges)
    cluster_ts, merges_df = track_clusters_over_time(
        clusters_by_t,
        link_metric="overlap",         # or "jaccard"
        sim_min=0.3,
        max_centroid_angle=np.deg2rad(15),
        w_sim=1.0, w_feat=0.7, w_pred=0.7,  # tune
        pred_step=1.0
    )
    print(f"Tracked clusters over time, got {len(cluster_ts)} tracklets, took {time()-start:.1f} s")
    start = time()
    # 3) stitch fragmented tracklets (bridge small gaps, fix flips)
    stitched_ts, stitch_log = stitch_tracklets(
        cluster_ts,
        gap_max=2, window=1,
        link_metric="overlap", sim_min=0.3,
        max_centroid_angle=np.deg2rad(15),
        w_sim=1.0, w_feat=0.7, w_pred=0.7, w_size=3.0,
        max_iters=3
    )
    print(f"Stitched tracklets, got {len(stitched_ts)} final tracks, took {time()-start:.1f} s")
    print("check")