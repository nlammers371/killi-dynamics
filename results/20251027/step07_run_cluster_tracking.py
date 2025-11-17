from src.data_io.track_io import _load_track_data
from pathlib import Path
from time import time
import yaml,json
from dataclasses import asdict
from src.symmetry_breaking.cell_cluster_tracking import (find_clusters_per_timepoint,
                                                        track_clusters_over_time,
                                                        stitch_tracklets,
                                                        ClusterTrackingConfig)


if __name__ == "__main__":
    # --- parameters ---
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    tracking_config = "tracking_20251102"

    out_dir = root / "symmetry_breaking" / project_name / tracking_config
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_config = ClusterTrackingConfig(d_thresh=25.0)

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
        config=cluster_config,
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
        config=cluster_config,
    )
    print(f"Tracked clusters over time, got {len(cluster_ts)} tracklets, took {time()-start:.1f} s")
    cluster_ts.to_csv(out_dir / "cell_clusters_tracked.csv", index=False)

    start = time()
    # 3) stitch fragmented tracklets (bridge small gaps, fix flips)
    stitched_ts, stitch_log = stitch_tracklets(
        cluster_ts,
        config=cluster_config,
    )
    print(f"Stitched tracklets, got {len(stitched_ts)} final tracks, took {time()-start:.1f} s")


    # save final tracks
    stitched_ts.to_csv(out_dir / "cell_clusters_stitched.csv", index=False)
    # convert numpy scalars → native python scalars
    cfg_dict = json.loads(json.dumps(asdict(cluster_config)))
    with open("_Archive/cluster_tracking_config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)