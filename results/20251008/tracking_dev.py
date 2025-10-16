import pandas as pd
import numpy as np
from pathlib import Path
from src.symmetry_breaking.cluster_tracking import find_clusters_per_timepoint, track_clusters_over_time, stitch_tracklets
import zarr
import hdbscan

if __name__ == '__main__':
    root = "/media/nick/cluster/projects/data/killi_tracker/"
    project_name = "20250716"
    projection_path = Path(root) / "output_data" / "sphere_projections" / project_name
    well_ind = 2

    # load nuccleus df
    nucleus_df_path = projection_path / f"well{well_ind:04}_nucleus_df.csv"
    nucleus_df = pd.read_csv(nucleus_df_path)
    nucleus_df = nucleus_df.rename(columns={"t": "t_phys"})

    # load sphere df
    sphere_df_path = projection_path / f"well{well_ind:04}_sphere_fits.csv"
    sphere_df = pd.read_csv(sphere_df_path)

    # join
    nucleus_df = nucleus_df.merge(sphere_df.loc[:, ["t", "center_z_smooth", "center_y_smooth", "center_x_smooth"]],
                                  how="left", left_on="t_int", right_on="t")
    nucleus_df = nucleus_df.drop(columns=["t"]).rename(
        columns={"center_z_smooth": "zs", "center_y_smooth": "ys", "center_x_smooth": "xs", "t_int": "t"})
    #
    # 1) clusters per frame
    clusters_by_t = find_clusters_per_timepoint(
        nucleus_df,
        sphere_df,
        d_thresh=25.0,
        min_size=25,  # tune
        fluo_col="intensity",
        time_col="t",
        sphere_radius_col="radius",
        sphere_center_cols=( "center_z_smooth", "center_y_smooth", "center_x_smooth")
    )

    # 2) link across time (motion/feature-aware, with merges)
    cluster_ts, merges_df = track_clusters_over_time(
        clusters_by_t,
        link_metric="overlap",         # or "jaccard"
        sim_min=0.3,
        max_centroid_angle=np.deg2rad(15),
        w_sim=1.0, w_feat=0.7, w_pred=0.7,  # tune
        pred_step=1.0
    )

    # # 3) stitch fragmented tracklets (bridge small gaps, fix flips)
    # stitched_ts, stitch_log = stitch_tracklets(
    #     cluster_ts,
    #     gap_max=2, window=1,
    #     link_metric="overlap", sim_min=0.3,
    #     max_centroid_angle=np.deg2rad(15),
    #     w_sim=1.0, w_feat=0.7, w_pred=0.7, w_size=3.0,
    #     max_iters=3
    # )
