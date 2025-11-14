import pandas as pd
import numpy as np
import os
from ultrack.tracks.gap_closing import close_tracks_gaps
from ultrack.tracks.graph import inv_tracks_df_forest

if __name__ == "__main__":
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # call tracking
    project_name = "20250311_LCP1-NLSMSC"
    tracking_config = "tracking_20250328_redux"
    tracking_name = "track_0000_2339_cb"

    # load tracks
    track_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000", tracking_name, "tracks_fluo.csv")
    tracks_df = pd.read_csv(track_path)

    # call track stitching
    print("Stitching tracks...")
    tracks_df_stitched = close_tracks_gaps(tracks_df, max_gap=3, max_radius=25, scale=np.asarray([3.0, 1.0, 1.0]))

    # save stitched tracks
    print("Saving stitched tracks...")
    tracks_df_stitched.to_csv(os.path.join(root, "tracking", project_name, tracking_config, "well0000", tracking_name, "tracks_fluo_stitched.csv"), index=False)

    print("Check")