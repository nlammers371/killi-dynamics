import napari
import pandas as pd
import numpy as np
import os
from src.geometry import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh, fit_sphere
import zarr
from tqdm import tqdm

if __name__ == "__main__":
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"
    tracking_config = "tracking_20250328_redux"

    start_i = 0
    stop_i = 2339
    suffix = "_cb"
    scale_vec = np.asarray([3.0, 1.0, 1.0])

    # load track df
    print("Loading track data...")
    tracks_df = pd.read_csv(os.path.join(root, "tracking", project_name, tracking_config, "well0000",
                                         f"track_{start_i:04}_{stop_i:04}{suffix}", "tracks_fluo.csv"))

    track_index = tracks_df["t"].unique()

    sphere_coord_list = []
    for i in tqdm(track_index[1500:], desc="Processing frames", unit="frame"):
        points = np.multiply(tracks_df[tracks_df["t"] == i][["z", "y", "x"]].to_numpy(), scale_vec)
        fitted_center, fitted_radius, _, _ = fit_sphere(points)
        sphere_coord_list.append([i, fitted_center[0], fitted_center[1], fitted_center[2], fitted_radius])

    sphere_df = pd.DataFrame(np.asarray(sphere_coord_list), columns=["t", "zs", "ys", "xs", "r"])

    # sphere_mesh = create_sphere_mesh(fitted_center, fitted_radius, 100)
    sphere_df.to_csv(os.path.join(root, "tracking", project_name, tracking_config, "well0000", f"track_{start_i:04}_{stop_i:04}{suffix}", "sphere_fit.csv"), index=False)
    # viewer = napari.Viewer()
    # viewer.add_points(points, size=15, face_color="Green", name="sphere centers")
    # viewer.add_surface(sphere_mesh)
    print("Saving sphere data...")

