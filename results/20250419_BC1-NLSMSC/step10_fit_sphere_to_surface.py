import napari
import pandas as pd
import numpy as np
import os
from src.build_killi.fit_embryo_surface import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh, fit_sphere
import zarr
from tqdm import tqdm

if __name__ == "__main__":
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20240611_NLS-Kikume_24hpf_side2" #"20250419_BC1-NLSMSC"
    tracking_config = "tracking_jordao_20240918" #"tracking_20250328_redux"

    start_i = 0
    stop_i = 1600 # 614
    suffix = ""
    scale_vec = np.asarray([3.0, 0.85, 0.85])

    # load track df
    print("Loading track data...")
    try:
        tracks_df = pd.read_csv(os.path.join(root, "tracking", project_name, tracking_config, "well0000", f"track_{start_i:04}_{stop_i:04}{suffix}", "tracks_fluo.csv"))
    except:
        tracks_df = pd.read_csv(os.path.join(root, "tracking", project_name, tracking_config, "well0000",
                                             f"track_{start_i:04}_{stop_i:04}{suffix}", "tracks.csv"))
    track_index = tracks_df["t"].unique()

    sphere_coord_list = []
    for i in tqdm(track_index, desc="Processing frames", unit="frame"):
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
