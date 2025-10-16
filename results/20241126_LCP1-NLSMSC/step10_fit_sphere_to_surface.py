import napari
import pandas as pd
import numpy as np
import os
from src.geometry import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh, fit_sphere
import zarr
from tqdm import tqdm
from skimage.measure import regionprops

if __name__ == "__main__":
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241126_LCP1-NLSMSC"
    tracking_config = "tracking_lcp_nuclei"
    use_tracking = False

    if use_tracking:
        start_i = 0
        stop_i = 719
        suffix = ""
        scale_vec = np.asarray([3.0, 0.8, 0.8])

        # load track df
        print("Loading track data...")
        try:
            tracks_df = pd.read_csv(
                os.path.join(
                    root, "tracking", project_name, tracking_config, "well0000",
                    f"track_{start_i:04}_{stop_i:04}{suffix}", "tracks_fluo.csv"))
        except:
            tracks_df = pd.read_csv(
                os.path.join(
                    root, "tracking", project_name, tracking_config, "well0000",
                    f"track_{start_i:04}_{stop_i:04}{suffix}", "tracks.csv"))
        track_index = tracks_df["t"].unique()

    else:
        mask_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_aff_nls.zarr")
        mask_zarr = zarr.open(mask_path, mode="r")
        scale_vec = np.asarray([
            mask_zarr.attrs['PhysicalSizeZ'],
            mask_zarr.attrs['PhysicalSizeY'],
            mask_zarr.attrs['PhysicalSizeX']
        ])

        print("Extracting centroids from masks...")
        centroid_list = []
        # assume mask_zarr shape = (T, Z, Y, X)
        for t in tqdm(range(mask_zarr.shape[0]), desc="Extracting centroids", unit="frame"):
            labels = np.asarray(mask_zarr[t])  # load single frame labels
            props = regionprops(labels)
            for p in props:
                z, y, x = p.centroid  # in pixel coordinates
                centroid_list.append([t, z, y, x])

        tracks_df = pd.DataFrame(centroid_list, columns=["t", "z", "y", "x"])
        track_index = tracks_df["t"].unique()
        start_i, stop_i, suffix = 0, track_index.max(), ""

    # ---- sphere fitting ----
    sphere_coord_list = []
    for i in tqdm(track_index, desc="Processing frames", unit="frame"):
        points = np.multiply(
            tracks_df[tracks_df["t"] == i][["z", "y", "x"]].to_numpy(),
            scale_vec
        )
        fitted_center, fitted_radius, _, _ = fit_sphere(points)
        sphere_coord_list.append([i, fitted_center[0], fitted_center[1], fitted_center[2], fitted_radius])

    sphere_df = pd.DataFrame(
        np.asarray(sphere_coord_list),
        columns=["t", "zs", "ys", "xs", "r"]
    )

    if use_tracking:
        print("Saving sphere data...")
        sphere_df.to_csv(
            os.path.join(
                root, "tracking", project_name, tracking_config, "well0000",
                f"track_{start_i:04}_{stop_i:04}{suffix}", "sphere_fit.csv"),
            index=False
        )
    else:
        out_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000")
        os.makedirs(out_path, exist_ok=True)
        sphere_df.to_csv(
            os.path.join( out_path, "sphere_fit.csv"),
            index=False
            )