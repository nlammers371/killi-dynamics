import napari
import os
import numpy as np
import pandas as pd
import dask.array as da
import zarr


root = r"E:\Nick\Cole Trapnell's Lab Dropbox\Nick Lammers\Nick\killi_tracker"
project_name = "20241126_LCP1-NLSMSC"
image_path = os.path.join(root, "built_data", "zarr_image_files",  project_name + ".zarr")
track_name = "tracking_lcp_nuclei"
tracks_csv = os.path.join(root, "tracking", project_name, track_name, "well0000",
                          "track_0000_0719", "tracks_dist.csv")

# --- NEW: smoothing + color params ---
SMOOTH_WINDOW = 20          # in frames; odd numbers look nice with center=True
COLOR_MIN, COLOR_MAX = -1.5, 1.5   # set the color range you want for v_rad_smooth
N = 40

# Lazy array with time axis intact: shape like (T, Z, Y, X)
image_da = da.from_zarr(image_path)
image_z = zarr.open(image_path, mode="r")

scale_vec = (
    image_z.attrs["PhysicalSizeZ"],
    image_z.attrs["PhysicalSizeY"],
    image_z.attrs["PhysicalSizeX"],
)

tracks_df = pd.read_csv(tracks_csv)

# --- NEW: compute smoothed v_radial_s per track (rolling over time) ---
tracks_df = tracks_df.sort_values(["track_id", "t"])
tracks_df["v_rad_smooth"] = (
    tracks_df.groupby("track_id")["v_radial_s"]
             .transform(lambda s: s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean())
)
tracks_df = tracks_df.dropna(subset=["v_rad_smooth"])

viewer = napari.Viewer(ndisplay=3)
img_layer = viewer.add_image(
    image_da,
    name="image",
    channel_axis=1,
    colormap=["gray", "green"],
    scale=scale_vec,    # include time scale so coords match tracks
    contrast_limits=[(0, 2000), (0, 10000)],
    rgb=False,
)

# Optional: jump directly to the time you want to view (only that T will load)

viewer.dims.set_current_step(0, int(N))  # axis 0 is time

# Tracks expect columns [track_id, t, z, y, x]
props = {
    "distance": -tracks_df["d_geo"].to_numpy(), "time": -tracks_df["t"].to_numpy(),
    "start_dist": tracks_df.groupby("track_id")["d_geo"].transform("first").to_numpy(),
    "v_rad": tracks_df["v_radial_s"].to_numpy(),
    # --- NEW: expose smoothed values to layer properties ---
    "v_rad_smooth": tracks_df["v_rad_smooth"].to_numpy(),
}

tracks_layer = viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(),
    name="tracks",
    scale=(1,1,1),        # time+space scale to match image
    properties=props,
    tail_length=100,
    visible=True,
)

# --- NEW: make smoothed field the default color & set color range ---
tracks_layer.color_by = "v_rad_smooth"
tracks_layer.colormap = "turbo"           # pick any colormap name you like
tracks_layer.color_limits = (COLOR_MIN, COLOR_MAX)

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"

napari.run()
