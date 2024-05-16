import napari
import os
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import pandas as pd
import json


####################
# # set parameters
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "230425_EXP21_LCP1_D6_1pm_DextranStabWound"
image_zarr = os.path.join(root, "built_data", "zarr_image_files",  project_name + ".zarr")
# label_zarr = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
ds_factor = 1
config_name = "tracking_jordao_full.txt"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")

save_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)

metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)
scale_vec_im = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
scale_vec = scale_vec_im # np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

# specify time points to load
start_i = 0
stop_i = 119
###########################

# load track velocity data
vel_df = pd.read_csv(os.path.join(root, "metadata", project_name, "velocity_df.csv"))
vel_df["t"] = vel_df["frame"]
# load image dataset
data_tzyx = zarr.open(image_zarr, mode='r')

viewer = napari.view_image(data_tzyx[start_i:stop_i], scale=tuple(scale_vec_im))

# viewer.add_labels(label_tzyx[start_i:stop_i], scale=tuple(scale_vec), name="raw labels")

cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
_, graph = to_tracks_layer(cfg)
tracks_df = pd.read_csv(os.path.join(root, "metadata", project_name, "tracks_df_dist.csv"))
tracks_df["dist_binary"] = 1*(tracks_df["distance"] <= 250)

tracks_df_first = tracks_df.loc[:, ["track_id", "distance"]].drop_duplicates(subset="track_id",keep="first")
tracks_df_first = tracks_df_first.rename(columns={"distance": "start_dist"})
tracks_df = tracks_df.merge(tracks_df_first, how="left", on="track_id")


properties = {"distance": -tracks_df.loc[:, "distance"].to_numpy(), "dist_bin": tracks_df.loc[:, "dist_binary"].to_numpy(),
              "start_dist": tracks_df.loc[:, "start_dist"].to_numpy()}


viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]],
    name="tracks",
    scale=tuple(scale_vec),
    properties=properties,
    translate=(0, 0, 0, 0),
    visible=False,
)

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"


# segments = zarr.open(os.path.join(save_directory, "segments.zarr"), mode='r')
#
# viewer.add_labels(
#     segments[start_i:stop_i],
#     name="segments",
#     scale=tuple(scale_vec),
#     translate=(0, 0, 0, 0),
# ).contour = 2


if __name__ == '__main__':
    napari.run()