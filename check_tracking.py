import napari
import os

import pandas as pd
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import skimage.io as io
from skimage.transform import resize
import json

# # set parameters
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "20240611_NLS-Kikume_24hpf_side2"
image_zarr = os.path.join(root, "built_data", "zarr_image_files",  project_name, project_name + ".zarr")
# label_zarr = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
ds_factor = 1
config_name = "tracking_jordao_20240918.txt"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")

save_directory = os.path.join(root, "tracking", project_name, tracking_folder)


# specify time points to load
start_i = 0
stop_i = 119


data_tzyx = zarr.open(image_zarr, mode='r')
# label_tzyx = zarr.open(label_zarr, mode='r')

metadata = data_tzyx.attrs
scale_vec_im = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
scale_vec = scale_vec_im

viewer = napari.view_image(data_tzyx[start_i:stop_i], scale=tuple(scale_vec_im))

# viewer.add_labels(label_tzyx[start_i:stop_i], scale=tuple(scale_vec), name="raw labels")

cfg = load_config(os.path.join(root, "metadata", "tracking", config_name))

tracks_df = pd.read_csv(os.path.join(save_directory, "well0000", "tracks.csv"))
# tracks_df, graph = to_tracks_layer(cfg)
# # tracks_df_ft = tracks_df.loc[(tracks_df["t"] >= start_i) & (tracks_df["t"] < stop_i), :]
# #
# #
# # track_index = np.unique(tracks_df_ft["track_id"])
# # keys = graph.keys()
# # graph_ft = {k:graph[k] for k in keys if k in track_index}
viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]],
    name="tracks",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
    visible=False,
)

segments = zarr.open(os.path.join(save_directory, "well0000", "segments.zarr"), mode='r')

viewer.add_labels(
    segments[start_i:stop_i],
    name="segments",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2


if __name__ == '__main__':
    napari.run()