import napari
import os
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import skimage.io as io
from skimage.transform import resize
import json

# # set parameters
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "240219_LCP1_93hpf_to_127hpf"  #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
image_zarr = os.path.join(root, "built_data", "exported_image_files",  project_name + ".zarr")
label_zarr = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
ds_factor = 2
config_name = "tracking_v1.txt"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")

save_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)

metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)
scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
scale_vec_im = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
#
# specify time points to load
start_i = 0

# image_list = sorted(glob.glob(image_dir + "*.tiff"))
# label_list = sorted(glob.glob(label_dir + "*_labels.tif"))
stop_i = 200  #len(image_list)# 25
# image_list = image_list[start_i:stop_i]
# n_time_points = len(image_list)

# time_points = np.arange(1, n_time_points + 1)
# time_points = time_points[:50]
# Load and resize
# print("Loading time points...")
# for t, time_point in enumerate(tqdm(time_points)):
#
#     data_zyx = io.imread(image_list[t])
#     lb_zyx = io.imread(label_list[t])
#     # image_data = io.imread(image_list[t])
#     # dims_orig = image_data.shape
#     # dims_new = tuple(np.asarray(dims_orig) // 2)
#     # data_zyx = resize(image_data, dims_new, order=1, preserve_range=True)
#     if t == 0:
#         data_tzyx = np.empty((len(time_points), data_zyx.shape[0], data_zyx.shape[1], data_zyx.shape[2]), dtype=data_zyx.dtype)
#         label_tzyx = np.empty((len(time_points), lb_zyx.shape[0], lb_zyx.shape[1], lb_zyx.shape[2]),
#                              dtype=data_zyx.dtype)
#
#     data_tzyx[t, :, :, :] = data_zyx
#     label_tzyx[t, :, :, :] = lb_zyx

# data_tzyx = np.load(project_path + "image_data_ds.npy")

data_tzyx = zarr.open(image_zarr, mode='r')
label_tzyx = zarr.open(label_zarr, mode='r')

viewer = napari.view_image(data_tzyx[start_i:stop_i], scale=tuple(scale_vec_im))

viewer.add_labels(label_tzyx[start_i:stop_i], scale=tuple(scale_vec), name="raw labels")

cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
tracks_df, graph = to_tracks_layer(cfg)
tracks_df_ft = tracks_df.loc[(tracks_df["t"] >= start_i) & (tracks_df["t"] < stop_i), :]


track_index = np.unique(tracks_df_ft["track_id"])
keys = graph.keys()
graph_ft = {k:graph[k] for k in keys if k in track_index}
viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]],
    name="tracks",
    graph=graph,
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
    visible=False,
)

segments = zarr.open(os.path.join(save_directory, "segments.zarr"), mode='r')

viewer.add_labels(
    segments[start_i:stop_i],
    name="segments",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2

# shape = cfg.data_config.metadata["shape"]
# dtype = np.int32
# chunks = large_chunk_size(shape, dtype=dtype)
#
# array = create_zarr(
#             shape,
#             dtype=dtype,
#             store_or_path=project_path + "image_data_ds.zarr",
#             chunks=chunks,
#             default_store_type=zarr.TempStore,
#             overwrite=True,
#         )
# nbscreenshot(viewer)

if __name__ == '__main__':
    napari.run()