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
project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
image_dir = os.path.join(root, "built_data", project_name, "")
ds_factor = 2

save_directory = os.path.join(root, "built_data", "tracking", project_name)

metadata_file_path = os.path.join(save_directory, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)
scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
scale_vec_im = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
#
# specify time points to load
image_list = sorted(glob.glob(image_dir + "*.tiff"))
image_list = image_list[:10]
n_time_points = len(image_list)

time_points = np.arange(1, n_time_points + 1)
# time_points = time_points[:50]
# Load and resize
print("Loading time points...")
for t, time_point in enumerate(tqdm(time_points)):

    image_data = io.imread(image_list[t])
    dims_orig = image_data.shape
    dims_new = tuple(np.asarray(dims_orig) // 2)
    data_zyx = resize(image_data, dims_new, order=1, preserve_range=True)
    if t == 0:
        data_tzyx = np.empty((len(time_points), data_zyx.shape[0], data_zyx.shape[1], data_zyx.shape[2]), dtype=data_zyx.dtype)

    data_tzyx[t, :, :, :] = data_zyx

# data_tzyx = np.load(project_path + "image_data_ds.npy")

cfg = load_config(os.path.join(root, "metadata", project_name, "tracking_config.toml"))
tracks_df, graph = to_tracks_layer(cfg)


viewer = napari.view_image(data_tzyx, scale=tuple(scale_vec))

viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]],
    name="tracks",
    graph=graph,
    scale=scale_vec,
    translate=(0, 0, 0, 0),
    visible=False,
)

# segments = tracks_to_zarr(
#     cfg,
#     tracks_df,
#     store_or_path=project_path + "segments_v2.zarr",
#     overwrite=True,
# )
segments = zarr.open(os.path.join(save_directory, "segments.zarr"), mode='r')

viewer.add_labels(
    segments,
    name="segments",
    scale=scale_vec,
    translate=(0, 0, 0, 0),
).contour = 2
print("check")
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