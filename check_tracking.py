import napari
import os
from aicsimageio import AICSImage
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import dask as da
import glob2 as glob
from ultrack.utils.array import create_zarr, large_chunk_size
from czitools import misc_tools
import zarr

# # set parameters
raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
file_prefix = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
ds_factor = 4
save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\kf_tracking\\built_data\\"
project_name = "20230108"
project_path = os.path.join(save_root, project_name, '')
#
# specify time points to load
image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))
# n_time_points = len(image_list)

# time_points = np.arange(1, n_time_points + 1)
# time_points = time_points[:50]
# Load and resize
print("Loading time points...")
# for t, time_point in enumerate(tqdm(time_points)):
readPath = os.path.join(raw_data_root, file_prefix + f"({1}).czi")
#
imObject = AICSImage(readPath)
#     image_data = np.squeeze(imObject.data)
#     dims_orig = image_data.shape
#     dims_new = np.round([dims_orig[0] / ds_factor * 2, dims_orig[1] / ds_factor, dims_orig[2] / ds_factor]).astype(int)
#     data_zyx = resize(image_data, dims_new, order=1)
#     if t == 0:
#         scale_vec = np.asarray(imObject.physical_pixel_sizes)
#         scale_vec = scale_vec*ds_factor
#         scale_vec[0] = scale_vec[0]/2
#
#         data_tzyx = np.empty((len(time_points), data_zyx.shape[0], data_zyx.shape[1], data_zyx.shape[2]), dtype=data_zyx.dtype)
#
#     data_tzyx[t, :, :, :] = data_zyx

data_tzyx = np.load(project_path + "image_data_ds.npy")

cfg = load_config(project_path + "config.txt")
tracks_df, graph = to_tracks_layer(cfg)
# tracks_df.to_csv(project_path + "tracks.csv", index=False)

scale_vec = np.asarray(imObject.physical_pixel_sizes)
scale_vec[0] = scale_vec[0]*2
scale_vec[1:] = scale_vec[1:]*4

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
segments = zarr.open(project_path + "segments_v2.zarr", mode='r')

viewer.add_labels(
    segments,
    name="segments",
    scale=scale_vec,
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