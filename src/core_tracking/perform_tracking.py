# import napari
import os
import skimage.io as io
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import json
import glob2 as glob


# set path to mask files
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
mask_directory = os.path.join(root, "built_data", "cleaned_cell_labels", project_name, "")
mask_list = sorted(glob.glob(mask_directory + "*.tif"))

# make save directory
save_directory = os.path.join(root, "built_data", "tracking", project_name, "")
if not os.path.isdir(save_directory):
    os.makedirs(save_directory)

# load metadata
metadata_file_path = os.path.join(mask_directory, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)

mask_list = mask_list[:50]
# Load and resize
print("Loading time points...")
for m, mask_path in enumerate(tqdm(mask_list)):

    data_zyx = io.imread(mask_path)
    # dims_orig = image_data.shape
    # dims_new = np.round([dims_orig[0] / ds_factor * 2, dims_orig[1] / ds_factor, dims_orig[2] / ds_factor]).astype(int)
    # data_zyx = resize(image_data, dims_new, order=1)
    if m == 0:
        data_tzyx = np.empty((len(mask_list), data_zyx.shape[0], data_zyx.shape[1], data_zyx.shape[2]), dtype=data_zyx.dtype)

    data_tzyx[m, :, :, :] = data_zyx


# segment
# start = time.time()
# detection = np.empty(data_tzyx.shape, dtype=np.uint)
# array_apply(
#     data_tzyx,
#     out_array=detection,
#     func=detect_foreground,
#     sigma=15.0,
#     voxel_size=scale_vec,
# )
scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
# boundaries = np.empty(data_tzyx.shape, dtype=np.uint)
# array_apply(
#     data_tzyx,
#     out_array=boundaries,
#     func=robust_invert,
#     voxel_size=scale_vec,
# )


detection, boundaries = labels_to_edges(data_tzyx)

# print("Examine segmentation in napari")
# viewer = napari.view_image(data_tzyx, scale=tuple(scale_vec))
# label_layer = viewer.add_labels(detection, name='segmentation', scale=tuple(scale_vec))
# boundary_layer = viewer.add_image(boundaries, visible=False, scale=tuple(scale_vec))
# viewer.theme = "dark"

config_path = os.path.join(root, "metadata", project_name, "tracking_config.toml")
cfg = load_config(config_path)
# cfg =  MainConfig()  # or load default config
# cfg.segmentation_config.threshold = 0.5
# print(cfg)

# Perform tracking
track(
    cfg,
    detection=detection,
    edges=boundaries,
    scale=scale_vec,
    overwrite=True,
)

tracks_df, graph = to_tracks_layer(cfg)
tracks_df.to_csv(project_path + "tracks.csv", index=False)

segments = tracks_to_zarr(
    cfg,
    tracks_df,
    store_or_path=project_path + "segments.zarr",
    overwrite=True,
)

print("Saving downsampled image data...")
np.save(project_path + "image_data_ds.npy", data_tzyx)