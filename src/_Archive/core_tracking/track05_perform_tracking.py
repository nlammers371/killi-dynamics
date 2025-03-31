# import napari
import os
import skimage.io as io
import numpy as np
from tqdm import tqdm
import zarr
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.imgproc.segmentation import detect_foreground
from skimage.measure import regionprops
from skimage.morphology import ball, dilation
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import json
import glob2 as glob

def perform_tracking(root, project_name, config_name, model_name, first_i=None, overwrite_flag=True,
                     stitched_lb_flag=True, last_i=None):

    # make save directory
    track_dir = config_name
    track_dir = track_dir.replace(".txt", "")
    track_dir = track_dir.replace(".toml", "")
    save_directory = os.path.join(root, "built_data", "tracking", project_name, track_dir, "")
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # load metadata
    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)
    metadata = json.load(f)
    scale_vec = np.asarray(
        [metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])

    config_path = os.path.join(root, "metadata", project_name, config_name)
    cfg = load_config(config_path)

    # Load and resize
    print("Loading time points...")
    if stitched_lb_flag:
        mask_path = os.path.join(root, "built_data", "stitched_labels", model_name, project_name + "_labels_stitched.zarr")
    elif track_centroids:
        print("Using cell centroids")
        mask_path = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + "_centroids.zarr")
    else:
        mask_path = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
    mask_data = zarr.open(mask_path, mode="r")

    if first_i is None:
        first_i = 0
    if last_i is None:
        last_i = mask_data.shape[0]
    mask_data = mask_data[first_i:last_i]


    detection, boundaries = labels_to_edges(mask_data)
    detection = detection.astype(np.uint16)
    boundaries = boundaries.astype(np.uint16)

    # Perform tracking
    track(
        cfg,
        detection=detection,
        edges=boundaries,
        scale=scale_vec,
        overwrite=overwrite_flag,
    )

    tracks_df, graph = to_tracks_layer(cfg)
    tracks_df.to_csv(os.path.join(save_directory + "tracks.csv"), index=False)

    segments = tracks_to_zarr(
        cfg,
        tracks_df,
        store_or_path=os.path.join(save_directory, "segments.zarr"),
        overwrite=True,
    )

    return tracks_df, segments


if __name__ == '__main__':

    # set path to mask files
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "230425_EXP21_LCP1_D6_1pm_DextranStabWound"
    model_name = "LCP-Multiset-v1"
    tracking_config = "tracking_jordao_full.txt"

    segments, tracks_df = perform_tracking(root, project_name, config_name=tracking_config, model_name=model_name,
                                           last_i=None)
                                           # first_i=1000, last_i=1050, track_centroids=False)
