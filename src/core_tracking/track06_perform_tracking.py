# import napari
import os
import skimage.io as io
import numpy as np
from tqdm import tqdm
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

def perform_tracking(root, project_name, config_name, first_i=None, overwrite_flag=True, track_centroids=False, center_rad=2):

    mask_directory = os.path.join(root, "built_data", "cleaned_cell_labels", project_name, "")
    mask_list = sorted(glob.glob(mask_directory + "*.tif"))
    if first_i is None:
        first_i = len(mask_list)
    mask_list = mask_list[:first_i]

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
        [metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
    # mask_list = mask_list[:5]

    config_path = os.path.join(root, "metadata", project_name, config_name)
    cfg = load_config(config_path)

    # Load and resize
    print("Loading time points...")
    for m, mask_path in enumerate(tqdm(mask_list)):

        data_zyx = io.imread(mask_path)

        if m == 0:
            data_tzyx = np.empty((len(mask_list), data_zyx.shape[0], data_zyx.shape[1], data_zyx.shape[2]),
                                 dtype=data_zyx.dtype)


        if track_centroids:
            fp = ball(center_rad)
            regions = regionprops(data_zyx)
            new_mask_array = np.zeros(data_zyx.shape, dtype=np.uint16)
            for region in regions:
                centroid = np.asarray(region.centroid).astype(int)
                new_mask_array[centroid[0], centroid[1], centroid[2]] = region.label

            new_mask = dilation(new_mask_array, fp)
            data_tzyx[m, :, :, :] = new_mask
        else:
            data_tzyx[m, :, :, :] = data_zyx

    detection, boundaries = labels_to_edges(data_tzyx)
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
    project_name = "240219_LCP1_93hpf_to_127hpf" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"  #

    segments, tracks_df = perform_tracking(root, project_name, config_name="tracking_v1.txt", track_centroids=False)


