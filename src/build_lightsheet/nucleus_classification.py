import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from skimage.measure import regionprops
import zarr
from functools import partial
from tqdm.contrib.concurrent import process_map
import multiprocessing
from glob2 import glob

def build_mask_feature_wrapper(root, project_name, tracking_config, well_num=0, start_i=0, stop_i=None, par_flag=False,
                               overwrite=False, suffix="", nls_channel=None, n_workers=None):

    """
    Build a feature table for the nuclei in the given project and tracking configuration.
    """

    # load image data
    image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    image_zarr = zarr.open(image_zarr_path, mode='r')

    if stop_i is None:
        stop_i = image_zarr.shape[0]

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    if nls_channel is None:
        channels = image_zarr.attrs["Channels"]
        nls_flags = np.asarray(["nls" in name for name in channels])
        if np.sum(nls_flags == 1) == 1:
            nls_channel = np.where(nls_flags == 1)[0][0]
        elif np.sum(nls_flags == 0) > 1:
            raise Exception("Multiple non-nuclear channels found")
        else:
            raise Exception("No non-nuclear channels found")

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    save_path = project_sub_path #os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")
    temp_dir = os.path.join(save_path, "temp_df_files", "")
    os.makedirs(temp_dir, exist_ok=True)

    # load input mask
    if "voxel_size_um" not in image_zarr.attrs.keys():
        ad = image_zarr.attrs
        scale_vec = [ad["PhysicalSizeZ"], ad["PhysicalSizeY"], ad["PhysicalSizeX"]]
    else:
        scale_vec = image_zarr.attrs["voxel_size_um"]

    # load tracking masks
    label_path = os.path.join(project_sub_path, "segments.zarr")
    seg_zarr = zarr.open(label_path, mode='r')

    # call build function
    run_build = partial(call_build_features, seg_zarr=seg_zarr, image_zarr=image_zarr,
                        scale_vec=scale_vec, nls_channel=nls_channel, temp_path=temp_dir)
    if par_flag:
        df_flags = process_map(run_build, range(start_i, stop_i), max_workers=n_workers, chunksize=1,
                                  desc="Building features", unit="frame")
    else:
        df_flags = []
        for t in tqdm(range(start_i, stop_i), desc="Building features", unit="frame"):
            df_flags.append(run_build(t))

    # load and combine frames
    df_list = sorted(glob(os.path.join(temp_dir, "features_*.csv")))
    feature_df_list = []
    for f in tqdm(df_list, "Loading feature files", unit="file"):
        df = pd.read_csv(f)
        feature_df_list.append(df)

    # concatenate
    feature_df = pd.concat(feature_df_list, ignore_index=True)
    feature_df.to_csv(save_path + "mask_features.csv", index=False)



def call_build_features(t, seg_zarr, image_zarr, scale_vec, nls_channel, temp_path):
    """
    Call the build_feature_table function for a specific time point.
    """

    # Extract the region labels from the segmentation mask
    seg_frame = np.squeeze(seg_zarr[t])
    im_frame = np.squeeze(image_zarr[t, nls_channel])
    regions = regionprops(seg_frame, intensity_image=im_frame, spacing=scale_vec)

    # Build the feature table
    df_features = build_feature_table(regions)

    df_features["frame"] = t

    # save intermediate results
    df_features.to_csv(os.path.join(temp_path, f"features_{t:04}.csv"), index=False)

    return True

def build_feature_table(regions):
    features_list = []  # to hold dictionaries of featurea
    # Only process regions for which we have ground-truth labels
    for region in regions: #, "Building feature dicts..."):

        # Basic shape features:
        volume = region.area                   # In 3D, area is volume (voxel count)
        convex_vol = region.convex_area        # Convex volume
        solidity = volume / convex_vol if convex_vol > 0 else np.nan
        extent = region.extent                 # Ratio of region volume to bounding-box volume

        # Intensity features:
        mean_intensity = region.mean_intensity
        min_intensity = region.min_intensity if hasattr(region, 'min_intensity') else np.nan
        max_intensity = region.max_intensity if hasattr(region, 'max_intensity') else np.nan

        # Inertia tensor eigenvalues: used to gauge elongation/sphericity.
        # For 3D objects, region.inertia_tensor_eigvals returns a tuple of three eigenvalues.
        eigvals = region.inertia_tensor_eigvals
        if eigvals is not None and len(eigvals) == 3 and eigvals[0] != 0:
            elongation_ratio = eigvals[-1] / eigvals[0]
        else:
            elongation_ratio = np.nan

        # Optionally, we can compute the bounding box dimensions:
        # region.bbox returns (min_z, min_y, min_x, max_z, max_y, max_x)
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        bbox_dims = (max_z - min_z, max_y - min_y, max_x - min_x)

        feature_dict = {
            'label': region.label,
            'volume': volume,
            'convex_volume': convex_vol,
            'solidity': solidity,
            'extent': extent,
            'mean_intensity': mean_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'elongation_ratio': elongation_ratio,
            # Optionally include bounding box dimensions:
            'bbox_z': bbox_dims[0],
            'bbox_y': bbox_dims[1],
            'bbox_x': bbox_dims[2]
        }
        features_list.append(feature_dict)

    # Create a DataFrame from the feature dictionaries
    df_features = pd.DataFrame(features_list)

    return df_features