import os
import zarr
import napari
from src.utilities.functions import path_leaf
import numpy as np
from omnipose.core import compute_masks
from cellpose.core import use_gpu
from skimage.segmentation import watershed
from scipy.ndimage import zoom
from tqdm import tqdm
from skimage import morphology
from skimage.morphology import label
import json
import pandas as pd
import glob2 as glob


def do_affinity_stitching(prob_array, grad_array, scale_vec, max_prob=12, min_prob=-8, seg_res=None, do_hierarchy=True,
                            prob_increment=4, niter=100, min_mask_size=25, max_mask_size=1e5, target_prob=None):

    # get resizing info
    shape_orig = np.asarray(prob_array.shape)
    shape_iso = shape_orig.copy()
    iso_factor = scale_vec[0] / scale_vec[1]
    shape_iso[0] = int(shape_iso[0] * iso_factor)

    if seg_res is not None:
        rs_factor = scale_vec[1] / seg_res
        shape_iso = (shape_iso * rs_factor).astype(int)
    else:
        rs_factor = 1.0

    use_GPU = use_gpu()

    # get device info
    device = (
        "cuda"
        if use_GPU
        else "cpu"
    )

    print("Resizing arrays...")
    zoom_factor = np.divide(shape_iso, shape_orig)
    # cp_array_rs = zoom(cp_mask_array, zoom_factor, order=0)
    grad_array_rs = zoom(grad_array, (1,) + tuple(zoom_factor), order=1) * rs_factor
    prob_array_rs = zoom(prob_array, zoom_factor,
                         order=1)  # resize(prob_array, shape_iso, preserve_range=True, order=1)

    # list of prob thresholds to use
    if do_hierarchy:
        mask_thresh_list = list(range(min_prob, max_prob + prob_increment, prob_increment))
    else:
        mask_thresh_list = [target_prob]
    seg_hypothesis_array = np.zeros((len(mask_thresh_list),) + prob_array_rs.shape, dtype=np.uint16)

    for m, mask_threshold in enumerate(tqdm(mask_thresh_list, "Extracting affinity masks...")):
        mask_aff, _, _, _, _ = compute_masks(grad_array_rs, prob_array_rs,
                                             do_3D=True,
                                             niter=niter,
                                             boundary_seg=False,
                                             affinity_seg=True,
                                             min_size=10,
                                             max_size=max_mask_size,
                                             mask_threshold=mask_threshold,
                                             verbose=False,
                                             interp=True,
                                             omni=True,
                                             cluster=False,
                                             use_gpu=use_GPU,
                                             device=device,
                                             nclasses=2,
                                             dim=3)

        seg_hypothesis_array[m] = mask_aff

    ######
    # performing hierarchical watershed
    # initialize
    masks_curr = seg_hypothesis_array[0]  # start with the most permissive mask
    if do_hierarchy:
        for m in tqdm(range(1, len(mask_thresh_list))):
            # get next layer of labels
            aff_labels = seg_hypothesis_array[m]

            # get union of two masks
            mask_u = (masks_curr + aff_labels) > 0

            # get label vectors
            curr_vec = masks_curr[mask_u]
            next_vec = aff_labels[mask_u]

            # get index vec
            u_indices = np.where(mask_u)

            # get lists of unique labels
            labels_u_curr = np.unique(curr_vec)

            # for each label in the new layer, find label in prev layer that it most overlaps
            lb_df = pd.DataFrame(next_vec, columns=["next"])
            lb_df["curr"] = curr_vec

            # get most frequent curr label for each new label
            m_df = lb_df.groupby(by=["next"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
            top_label_vec = m_df.loc[:, "curr"].to_numpy()

            # get most frequent new label for each curr label
            # m_df2 = lb_df.groupby(by=["curr"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()

            # merge info back on
            lb_df = lb_df.merge(m_df.rename(columns={"curr": "top_curr"}), how="left", on="next")
            # lb_df = lb_df.merge(m_df2.rename(columns={"next": "top_next"}), how="left", on="curr")

            # initialize mask and marker arrays for watershed
            mask_array = (masks_curr > 0) * 1  # this dictates the limits of what can be foreground

            # initialize marker array for watershed seeding
            marker_array = np.zeros(masks_curr.shape, dtype=np.uint16)

            # get indices to populate
            ft = (lb_df.loc[:, "curr"] == lb_df.loc[:, "top_curr"]) & (lb_df.loc[:, "next"] > 0)
            lb_indices = tuple(u[ft] for u in u_indices)

            # generate new label set
            _, new_labels = np.unique(next_vec[ft], return_inverse=True)
            marker_array[lb_indices] = new_labels + 1

            # add markers from base that do not appear in new layer
            included_base_labels = np.unique(top_label_vec)
            max_lb_curr = np.max(marker_array) + 1
            missing_labels = np.asarray(list(set(labels_u_curr) - set(included_base_labels)))

            ft2 = np.isin(curr_vec, missing_labels) & (~ft)
            lb_indices2 = tuple(u[ft2] for u in u_indices)

            _, missed_labels = np.unique(curr_vec[ft2], return_inverse=True)
            marker_array[lb_indices2] = missed_labels + 1 + max_lb_curr

            # finally, expand the mask array to accommodate markers from the new labels that are not in the reference
            mask_array = (mask_array + marker_array) > 0

            # calculate watershed
            wt_array = watershed(image=-prob_array_rs, markers=marker_array, mask=mask_array, watershed_line=True)

            masks_curr = wt_array
    # viewer.add_labels(wt_array)

    # resize
    masks_out_rs = zoom(masks_curr, zoom_factor**-1, order=0)
    masks_out_rs = morphology.remove_small_objects(masks_out_rs, min_mask_size)

    return masks_out_rs


def stitch_cellpose_labels(root, model_name, project_name, overwrite=False):
    # get path to cellpose output
    cellpose_directory = os.path.join(root, "built_data", "cellpose_output", model_name, project_name, '')

    # make directory to write stitched labels
    out_directory = os.path.join(root, "built_data", "stitched_labels", model_name, '')
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    # load metadata
    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)

    # returns JSON object as
    # a dictionary
    metadata = json.load(f)

    pixel_res_raw = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])

    print("Stitching data from " + project_name)
    prob_name = os.path.join(cellpose_directory, project_name + "_probs.zarr")
    grad_name = os.path.join(cellpose_directory, project_name + "_grads.zarr")
    # mask_name = os.path.join(cellpose_directory, file_prefix + "_labels.zarr")

    # mask_zarr = zarr.open(mask_name, mode="r")
    prob_zarr = zarr.open(prob_name, mode="r")
    grad_zarr = zarr.open(grad_name, mode="r")

    # get number of time points
    n_time_points = prob_zarr.shape[0]

    # generate zarr store for stitched masks
    s_mask_zarr_path = os.path.join(out_directory, project_name + "_labels_stitched.zarr")
    prev_flag = os.path.isdir(s_mask_zarr_path)
    s_mask_zarr = zarr.open(s_mask_zarr_path, mode='a', shape=prob_zarr.shape, dtype=np.uint16,
                          chunks=(1,) + prob_zarr.shape[1:])

    # determine which indices to stitch
    print("Determining which time points need stitching...")
    if overwrite | (not prev_flag):
        write_indices = np.arange(n_time_points)
    else:
        write_indices = []
        for t in tqdm(range(n_time_points), "Checking which frames to segment..."):
            nz_flag_to = np.any(s_mask_zarr[t, :, :, :] != 0)
            if not nz_flag_to:  # if the cellpose output is all zeros
                nz_flag_from = np.any(prob_zarr[t, :, :, :] != 0)
                if nz_flag_from:  # guard against edge case where cellpose output was initialized but not filled
                    write_indices.append(t)

    # iterate through time points
    for time_int in tqdm(write_indices, "Stitching labels..."):

        # use affinity graph method from omnipose core to stitch masks at different probability levels
        # do the stitching
        # cp_mask_array = mask_zarr[time_int, :, :, :]
        grad_array = grad_zarr[time_int, :, :, :, :]
        prob_array = prob_zarr[time_int, :, :, :]
        # cp_mask_array = cp_mask_array[:, 340:590, 90:270] #[:, 500:775, 90:435]
        # grad_array = grad_array[:, :, 340:590, 90:270]  #[:, :, 500:775, 90:435]
        # prob_array = prob_array[:, 340:590, 90:270]  #[:, 500:775, 90:435]

        # viewer = napari.view_image(prob_array, scale=tuple(scale_vec))
        # perform stitching
        stitched_labels = do_affinity_stitching(prob_array, grad_array, pixel_res_raw, seg_res=1.5,
                                                do_hierarchy=False, target_prob=-8)#max_prob=0, min_prob=-16)

        # save
        s_mask_zarr[time_int] = stitched_labels


if __name__ == "__main__":
    overwrite = False
    xy_ds_factor = 1
    cell_diameter = 20
    cellprob_threshold = 16.0
    project_name = "230425_EXP21_LCP1_D6_1pm_DextranStabWound"
    # set path to CellPose model to use
    model_name = "LCP-Multiset-v1"

    # set read/write paths
    root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    stitch_cellpose_labels(root, model_name, project_name, overwrite)