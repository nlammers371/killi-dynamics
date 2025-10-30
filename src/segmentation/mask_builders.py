"""Mask construction helpers for hierarchical watershed workflows."""
from __future__ import annotations

from typing import Sequence
from pathlib import Path
import zarr
import numpy as np
import pandas as pd
from skimage.morphology import label
from skimage import morphology
from skimage.segmentation import watershed

from src.segmentation.li_thresholding import compute_li_threshold_single_frame
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


def do_hierarchical_watershed(
    im_log: np.ndarray,
    thresh_range: Sequence[float],
    min_mask_size: int = 15,
):
    """Run hierarchical watershed stitching over a threshold sweep."""
    mask_stack = np.zeros(((len(thresh_range),) + im_log.shape), dtype=np.uint16)

    for t, thresh in enumerate(thresh_range):
        mask_stack[t] = label(im_log > thresh)

    masks_curr = mask_stack[0, :, :, :]

    for m in range(1, len(thresh_range)):
        aff_labels = mask_stack[m, :, :, :]
        mask_u = (masks_curr + aff_labels) > 0

        curr_vec = masks_curr[mask_u]
        next_vec = aff_labels[mask_u]
        u_indices = np.where(mask_u)

        labels_u_curr = np.unique(curr_vec)

        lb_df = pd.DataFrame(next_vec, columns=["next"])
        lb_df["curr"] = curr_vec
        m_df = lb_df.groupby(by=["next"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
        top_label_vec = m_df.loc[:, "curr"].to_numpy()
        lb_df = lb_df.merge(m_df.rename(columns={"curr": "top_curr"}), how="left", on="next")

        mask_array = (masks_curr > 0) * 1
        marker_array = np.zeros(masks_curr.shape, dtype=np.uint16)

        ft = (lb_df.loc[:, "curr"] == lb_df.loc[:, "top_curr"]) & (lb_df.loc[:, "next"] > 0)
        lb_indices = tuple(u[ft] for u in u_indices)
        _, new_labels = np.unique(next_vec[ft], return_inverse=True)
        marker_array[lb_indices] = new_labels + 1

        included_base_labels = np.unique(top_label_vec)
        max_lb_curr = np.max(marker_array) + 1
        missing_labels = np.asarray(list(set(labels_u_curr) - set(included_base_labels)))

        ft2 = np.isin(curr_vec, missing_labels) & (~ft)
        lb_indices2 = tuple(u[ft2] for u in u_indices)
        _, missed_labels = np.unique(curr_vec[ft2], return_inverse=True)
        marker_array[lb_indices2] = missed_labels + 1 + max_lb_curr

        mask_array = (mask_array + marker_array) > 0
        wt_array = watershed(image=-im_log, markers=marker_array, mask=mask_array, watershed_line=True)
        masks_curr = wt_array

    masks_out = morphology.remove_small_objects(masks_curr, min_mask_size)
    return masks_out, mask_stack


def perform_li_segmentation(
    time_int: int,
    thresh_range_table: pd.DataFrame,
    image_zarr_path: str | Path,
    mask_zarr_path: str | Path,
    group_key: str,
    nuclear_channel: int = None,
    preproc_flag: bool = True,
):
    """Segment a single timepoint using Li-threshold sweeping."""
    # --- open stores locally inside worker (avoid pickling open objects)
    image_store = zarr.open_group(image_zarr_path, mode="r")
    image_zarr = image_store[group_key]
    mask_store = zarr.open_group(Path(mask_zarr_path) / group_key, mode="a")
    # mask_zarr = mask_store[group_key]

    # get thresh range
    thresh_range = thresh_range_table.loc[thresh_range_table["frame"] == time_int, :].drop(
                    columns=["frame", "li_thresh"]).to_numpy()[0]

    if nuclear_channel is not None and image_zarr.ndim == 5:
        image_array = np.squeeze(image_zarr[time_int, nuclear_channel, :, :, :])
    else:
        image_array = np.squeeze(image_zarr[time_int, :, :, :])

    if np.any(image_array != 0):
        if preproc_flag:
            data_log_i, thresh_li = compute_li_threshold_single_frame(image_array, thresh_li=1) # use dummy thresh
        else:
            data_log_i = image_array
            # thresh_li = li_thresh

        # thresh_range = np.linspace(thresh_li * thresh_factors[0], thresh_factors[1] * thresh_li, n_thresh)
        aff_mask, mask_stack = do_hierarchical_watershed(data_log_i, thresh_range=thresh_range)

        mask_store["thresh_stack"][time_int] = mask_stack
        mask_store["stitched"][time_int] = aff_mask

        return 1

    print(f"Skipping time point {time_int:04}: empty image found")
    return 0


__all__ = ["do_hierarchical_watershed", "perform_li_segmentation"]

