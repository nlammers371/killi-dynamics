import napari
import numpy as np
import zarr
from ultrack.utils import labels_to_edges
from skimage.morphology import label
import skimage as ski
import SimpleITK as sitk
from skimage import filters
from skimage import morphology
from tqdm import tqdm
import pandas as pd
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def restitch_masks(im_log, thresh_range, min_mask_size=15):

    # open the zarr file
    # mask_stack_zarr = zarr.open(mask_stack_zarr_path, mode="a")
    # mask_aff_zarr_path = mask_stack_zarr_path.replace("stacks.zarr", "aff.zarr")
    # mask_aff_zarr = zarr.open(mask_aff_zarr_path, mode="r+")
    # prob_zarr = zarr.open(prob_zarr_path, mode="r")
    # prob_levels = mask_stack_zarr.attrs["prob_levels"]
    # if len(prob_levels) == 0:
    #     prob_levels = dict({})
    #     for t in range(0, mask_stack_zarr.shape[0]):
    #         prob_levels[str(int(t))] = [-4, -2, 0, 2, 4, 6, 8]
    #
    #     mask_stack_zarr.attrs["prob_levels"] = prob_levels
    #     mask_aff_zarr.attrs["prob_levels"] = prob_levels

    # iterate through time points
    mask_stack = np.zeros(((len(thresh_range),) + im_log.shape), dtype=np.uint16)
    # mask_out = np.zeros(im_log.shape, dtype=np.uint16)

    for t, thresh in enumerate(thresh_range):
        mask_stack[t] = label(im_log > thresh)

    # initialize
    masks_curr = mask_stack[0, :, :, :]  # start with the most permissive mask

    for m in tqdm(range(1, len(thresh_range)), "Performing hierarchical stitching..."):

        # get next layer of labels
        aff_labels = mask_stack[m, :, :, :]

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
        wt_array = watershed(image=-im_log, markers=marker_array, mask=mask_array, watershed_line=True)

        masks_curr = wt_array

    masks_out = morphology.remove_small_objects(masks_curr, min_mask_size)

    return masks_out, mask_stack

if __name__ == '__main__':
    import os

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241114_LCP1-NLSMSC_side1"
    out_directory = os.path.join(root, "built_data", "mask_stacks", '')
    aff_mask_zarr_path = os.path.join(out_directory, project_name + "_mask_aff.zarr")

    test = zarr.open(aff_mask_zarr_path, mode="r")
    # load zarr image file
    zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\zarr_image_files\\20241114_LCP1-NLSMSC_side1.zarr"
    image_data = zarr.open(zarr_path, mode="r")
    scale_vec = tuple([image_data.attrs['PhysicalSizeZ'],
                      image_data.attrs['PhysicalSizeY'],
                      image_data.attrs['PhysicalSizeX']])


    # pull frame
    frame_i = -250
    channel_i = 1
    data_zyx = np.squeeze(image_data[frame_i, channel_i])
    n_thresh = 7
    # thresh_manual = 100
    # calculate LoG

    # denoised = ndi.median_filter(data_zyx, size=3)
    image = data_zyx.copy()
    gaussian_background = ski.filters.gaussian(image, sigma=(1.33, 4, 4), preserve_range=True)  # NL this should be set dynamically based on image scale
    data_bkg = image - gaussian_background

    data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data_bkg), sigma=1))
    data_log_i = ski.util.invert(data_log)
    thresh_li = filters.threshold_li(data_log_i)

    thresh_range = np.linspace(thresh_li*0.7, 1.25*thresh_li, n_thresh)

    im_mask, mask_stack = restitch_masks(data_log_i, thresh_range, min_mask_size=20)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(data_bkg, scale=scale_vec)
    viewer.add_image(image, scale=scale_vec)
    viewer.add_image(data_log_i, scale=scale_vec)
    viewer.add_labels(test[0, 6], scale=scale_vec)
    viewer.add_labels(im_label, scale=scale_vec)

if __name__ == '__main__':
    napari.run()