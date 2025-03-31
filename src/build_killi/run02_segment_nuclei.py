import napari
import numpy as np
import zarr
import os
from skimage.morphology import label
import skimage as ski
import SimpleITK as sitk
from skimage import filters
from skimage import morphology
from tqdm import tqdm
import pandas as pd
from functools import partial
from skimage.segmentation import watershed
from scipy.interpolate import interp1d
from tqdm.contrib.concurrent import process_map
from func_timeout import func_timeout, FunctionTimedOut
import statsmodels.api as sm
import multiprocessing


def calculate_li_trend(root, project_prefix, first_i=0, last_i=None, multiside_experiment=True):

    if not multiside_experiment:
        manual_path = os.path.join(root, "built_data", "mask_stacks", project_prefix + "_li_df_manual.csv")
        if os.path.exists(manual_path):
            li_df = pd.read_csv(manual_path)
        else:
            alt_path = manual_path.replace("_manual", "")
            li_df = pd.read_csv(alt_path)
    else:
        manual_path1 = os.path.join(root, "built_data", "mask_stacks", project_prefix + "_side1_li_df_manual.csv")
        if os.path.exists(manual_path1):
            li_df_raw1 = pd.read_csv(manual_path1)
        else:
            alt_path1 = manual_path1.replace("_manual", "")
            li_df_raw1 = pd.read_csv(alt_path1)
        manual_path2 = os.path.join(root, "built_data", "mask_stacks", project_prefix + "_side2_li_df_manual.csv")
        if os.path.exists(manual_path2):
            li_df_raw2 = pd.read_csv(manual_path2)
        else:
            alt_path2 = manual_path2.replace("_manual", "")
            li_df_raw2 = pd.read_csv(alt_path2)
        li_df = pd.concat([li_df_raw1, li_df_raw2], axis=0, ignore_index=True)

    # get last index if not provided
    if last_i is None:
        if not multiside_experiment:
            zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_prefix + ".zarr")
        else:
            zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_prefix + "_side1.zarr")
        im_zarr = zarr.open(zarr_path, mode="r")
        last_i = im_zarr.shape[0]

    # get raw estimates
    x = li_df["frame"].to_numpy()
    y = li_df["li_thresh"].to_numpy()

    # set lower bound for 'reasonable' threshold. The estimator sometimes produces infeasibly low values
    y_thresh = np.percentile(y, 95) / 20

    # filter
    outlier_filter = y > y_thresh
    x = x[outlier_filter]
    y = y[outlier_filter]
    si = np.argsort(x)
    x = x[si]
    y = y[si]

    # perform smoothing
    lowess_result = sm.nonparametric.lowess(y, x, frac=0.3, it=3)
    x_lowess = lowess_result[:, 0]
    y_lowess = lowess_result[:, 1]

    # interpolate/extrapolate to get full trend
    frames_full = np.arange(first_i, last_i)
    thresh_interp = interp1d(x_lowess, y_lowess, kind="linear", fill_value="extrapolate")
    thresh_predictions = thresh_interp(frames_full)

    # generate li table
    li_df_full = pd.DataFrame(frames_full, columns=["frame"])
    li_df_full["li_thresh"] = thresh_predictions
    if not multiside_experiment:
        li_df_full.to_csv(os.path.join(root, "built_data", "mask_stacks", project_prefix + "_li_thresh_trend.csv"), index=False)
    else:
        li_df_full.to_csv(os.path.join(root, "built_data", "mask_stacks", project_prefix + "_side1_li_thresh_trend.csv"),
                          index=False)
        li_df_full.to_csv(os.path.join(root, "built_data", "mask_stacks", project_prefix + "_side2_li_thresh_trend.csv"),
                          index=False)

    return li_df_full

def perform_li_segmentation(time_int, li_df, image_zarr, nuclear_channel, multichannel_flag, stack_zarr, aff_zarr, n_thresh=5):

    li_thresh = li_df.loc[time_int, "li_thresh"]
    # do the stitching
    if multichannel_flag:
        image_array = np.squeeze(image_zarr[time_int, nuclear_channel, :, :, :])
    else:
        image_array = np.squeeze(image_zarr[time_int, :, :, :])

    if np.any(image_array != 0):

        data_log_i, thresh_li = calculate_li_thresh(image_array, thresh_li=li_thresh)

        thresh_range = np.linspace(thresh_li * 0.75, 1.25 * thresh_li, n_thresh)

        # perform stitching
        aff_mask, mask_stack = do_hierarchical_watershed(data_log_i, thresh_range=thresh_range)

        # save
        stack_zarr[time_int] = mask_stack
        aff_zarr[time_int] = aff_mask

        mms = stack_zarr.attrs["thresh_levels"]
        mms[int(time_int)] = list(thresh_range)
        mms_str_keys = {str(k): v for k, v in mms.items()}
        stack_zarr.attrs["thresh_levels"] = mms_str_keys

        ams = aff_zarr.attrs["thresh_levels"]
        ams[int(time_int)] = list(thresh_range)
        ams_str_keys = {str(k): v for k, v in ams.items()}
        aff_zarr.attrs["thresh_levels"] = ams_str_keys

        seg_flag = 1

    else:
        print(f"Skipping time point {time_int:04}: empty image found")
        seg_flag = 0

    return seg_flag


def calculate_li_thresh(image, LoG_sigma=1, gauss_sigma=None, thresh_li=None):
    
    if gauss_sigma is None:
        gauss_sigma = (1.33, 4, 4)
    
    # denoised = ndi.median_filter(data_zyx, size=3)
    gaussian_background = ski.filters.gaussian(image, sigma=gauss_sigma, preserve_range=True)
    data_bkg = image - gaussian_background

    data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data_bkg), sigma=LoG_sigma))
    data_log_i = ski.util.invert(data_log)
    if thresh_li is None:
        thresh_li = filters.threshold_li(data_log_i)
    
    return data_log_i, thresh_li


def do_hierarchical_watershed(im_log, thresh_range, min_mask_size=15):

    # iterate through time points
    mask_stack = np.zeros(((len(thresh_range),) + im_log.shape), dtype=np.uint16)

    for t, thresh in enumerate(thresh_range):
        mask_stack[t] = label(im_log > thresh)

    # initialize
    masks_curr = mask_stack[0, :, :, :]  # start with the most permissive mask

    for m in range(1, len(thresh_range)):  # tqdm(range(1, len(thresh_range)), "Performing hierarchical stitching..."):

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


def segment_nuclei(root, project_name, nuclear_channel=None, n_workers=None, par_flag=False, n_thresh=5, overwrite=False, last_i=None):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 2)

    # get raw data dir
    zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
    
    # make directory to write stitched labels
    out_directory = os.path.join(root, "built_data", "mask_stacks", '')
    os.makedirs(out_directory, exist_ok=True)

    # load LI thresh DF
    li_df = pd.read_csv(out_directory + project_name + "_li_thresh_trend.csv")

    #########
    print("Stitching data from " + project_name)
    image_zarr = zarr.open(zarr_path, mode="r")
    multichannel_flag = False
    if nuclear_channel is None:
        channel_list = image_zarr.attrs["Channels"]
        if len(channel_list) > 1:
            multichannel_flag = True
            nuclear_channel = [i for i in range(len(channel_list)) if ("H2B" in channel_list[i]) or ("nls" in channel_list[i])][0]
        else:
            nuclear_channel = 0
            
    if last_i is None:
        last_i = image_zarr.shape[0]

    # generate zarr store for stitched masks
    multi_mask_zarr_path = os.path.join(out_directory, project_name + "_mask_stacks.zarr")
    aff_mask_zarr_path = os.path.join(out_directory, project_name + "_mask_aff.zarr")

    if multichannel_flag:
        dim_shape = image_zarr.shape[2:]
    else:
        dim_shape = image_zarr.shape[1:]

    # initialize zarr file to save mask hierarchy
    stack_zarr = zarr.open(multi_mask_zarr_path, mode='a',
                                shape=(image_zarr.shape[0],) + (n_thresh,) + dim_shape,
                                dtype=np.uint16, chunks=(1, 1,) + dim_shape)

    # initialize zarr to save current best mask
    aff_zarr = zarr.open(aff_mask_zarr_path, mode='a', shape=(image_zarr.shape[0],) + dim_shape,
                              dtype=np.uint16, chunks=(1,) + dim_shape)

    # get all indices
    all_indices = set(range(last_i))

    # List files directly within zarr directory (recursive search):
    existing_chunks = os.listdir(aff_mask_zarr_path)

    # Extract time indices from chunk filenames:
    written_indices = set(int(fname.split('.')[0])
                          for fname in existing_chunks if fname[0].isdigit())

    empty_indices = np.asarray(sorted(all_indices - written_indices))

    if overwrite:
        write_indices = np.asarray(list(all_indices))
    else:
        write_indices = empty_indices

    # transfer metadata from raw data to cellpose products
    meta_keys = image_zarr.attrs.keys()

    for meta_key in meta_keys:
        stack_zarr.attrs[meta_key] = image_zarr.attrs[meta_key]
        aff_zarr.attrs[meta_key] = image_zarr.attrs[meta_key]

    stack_zarr.attrs["thresh_levels"] = dict({})
    aff_zarr.attrs["thresh_levels"] = dict({})

    # scale_vec = tuple([image_zarr.attrs['PhysicalSizeZ'],
    #                    image_zarr.attrs['PhysicalSizeY'],
    #                    image_zarr.attrs['PhysicalSizeX']])

    # iterate through time points
    li_thresh_call = partial(perform_li_segmentation, li_df=li_df, image_zarr=image_zarr, nuclear_channel=nuclear_channel,
                             multichannel_flag=multichannel_flag, stack_zarr=stack_zarr, aff_zarr=aff_zarr, n_thresh=n_thresh)

    if par_flag:
        print("Conducting segmentation in parallel...")
        seg_flags = process_map(li_thresh_call, write_indices, max_workers=n_workers, chunksize=1)
    else:
        seg_flags = []
        print("Conducting segmentation serially...")
        for time_int in tqdm(write_indices):
            seg_flag = li_thresh_call(time_int=time_int)
            seg_flags.append(seg_flag)

    return seg_flags



def estimate_li_thresh(root, project_name, interval=125, nuclear_channel=None, start_i=0, last_i=None, timeout=60*6):
    # get raw data dir
    zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")

    # make directory to write stitched labels
    out_directory = os.path.join(root, "built_data", "mask_stacks", '')
    os.makedirs(out_directory, exist_ok=True)

    #########
    print("Stitching data from " + project_name)
    image_zarr = zarr.open(zarr_path, mode="r")
    multichannel_flag = False
    if nuclear_channel is None:
        channel_list = image_zarr.attrs["Channels"]
        if len(channel_list) > 1:
            multichannel_flag = True
            nuclear_channel = \
            [i for i in range(len(channel_list)) if ("H2B" in channel_list[i]) or ("nls" in channel_list[i])][0]
        else:
            nuclear_channel = 0

    if last_i is None:
        last_i = image_zarr.shape[0]

    thresh_frames = np.arange(start_i, last_i, interval)
    thresh_frames[-1] = last_i - 1

    li_vec = []
    frame_vec = []
    for time_int in tqdm(thresh_frames, "Estimating Li thresholds..."):
        if multichannel_flag:
            image_array = np.squeeze(image_zarr[time_int, nuclear_channel, :, :, :]).copy()
        else:
            image_array = np.squeeze(image_zarr[time_int, :, :, :]).copy()
        try:
            # Runs calculate_li_thresh(image_array) with a timeout (in seconds)
            result = func_timeout(timeout, calculate_li_thresh, args=(image_array,))
            # Unpack the result as needed, for example:
            _, li_thresh = result
            li_vec.append(li_thresh)
            frame_vec.append(time_int)
            print(time_int)
            print(li_thresh)
        except FunctionTimedOut:
            print(f"Function timed out for time: {time_int}")
            pass


    frames_full = np.arange(0, last_i)
    # li_interp = np.interp(frames_full, thresh_frames[:7], li_vec)
    # li_interpolator = interp1d(thresh_frames[:7], li_vec, kind='linear', fill_value="extrapolate")

    # Fit a linear model (degree 1 polynomial)
    coefficients = np.polyfit(frame_vec, li_vec, deg=1)

    # Predict y values
    # li_interp = np.polyval(coefficients, frames_full)

    # interpolate
    li_df_raw = pd.DataFrame(frame_vec, columns=["frame"])
    li_df_raw["li_thresh"] = li_vec

    # li_interp = li_interpolator(frames_full)
    # li_df = pd.DataFrame(frames_full, columns=["frame"])
    # li_df["li_thresh"] = li_interp

    # Save
    out_directory = os.path.join(root, "built_data", "mask_stacks", '')
    os.makedirs(out_directory, exist_ok=True)
    # li_df.to_csv(out_directory + project_name + "_li_df.csv", index=False)
    li_df_raw.to_csv(out_directory + project_name + "_li_df.csv", index=False)

    return li_df

if __name__ == '__main__':
    
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_list = ["20241114_LCP1-NLSMSC_side1", "20241114_LCP1-NLSMSC_side2"]

    # segment_nuclei(root, project_name_list[0], last_i=805)
    project_name = project_name_list[1]
    last_i = 804
    li_df = estimate_li_thresh(root, project_name, last_i=805, timeout=3*60)

    # values for SIDE 1
    # li_vec = [61.835632, 72.91312, 82.08261, 93.956024, 112.982605, 161.982605]
    # frame_vec = [0, 125, 250, 375, 500, 804]
    # side 2
    li_vec = [70.920746, 79.20566, 86.81677, 101.95398, 123.53754, 156.8913, 193.03552]
    frame_vec = [0, 125, 250, 375, 500,625,  804]

    frames_full = np.arange(0, last_i)
    # li_interp = np.interp(frames_full, thresh_frames[:7], li_vec)
    li_interpolator = interp1d(frame_vec, li_vec, kind='linear', fill_value="extrapolate")

    # Fit a linear model (degree 1 polynomial)
    # coefficients = np.polyfit(frame_vec, li_vec, deg=1)

    # Predict y values
    # li_interp = np.polyval(coefficients, frames_full)

    # interpolate
    li_interp = li_interpolator(frames_full)
    li_df = pd.DataFrame(frames_full, columns=["frame"])
    li_df["li_thresh"] = li_interp

    # Save
    out_directory = os.path.join(root, "built_data", "mask_stacks", '')
    os.makedirs(out_directory, exist_ok=True)
    li_df.to_csv(out_directory + project_name + "_li_df.csv", index=False)
    # for project_name in project_name_list:
    #     segment_nuclei(root, project_name, last_i=805)