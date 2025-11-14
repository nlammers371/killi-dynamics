import zarr
import os
import pandas as pd
import numpy as np
import zarr
from tqdm import tqdm
from skimage.measure import regionprops, label
from functools import partial
from tqdm.contrib.concurrent import process_map
import multiprocessing

from typing import Optional, Sequence, Tuple, Union
import scipy.ndimage as ndi
from glob2 import glob



def add_spherical_mask(mask, center, radius, scale_vec, value=1):
    """
    Efficiently add a spherical mask to a 3D numpy array by computing the region
    of interest (the bounding box around the sphere) and applying the sphere mask there.

    Parameters:
        mask (np.ndarray): 3D array (shape: (z, y, x)) representing your mask.
        center (tuple of floats): Coordinates (z, y, x) of the sphere's centroid.
        radius (float): Radius of the sphere.
        value (int, optional): The value to assign to voxels inside the sphere.

    Returns:
        np.ndarray: The modified mask with the spherical region set to the specified value.
    """
    z_dim, y_dim, x_dim = mask.shape
    z0, y0, x0 = center

    # Determine the bounding box indices, ensuring they stay within mask bounds.
    z_min = max(int(np.floor(z0 - radius)), 0)
    z_max = min(int(np.ceil(z0 + radius)) + 1, z_dim)
    y_min = max(int(np.floor(y0 - radius)), 0)
    y_max = min(int(np.ceil(y0 + radius)) + 1, y_dim)
    x_min = max(int(np.floor(x0 - radius)), 0)
    x_max = min(int(np.ceil(x0 + radius)) + 1, x_dim)

    # Generate coordinate arrays only for the bounding box.
    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

    # Compute the mask for the sphere in the subregion.
    sphere = (((z - z0) * scale_vec[0]) ** 2 + ((y - y0) * scale_vec[1]) ** 2 + (
                (x - x0) * scale_vec[2]) ** 2) <= radius ** 2

    # Update only the subregion of the main mask.
    mask[z_min:z_max, y_min:y_max, x_min:x_max][sphere] = value

    return mask


def generate_marker_mask(frame_i, mask_zarr, image_zarr, out_zarr, fluo_channel, fluo_df_dict,
                         n_masks_per_frame, fluo_thresh, min_size, overlap_thresh, ball_radius, scale_vec):
    # frame_i = 2200
    mask = np.squeeze(mask_zarr[frame_i])
    im = np.squeeze(image_zarr[frame_i, fluo_channel])
    fluo_df = fluo_df_dict[frame_i]

    # first get bright lcp regions (ig any)
    im_thresh = im > fluo_thresh
    # im_thresh_ft = morphology.remove_small_objects(im_thresh, min_size=min_size)
    im_thresh_lb_raw = label(im_thresh)
    props_raw = regionprops(im_thresh_lb_raw, spacing=scale_vec)
    areas = np.array([prop.area for prop in props_raw])
    label_vec = np.array([prop.label for prop in props_raw])
    im_thresh_lb = label(np.isin(im_thresh_lb_raw, label_vec[areas > min_size]))

    # first get brightest existing nuclear masks
    fluo_df = fluo_df.reset_index(drop=True)
    top_indices = np.argsort(fluo_df["mean_fluo"].to_numpy())[-n_masks_per_frame:]
    top_labels = fluo_df.loc[top_indices, "nucleus_id"].to_numpy()

    # get region coordinates
    mask_props = regionprops(mask, spacing=scale_vec)
    labels_to_skip = []
    for lb in top_labels:
        coords = mask_props[lb - 1].coords
        # check to see if region overlaps with
        fluo_pixels = im_thresh_lb[coords[:, 0], coords[:, 1], coords[:, 2]]
        fi, fc = np.unique(fluo_pixels, return_counts=True)
        if np.any(fi > 0):
            # get the largest region
            fc_max = np.max(fc[fi > 0]) / np.sum(fc)
            if fc_max > overlap_thresh:
                labels_to_skip.append(lb)

    # figure out what to add
    labels_to_skip = np.asarray(labels_to_skip)
    labels_to_add = top_labels[~np.isin(top_labels, labels_to_skip)]

    # add
    curr_label = np.max(im_thresh_lb) + 1
    for lb in labels_to_add:
        coords = mask_props[lb - 1].coords
        # check to see if region overlaps with
        fluo_pixels = im_thresh_lb[coords[:, 0], coords[:, 1], coords[:, 2]]
        z_filter = fluo_pixels == 0
        if np.sum(z_filter) > min_size:
            im_thresh_lb[coords[z_filter, 0], coords[z_filter, 1], coords[z_filter, 2]] = curr_label
            curr_label += 1

    # get centroids
    props = regionprops(im_thresh_lb)
    centroids = np.array([prop.centroid for prop in props]).astype(int)

    # make new mask array
    new_mask = np.zeros_like(mask)
    # Create a mask for the spheres
    for c in range(centroids.shape[0]):
        new_mask = add_spherical_mask(new_mask, centroids[c, :], ball_radius, value=c + 1, scale_vec=scale_vec)

    out_zarr[frame_i] = new_mask

    return True


def marker_mask_wrapper(root, project_name, fluo_channel,
                        overwrite_flag=False, par_flag=False, n_workers=None,
                        overlap_thresh=0.25, mask_range=None,
                        n_masks_per_frame=50, lcp_thresh=115, min_size=50, ball_radius=9, suffix=""):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    # load mask dataset
    mpath = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
    mask_full = zarr.open(mpath, mode="a")

    # load images
    zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    fused_image = zarr.open(zpath, mode="r")

    # initialize output path
    output_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_marker_masks" + suffix + ".zarr")
    marker_mask_zarr = zarr.open(output_path, mode="a", shape=mask_full.shape, chunks=mask_full.chunks, dtype=np.uint16)

    if mask_range is None:
        mask_range = [0, mask_full.shape[0]]

    # figure out which indices we need to process
    # determine which frames need to be analyzed
    all_indices = set(range(mask_range[0], mask_range[1]))
    # List files directly within output directory:
    existing_chunks = os.listdir(output_path)
    existing_indices = set(int(fname[-8:-4]) for fname in existing_chunks if fname[-8:-4].isdigit())
    # Extract time indices from chunk filenames:
    if overwrite_flag:
        write_indices = all_indices
    else:
        write_indices = all_indices - existing_indices

    # get scale info
    scale_vec = tuple(
        [mask_full.attrs['PhysicalSizeZ'], mask_full.attrs['PhysicalSizeY'], mask_full.attrs['PhysicalSizeX']])

    # load fluo datasts
    fluo_path = os.path.join(root, "built_data", "fluorescence_data", project_name, "")
    fluo_df_path_list = sorted(glob(fluo_path + "*.csv"))
    fluo_df_dict = dict({})
    for fluo_p in tqdm(fluo_df_path_list, "Loading fluorescence data files..."):
        df = pd.read_csv(fluo_p)
        frame = df.loc[0, "frame"]
        fluo_df_dict[frame] = df

    mask_fun_run = partial(generate_marker_mask, mask_zarr=mask_full, image_zarr=fused_image, out_zarr=marker_mask_zarr,
                           fluo_channel=fluo_channel, fluo_df_dict=fluo_df_dict, n_masks_per_frame=n_masks_per_frame,
                           fluo_thresh=lcp_thresh, min_size=min_size, overlap_thresh=overlap_thresh,
                           ball_radius=ball_radius, scale_vec=scale_vec)
    if par_flag:
        print("Generating masks using parallel processing...")
        process_map(mask_fun_run, write_indices, max_workers=n_workers, chunksize=1)

    else:
        for frame_i in tqdm(write_indices, "Adding masks..."):
            mask_fun_run(frame_i)


    return True
def fuse_images(frame, image1, image2, side1_shifts, side2_shifts, out_zarr=None, fuse_channel=None):
    """
    Fuses two masks for a given frame with shifts.

    Parameters:
    - frame: Time index.
    - image1: Zarr array of the first mask (4D: time, z, y, x).
    - image2: Zarr array of the second mask (4D: time, z, y, x).
    - side1_shifts: DataFrame with shifts for the first side.
    - side2_shifts: DataFrame with shifts for the second side.

    Returns:
    - mask_fused: Fused mask for the given frame.
    """
    imshape = image1.shape
    if (len(imshape) > 4) and (fuse_channel is not None):
        # if we have multiple channels, select the specified one
        image1 = np.squeeze(image1[frame, fuse_channel, :, :])
        image2 = np.squeeze(image2[frame, fuse_channel, :, :])

    elif (len(imshape) > 4) and (fuse_channel is None):
        raise Exception("fuse_channel must be specified if image has multiple channels.")

    # initialze 'full' array
    zdim1_orig = image1.shape[0]
    zdim2_orig = image2.shape[0]
    full_z = zdim1_orig + zdim2_orig  # int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
    full_shape = tuple([full_z]) + tuple(image1.shape[1:])

    # get shifts
    shift1 = side1_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()
    shift2 = side2_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()

    # assign to full array
    m1_full = np.zeros(full_shape, dtype=np.uint16)
    m1_full[zdim2_orig:, :, :] = image1[:, :, :]
    m2_full = np.zeros(full_shape, dtype=np.uint16)
    m2_full[:zdim2_orig, :, :] = image2[::-1, :, ::-1]

    # shift images
    image1_shifted = ndi.shift(m1_full, (shift1), order=1)
    image2_shifted = ndi.shift(m2_full, (shift2), order=1)

    # create blended overlap
    z_shift_size = int(np.ceil(shift2[0]))
    lin_weight_vec = np.linspace(0, 1, z_shift_size)
    side1_weights = np.zeros((full_shape[0]), dtype=np.float32)
    side2_weights = np.zeros((full_shape[0]), dtype=np.float32)

    # side 1 weights
    side1_weights[zdim2_orig:zdim2_orig + z_shift_size] = lin_weight_vec  # full weight for side2
    side1_weights[zdim2_orig + z_shift_size:] = 1

    # side 2 weights
    side2_weights[:zdim2_orig] = 1.0  # full weight for side2
    side2_weights[zdim2_orig:zdim2_orig + z_shift_size] = lin_weight_vec[::-1]

    # fuse maskes
    image_fused = (np.multiply(image1_shifted, side1_weights[:, np.newaxis, np.newaxis]) +
                  np.multiply(image2_shifted, side2_weights[:, np.newaxis, np.newaxis])).astype(np.uint16)

    # write to zarr if one is provided
    if out_zarr is not None:
        if (len(imshape) > 4) and (fuse_channel is not None):
            out_zarr[frame, fuse_channel] = image_fused
        elif imshape == 4:
            out_zarr[frame] = image_fused
        else:
            raise Exception("Unexpected image shape.")

        return True
    else:
        return image_fused

def image_fusion_wrapper(root, project_name, out_root=None, overwrite=False, par_flag=True, start_i=0, stop_i=None, n_workers=None):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    # load shift info
    metadata_path = os.path.join(root, "metadata", project_name + "_side1", "")
    half_shift_df = pd.read_csv(
        os.path.join(metadata_path, project_name + "_side2" + "_to_" + project_name + "_side1" + "_shift_df.csv"))
    # time_shift_df = pd.read_csv(os.path.join(metadata_path, "frame_shift_df.csv"))

    # generate shift arrays
    side2_shifts = half_shift_df.copy()  # + time_shift_df.copy()
    side1_shifts = half_shift_df.copy()
    side1_shifts[["zs", "ys", "xs"]] = 0  # no shift for side1

    # load zarr arrays for each side
    image_zarr_path1 = os.path.join(root, "built_data", "zarr_image_files", project_name + "_side1.zarr")
    image_zarr_path2 = os.path.join(root, "built_data", "zarr_image_files", project_name + "_side2.zarr")

    image_zarr1 = zarr.open(image_zarr_path1, mode='r')
    image_zarr2 = zarr.open(image_zarr_path2, mode='r')

    if stop_i is None:
        stop_i = image_zarr1.shape[0]

    # get output dims
    zdim1_orig = image_zarr1.shape[-3]
    zdim2_orig = image_zarr2.shape[-3]
    full_z = zdim1_orig + zdim2_orig
    multichannel_flag = False
    if len(image_zarr1.shape) == 4:  # tzyx
        full_shape = tuple([image_zarr1.shape[0], full_z]) + tuple(image_zarr1.shape[-2:])
        chunksize = (1, ) + (full_shape[1:])
    elif len(image_zarr1.shape) == 5:  # tczyx
        multichannel_flag = True
        full_shape = tuple([image_zarr1.shape[0], image_zarr1.shape[1], full_z]) + tuple(image_zarr1.shape[-2:])
        chunksize = (1, 1) + (full_shape[2:])
    else:
        raise Exception("Image zarr has unexpected shape.")

    # generate output zarr path
    if out_root is None:
        fused_image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    else:
        fused_image_zarr_path = os.path.join(out_root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    fused_image_zarr = zarr.open(fused_image_zarr_path, mode='a', shape=full_shape, dtype=np.uint16, chunks=chunksize)

    for key, val in image_zarr1.attrs.items():
        fused_image_zarr.attrs[key] = val
    # check which indices to write
    # get all indices
    all_indices = set(range(start_i, stop_i))

    # List files directly within zarr directory (recursive search):
    existing_chunks = os.listdir(fused_image_zarr_path)

    # Extract time indices from chunk filenames:
    written_frames = np.asarray([int(fname.split(".")[0]) for fname in existing_chunks if fname[0].isdigit()])
    if multichannel_flag:
        written_channels = np.asarray([int(fname.split(".")[1]) for fname in existing_chunks if fname[0].isdigit()])
        written_indices0 = set(written_frames[written_channels == 0])
        written_indices1 = set(written_frames[written_channels == 1])
        empty_indices0 = np.asarray(sorted(all_indices - written_indices0))
        empty_indices1 = np.asarray(sorted(all_indices - written_indices1))
        if overwrite:
            write_indices = np.asarray(list(all_indices))
            write_indices1 = np.asarray(list(all_indices))
        else:
            write_indices = empty_indices0
            write_indices1 = empty_indices1
    else:
        written_indices = set(written_frames)
        empty_indices = np.asarray(sorted(all_indices - written_indices))

        if overwrite:
            write_indices = np.asarray(list(all_indices))
        else:
            write_indices = empty_indices

    if multichannel_flag:
        fuse_run0 = partial(fuse_images, image1=image_zarr1, image2=image_zarr2, side1_shifts=side1_shifts,
                           side2_shifts=side2_shifts, out_zarr=fused_image_zarr, fuse_channel=0)
        fuse_run1 = partial(fuse_images, image1=image_zarr1, image2=image_zarr2, side1_shifts=side1_shifts,
                            side2_shifts=side2_shifts, out_zarr=fused_image_zarr, fuse_channel=1)
    else:
        fuse_run0 = partial(fuse_images, image1=image_zarr1, image2=image_zarr2, side1_shifts=side1_shifts,
                            side2_shifts=side2_shifts, out_zarr=fused_image_zarr)

    if par_flag:
        print("Using parallel processing")
        print("Fusing channel 0")
        results0 = process_map(fuse_run0, write_indices, max_workers=n_workers, chunksize=1)
        if multichannel_flag:
            print("Fusing channel 1")
            results1 = process_map(fuse_run1, write_indices1, max_workers=n_workers, chunksize=1)

    else:
        results0 = []
        for f in tqdm(write_indices, desc="Processing channel 0 for image fusion..."):
            result = fuse_run0(f)
            results0.append(result)
        if multichannel_flag:
            results1 = []
            for f in tqdm(write_indices1, desc="Processing channel 1 for image fusion..."):
                result = fuse_run1(f)
                results1.append(result)

    if multichannel_flag:
        return results0, results1
    else:
        return results0


def integrate_fluorescence(t, image_zarr,  mask_zarr, fluo_channel, out_folder):
    """
    Integrates fluorescence values for a given time point and mask.

    Parameters:
    - t: Time index.
    - image_zarr: Zarr array of images (4D: time, z, y, x).
    - mask_zarr: Zarr array of masks (4D: time, z, y, x).
    - fluo_channel: Index of the fluorescence channel to use.

    Returns:
    - Integrated fluorescence value for the given time point.
    """

    mask_frame = np.squeeze(mask_zarr[t])
    # if image_zarr is not None:
    im_frame = np.squeeze(image_zarr[t, fluo_channel])
    meta = image_zarr.attrs

    if "voxel_size_um" not in meta.keys():
        # get scale info
        scale_vec = tuple([meta['PhysicalSizeZ'],
                           meta['PhysicalSizeY'],
                           meta['PhysicalSizeX']])
    else:
        scale_vec = meta["voxel_size_um"]

    # initialize array to store fluorescence values
    props = regionprops(mask_frame, spacing=scale_vec)
    label_vec = np.array([pr.label for pr in props])
    fluo_df = pd.DataFrame(label_vec, columns=["nucleus_id"])
    fluo_df["frame"] = t
    centroids = np.array([pr.centroid for pr in props])
    if centroids.size > 0:
        fluo_df[["z", "y", "x"]] = centroids
        # fluo_mean = ndi_mean(im_frame, labels=mask_frame, index=label_vec)
        # fluo_df["mean_fluo"] = fluo_mean
        for p, pr in enumerate(props):
            # get array indices corresponding to nucleus mask from props
            coords = pr.coords
            # get the mean fluorescence value for each nucleus using coordinates
            fluo_df.loc[fluo_df["nucleus_id"] == pr.label, "mean_fluo"] = np.mean(im_frame[coords[:, 0], coords[:, 1], coords[:, 2]])
    else:
        fluo_df = pd.DataFrame(columns=["nucleus_id", "frame", "z", "y", "x", "mean_fluo"])
    # save
    output_file = os.path.join(out_folder, f"fluorescence_data_frame_{t:04}.csv")
    fluo_df.to_csv(output_file, index=False)

    return fluo_df

def transfer_fluorescence(frame_i, fluo_df, tracked_mask_zarr, mask_zarr, start_i):

    # do the label transfer
    fluo_df_frame = fluo_df.loc[fluo_df["frame"] == frame_i, :]
    props = regionprops(tracked_mask_zarr[frame_i - start_i])
    label_vec = np.array([pr.label for pr in props])
    fluo_vec = np.zeros(len(label_vec), dtype=np.float32)  # initialize fluorescence vector
    mask_frame = mask_zarr[frame_i]

    fluo_lookup = fluo_df_frame.set_index("nucleus_id")["mean_fluo"]

    # for pr in props:
    for p, pr in enumerate(props):
        coords = pr.coords  # get array indices corresponding to nucleus mask from props
        lb, counts = np.unique(mask_frame[coords[:, 0], coords[:, 1], coords[:, 2]], return_counts=True)  # get labels in the mask
        counts = counts[lb > 0]
        weights = counts / np.sum(counts)
        lb = lb[lb > 0]  # remove background label
        # fluo_vals = fluo_df_frame.loc[fluo_df_frame["nucleus_id"].isin(lb), "mean_fluo"].values
        # lb_vec = fluo_df_frame.loc[fluo_df_frame["nucleus_id"].isin(lb), "nucleus_id"].values
        # fluo_vals = fluo_vals[np.argsort(lb_vec)]

        fluo_vals = fluo_lookup.loc[lb].values

        fluo_vec[p] = np.dot(weights, fluo_vals)

    temp_df = pd.DataFrame(label_vec, columns=["track_id"])
    temp_df["t"] = frame_i - start_i
    temp_df["mean_fluo"] = fluo_vec

    return temp_df

def transfer_fluorescence_wrapper(root, project_name, fused_flag=True, tracking_config=None, tracking_range=None,
                                  suffix="", well_num=0, overwrite=False, use_markers_flag=False, n_workers=None, par_flag=False):

    """
    :param root:
    :param project_name:
    :param fused_flag:
    :param tracking_config:
    :param tracking_range:
    :param suffix:
    :param well_num:
    :param overwrite:
    :return:
    """

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    # get path to fluo files
    if use_markers_flag:
        fluo_path = os.path.join(root, "built_data", "fluorescence_data", project_name + "_markers", "")
    else:
        fluo_path = os.path.join(root, "built_data", "fluorescence_data", project_name, "")
    fluo_df_path_list = sorted(glob(fluo_path + "*.csv"))
    fluo_df_list = []
    for fluo_file in tqdm(fluo_df_path_list, "Loading fluorescence data files..."):
        fluo_df = pd.read_csv(fluo_file)
        fluo_df_list.append(fluo_df)

    # combine all fluorescence dataframes
    # fluo_df = pd.concat(fluo_df_list, ignore_index=True)

    # path to raw masks
    if use_markers_flag:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_marker_masks.zarr")
    elif fused_flag:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
    else:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_aff.zarr")

    # load raw mask zarr
    mask_zarr = zarr.open(mask_zarr_path, mode='r')

    #############################
    # load tracked zarr mask
    # get name
    tracking_name = tracking_config.replace(".txt", "")
    start_i, stop_i = tracking_range[0], tracking_range[1]

    fluo_df_full = pd.concat(fluo_df_list, ignore_index=True)

    if use_markers_flag:
        project_name += "_marker"

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    # load tracking masks
    label_path = os.path.join(project_sub_path, "segments.zarr")
    tracked_mask_zarr = zarr.open(label_path, mode='r')

    # load dataframe of track info
    tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))
    # tracks_df_fluo = tracks_df.copy()

    transfer_run = partial(transfer_fluorescence, start_i=start_i, fluo_df=fluo_df_full,
                           tracked_mask_zarr=tracked_mask_zarr, mask_zarr=mask_zarr)

    if par_flag:
        tr_df_list = process_map(transfer_run, range(start_i, stop_i), max_workers=n_workers, chunksize=1)
    else:
        tr_df_list = []
        for f in tqdm(range(start_i, stop_i), desc="Processing frames for fluorescence transfer..."):
            tr_df = transfer_run(f)
            tr_df_list.append(tr_df)

    transfer_df = pd.concat(tr_df_list, ignore_index=0)
    tracks_df = tracks_df.merge(transfer_df, on=["track_id", "t"], how="left")  # merge fluorescence values into the tracks dataframe

    tracks_df.to_csv(os.path.join(project_sub_path, "tracks_fluo.csv"), index=False)  # save the updated tracks dataframe


# define wrapper function for parallel processing
def integrate_fluorescence_wrapper(root, project_name, fluo_channel, fused_flag=True, overwrite=False,
                                   par_flag=True, start_i=0, stop_i=None, n_workers=None, use_markers_flag=False):
    """

    :param root:
    :param project_name:
    :param fluo_channel:
    :param fused_flag:
    :param par_flag:
    :param start_i:
    :param stop_i:
    :param n_workers:
    :param tracking_config:
    :param seg_model:
    :param suffix:
    :param well_num:
    :return:
    """

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    # Save the integrated fluorescence values to a CSV file
    if use_markers_flag:
        output_path = os.path.join(root, "built_data", "fluorescence_data", project_name + "_markers", "")
    else:
        output_path = os.path.join(root, "built_data", "fluorescence_data", project_name, "")
    os.makedirs(output_path, exist_ok=True)

    # define paths
    if fused_flag:
        image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
        if use_markers_flag:
            mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_marker_masks.zarr")
        else:
            mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
    else:
        image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
        if use_markers_flag:
            mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_marker_masks.zarr")
        else:
            mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_aff.zarr")

    # load zarr files
    mask_zarr = zarr.open(mask_zarr_path, mode='r')
    image_zarr = zarr.open(image_zarr_path, mode='r')

    if stop_i is None:
        stop_i = mask_zarr.shape[0]

    # determine which frames need to be analyzed
    all_indices = set(range(start_i, stop_i))
    # List files directly within output directory:
    existing_chunks = os.listdir(output_path)
    existing_indices = set(int(fname[-8:-4]) for fname in existing_chunks if fname[-8:-4].isdigit())
    # Extract time indices from chunk filenames:
    if overwrite:
        write_indices = all_indices
    else:
        write_indices = all_indices - existing_indices

    if par_flag:
        print("Using parallel processing")
        # Use process_map for parallel processing
        results = process_map(
            partial(integrate_fluorescence, image_zarr=image_zarr, mask_zarr=mask_zarr, fluo_channel=fluo_channel, out_folder=output_path),
            write_indices,
            max_workers=n_workers,
            chunksize=1
        )
    else:
        print("Using sequential processing")
        # Sequential processing
        results = []
        for t in tqdm(write_indices):
            result = integrate_fluorescence(t, image_zarr=image_zarr, mask_zarr=mask_zarr, fluo_channel=fluo_channel, out_folder=output_path)
            results.append(result)

    return True





def fuse_and_filter(frame, mask1, mask2, mask_out, side1_shifts, side2_shifts, dist_thresh):

    # extract ID dictionary
    keep_dict1 = mask1.attrs["mask_keep_ids"]
    keep_dict2 = mask2.attrs["mask_keep_ids"]

    # get scale info
    scale_vec = tuple([mask1.attrs['PhysicalSizeZ'],
                       mask1.attrs['PhysicalSizeY'],
                       mask1.attrs['PhysicalSizeX']])

    # initialze 'full' array
    zdim1_orig = mask1.shape[1]
    zdim2_orig = mask2.shape[1]
    full_z = zdim1_orig + zdim2_orig  # int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
    full_shape = tuple([full_z]) + tuple(mask1.shape[2:])

    # get shifts
    shift1 = side1_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()
    shift2 = side2_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()

    # extract masks
    m1 = np.squeeze(mask1[frame])
    m2 = np.squeeze(mask2[frame])

    # filter
    keep_ids1 = keep_dict1[str(frame)]
    m1_binary = np.isin(m1, keep_ids1)
    keep_ids2 = keep_dict2[str(frame)]
    m2_binary = np.isin(m2, keep_ids2)

    # assign to full array
    m1_full = np.zeros(full_shape, dtype=np.uint16)
    m1_full[zdim2_orig:, :, :] = m1_binary[:, :, :]
    m2_full = np.zeros(full_shape, dtype=np.uint16)
    m2_full[:zdim2_orig, :, :] = m2_binary[::-1, :, ::-1]

    mask1_shifted = ndi.shift(m1_full, (shift1), order=0)
    mask2_shifted = ndi.shift(m2_full, (shift2), order=0)

    # fuse maskes
    mask_fused = label((mask1_shifted + mask2_shifted) > 0)

    # fit sphere
    props01 = regionprops(mask_fused, spacing=scale_vec)
    centroid_array = np.asarray([pr["Centroid"] for pr in props01])

    fitted_center, fitted_radius, inner_radius, outer_radius = fit_sphere(centroid_array)

    # Generate the mesh for the sphere:
    vertices, faces = create_sphere_mesh(fitted_center, fitted_radius, resolution=50)

    # get distances
    radial_dist_vec = np.sqrt(np.sum((centroid_array - fitted_center) ** 2, axis=1))
    # map centroids to sphere vertices
    surf_dist_mat = distance_matrix(vertices, centroid_array)
    closest_indices = np.argsort(surf_dist_mat, axis=1)[:, :5]
    dist_vec = np.mean(radial_dist_vec[closest_indices], axis=1) - fitted_radius
    dist_vec[dist_vec > dist_thresh] = 0

    # use mean closest points to screen for outliers
    point_to_surf_mapping = np.argmin(surf_dist_mat.T, axis=1)
    ref_dist_vec = dist_vec[point_to_surf_mapping]
    surf_delta_vec = radial_dist_vec - (ref_dist_vec + fitted_radius)

    radius_mask = np.zeros(mask_fused.shape, dtype=np.float32)
    for i in range(len(radial_dist_vec)):
        coords = props01[i].coords
        radius_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = surf_delta_vec[i]

    # outlier_mask = radius_mask > dist_thresh
    mask_filtered = label((radius_mask < dist_thresh) & (mask_fused > 0))

    # write
    mask_out[frame] = mask_filtered


def fuse_wrapper(root, project_prefix, overwrite=False, par_flag=False, dist_thresh=30, n_workers=None, start_i=0, last_i=None):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)

    # get project names 
    project1 = project_prefix + "_side1"
    project2 = project_prefix + "_side2"

    # load mask zarr files
    mpath1 = os.path.join(root, "built_data", "mask_stacks", project1 + "_mask_aff.zarr")
    mask1 = zarr.open(mpath1, mode="r")

    mpath2 = os.path.join(root, "built_data", "mask_stacks", project2 + "_mask_aff.zarr")
    mask2 = zarr.open(mpath2, mode="r")
    
    if last_i is None:
        last_i = mask1.shape[0]

    # initialze 'full' array
    zdim1_orig = mask1.shape[1]
    zdim2_orig = mask2.shape[1]
    full_z = zdim1_orig + zdim2_orig  # int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
    full_shape = tuple([full_z]) + tuple(mask1.shape[2:])
        
    # initialize new zarr file
    fused_mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_prefix + "_mask_fused.zarr")
    fused_mask_zarr = zarr.open(fused_mask_zarr_path, mode='a', shape=(mask2.shape[0],) + full_shape,
                         dtype=np.uint16, chunks=(1,) + full_shape)

    #  figur e out which indices to write
    # get all indices
    all_indices = set(range(start_i, last_i))

    # List files directly within zarr directory (recursive search):
    existing_chunks = os.listdir(fused_mask_zarr_path)

    # Extract time indices from chunk filenames:
    written_indices = set(int(fname.split('.')[0])
                          for fname in existing_chunks if fname[0].isdigit())

    empty_indices = np.asarray(sorted(all_indices - written_indices))

    if overwrite:
        write_indices = np.asarray(list(all_indices))
    else:
        write_indices = empty_indices

    # load shift info
    metadata_path = os.path.join(root, "metadata", project1, "")
    half_shift_df = pd.read_csv(os.path.join(metadata_path, project2 + "_to_" + project1 + "_shift_df.csv"))
    time_shift_df = pd.read_csv(os.path.join(metadata_path, "frame_shift_df.csv"))

    # zeroing out time registration for now--I don't trust it
    time_shift_df[["xs", "ys", "zs"]] = 0

    # extend time shifts
    # frames_full = half_shift_df["frame"].to_numpy()
    # # time_frames = time_shift_df["frame"].to_numpy()
    # # new_frames = frames_full[~np.isin(frames_full, time_frames)]
    # time_shift_df = pd.DataFrame(np.c_[frames_full[:, None], np.zeros((len(frames_full), 3))],
    #                              columns=["frame", "xs", "ys", "zs"])

    # generate shift arrays
    side1_shifts = time_shift_df.copy()
    side2_shifts = time_shift_df.copy() + half_shift_df.copy()

    # initialize fusion function
    fuse_to_run = partial(fuse_and_filter, mask1=mask1, mask2=mask2, mask_out=fused_mask_zarr,
                          side1_shifts=side1_shifts, side2_shifts=side2_shifts, dist_thresh=dist_thresh)

    if par_flag:
        process_map(fuse_to_run, write_indices, max_workers=n_workers, chunksize=1)
    else:
        for frame in tqdm(write_indices):
            fuse_to_run(frame)

    
if __name__ == "__main__":

    last_i = 410

    # load masks
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_prefix = "20250311_LCP1-NLSMSC"

