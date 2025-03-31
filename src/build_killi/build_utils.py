import zarr
import os
import pandas as pd
import scipy.ndimage as ndi
import numpy as np
from tqdm import tqdm
import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import Store
from skimage.measure import regionprops, label
from scipy.optimize import least_squares
from scipy.spatial import distance_matrix
from src.build_killi.process_masks import ellipsoid_axis_lengths
from scipy.special import sph_harm
from functools import partial
from tqdm.contrib.concurrent import process_map
import multiprocessing
from scipy.interpolate import interp1d
from ultrack.utils.array import create_zarr
from ultrack.utils.cuda import import_module, to_cpu
from typing import Optional, Sequence, Tuple, Union
import skimage.segmentation as segm
import scipy.ndimage as ndi
from glob2 import glob
from scipy.ndimage import mean as ndi_mean

def fuse_images(frame, image1, image2, side1_shifts, side2_shifts, fuse_channel=None):
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
    image_fused = np.multiply(image1_shifted, side1_weights[:, np.newaxis, np.newaxis]) + \
                  np.multiply(image2_shifted, side2_weights[:, np.newaxis, np.newaxis])

    return image_fused

def integrate_fluorescence(t, image_zarr, image_zarr1, image_zarr2, side1_shifts, side2_shifts, mask_zarr,
                           fluo_channel, out_folder):
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
    if image_zarr is not None:
        im_frame = np.squeeze(image_zarr[t, fluo_channel])
        meta = image_zarr.attrs
    elif image_zarr2 is not None:
        # print("Fusing images for frame", t)
        meta = image_zarr1.attrs
        im_frame = fuse_images(t, image_zarr1, image_zarr2, side1_shifts, side2_shifts, fuse_channel=fluo_channel)
    else:
        raise Exception("No image zarr provided.")

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
    fluo_df[["z", "y", "x"]] = centroids
    # fluo_mean = ndi_mean(im_frame, labels=mask_frame, index=label_vec)
    # fluo_df["mean_fluo"] = fluo_mean
    for p, pr in enumerate(props):
        # get array indices corresponding to nucleus mask from props
        coords = pr.coords
        # get the mean fluorescence value for each nucleus using coordinates
        fluo_df.loc[fluo_df["nucleus_id"] == pr.label, "mean_fluo"] = np.mean(im_frame[coords[:, 0], coords[:, 1], coords[:, 2]])

    # save
    output_file = os.path.join(out_folder, f"fluorescence_data_frame_{t:04}.csv")
    fluo_df.to_csv(output_file, index=False)

    return fluo_df

def transfer_fluorescence(frame_i, fluo_df_list, tracked_mask_zarr, mask_zarr):
    # do the label transfer
    fluo_df_frame = fluo_df_list[frame_i]
    props = regionprops(tracked_mask_zarr[frame_i])
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
    temp_df["t"] = frame_i
    temp_df["mean_fluo"] = fluo_vec

    return temp_df

def transfer_fluorescence_wrapper(root, project_name, fused_flag=True, tracking_config=None, tracking_range=None,
                                  suffix="", well_num=0, overwrite=False, n_workers=None, par_flag=False):

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
    fluo_path = os.path.join(root, "built_data", "fluorescence_data", project_name, "")
    fluo_df_path_list = sorted(glob(fluo_path + "*.csv"))
    fluo_df_list = []
    for fluo_file in tqdm(fluo_df_path_list, "Loading fluorescence data files..."):
        fluo_df = pd.read_csv(fluo_file)
        fluo_df_list.append(fluo_df)

    # combine all fluorescence dataframes
    # fluo_df = pd.concat(fluo_df_list, ignore_index=True)

    # path to raw masks
    if fused_flag:
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

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    # load tracking masks
    label_path = os.path.join(project_sub_path, "segments.zarr")
    tracked_mask_zarr = zarr.open(label_path, mode='r')

    # load dataframe of track info
    tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))
    # tracks_df_fluo = tracks_df.copy()

    transfer_run = partial(transfer_fluorescence, fluo_df_list=fluo_df_list, tracked_mask_zarr=tracked_mask_zarr,
                           mask_zarr=mask_zarr)

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
                                   par_flag=True, start_i=0, stop_i=None, n_workers=None):
                                   # tracking_config=None, suffix="", well_num=0):
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
    output_path = os.path.join(root, "built_data", "fluorescence_data", project_name, "")
    os.makedirs(output_path, exist_ok=True)

    # path to raw masks

    # if tracking_config is not None:
    #
    #     # get name
    #     tracking_name = tracking_config.replace(".txt", "")
    #
    #     # set output path for tracking results
    #     project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    #     project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")
    #     mask_zarr_path = os.path.join(project_sub_path, "segments.zarr")

    # define paths
    if fused_flag:
        image_zarr_path1 = os.path.join(root, "built_data", "zarr_image_files", project_name + "_side1.zarr")
        image_zarr_path2 = os.path.join(root, "built_data", "zarr_image_files", project_name + "_side2.zarr")
        image_zarr = None
        image_zarr1 = zarr.open(image_zarr_path1, mode='r')
        image_zarr2 = zarr.open(image_zarr_path2, mode='r')
        # if tracking_config is None:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_fused.zarr")
    else:
        image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", project_name + ".zarr")
        image_zarr = zarr.open(image_zarr_path, mode='r')
        image_zarr1 = None
        image_zarr2 = None
        # if tracking_config is None:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", project_name + "_mask_aff.zarr")

    # load zarr files
    mask_zarr = zarr.open(mask_zarr_path, mode='r')

    if fused_flag:
        # load shift info
        metadata_path = os.path.join(root, "metadata", project_name + "_side1", "")
        half_shift_df = pd.read_csv(os.path.join(metadata_path, project_name + "_side2" + "_to_" + project_name + "_side1" + "_shift_df.csv"))
        # time_shift_df = pd.read_csv(os.path.join(metadata_path, "frame_shift_df.csv"))

        # generate shift arrays
        side2_shifts = half_shift_df.copy() #+ time_shift_df.copy()
        side1_shifts = half_shift_df.copy()
        side1_shifts[["zs", "ys", "xs"]] = 0  # no shift for side1
    else:
        side2_shifts = None
        side1_shifts = None

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
            partial(integrate_fluorescence, image_zarr=image_zarr, mask_zarr=mask_zarr, fluo_channel=fluo_channel, out_folder=output_path,
                    side2_shifts=side2_shifts, side1_shifts=side1_shifts, image_zarr1=image_zarr1, image_zarr2=image_zarr2),
            write_indices,
            max_workers=n_workers,
            chunksize=1
        )
    else:
        print("Using sequential processing")
        # Sequential processing
        results = []
        for t in tqdm(write_indices):
            result = integrate_fluorescence(t, image_zarr=image_zarr, mask_zarr=mask_zarr, fluo_channel=fluo_channel, out_folder=output_path,
                    side2_shifts=side2_shifts, side1_shifts=side1_shifts, image_zarr1=image_zarr1, image_zarr2=image_zarr2)
            results.append(result)

    # # combine
    # fluo_df = pd.concat(results, ignore_index=True)
    #
    #
    return True

def sh_mask_filter(mask, scale_vec, L_max=15, mesh_res=100, max_surf_dist=30, area_thresh=500):

    # get mask locations
    props = regionprops(mask, spacing=scale_vec)
    points = np.array([prop.centroid for prop in props])
    label_vec = np.array([prop.label for prop in props])
    area_vec = np.array([prop.area for prop in props])

    # fit sphere and get SH info
    coeffs, fitted_center, fitted_radius = fit_sphere_and_sh(points, L_max=L_max)
    sphere_mesh = create_sphere_mesh(fitted_center, fitted_radius, resolution=mesh_res)

    # get sh mesh
    sh_mesh, r_sh = create_sh_mesh(coeffs, sphere_mesh)

    # get distances
    surf_dist_mat = distance_matrix(points, sh_mesh[0])
    surf_dist_vec = np.min(surf_dist_mat, axis=1)

    outlier_filter = (surf_dist_vec > max_surf_dist) & (area_vec < area_thresh)
    outlier_labels = label_vec[outlier_filter]  # threshold for outliers
    inlier_mask = label(~np.isin(mask, outlier_labels) & (mask > 0))  # create mask for outliers

    return inlier_mask

def label_fun(t, write_indices, labels, foreground, contours, shape, last_filter_start_i, scale_vec, sigma=None):

    foreground_frame = np.zeros(shape[1:], dtype=foreground.dtype)
    contours_frames = np.zeros(shape[1:], dtype=contours.dtype)

    lb_frame = np.asarray(labels[t])

    # get scale info
    if scale_vec is None:
        scale_vec = (1.0, 1.0, 1.0)  # default scale if not provided

    if last_filter_start_i is not None and t >= last_filter_start_i:
        lb_frame = sh_mask_filter(lb_frame, scale_vec)

    foreground_frame |= lb_frame > 0
    contours_frames += segm.find_boundaries(lb_frame, mode="outer")

    contours_frames /= len(labels)

    if sigma is not None:
        contours_frames = ndi.gaussian_filter(contours_frames, sigma)
        contours_frames = contours_frames / contours_frames.max()

    out_index = np.where(write_indices == t)[0][0]  # find the index in write_indices
    foreground[out_index] = to_cpu(foreground_frame)
    contours[out_index] = to_cpu(contours_frames)


def labels_to_contours_nl(
    labels: Union[ArrayLike, Sequence[ArrayLike]],
    write_indices: Sequence[int],
    sigma: Optional[Union[Sequence[float], float]] = None,
    foreground_store_or_path: Union[Store, str, None] = None,
    contours_store_or_path: Union[Store, str, None] = None,
    overwrite: bool = False,
    n_workers: Optional[int] = None,
    par_flag: bool = True,
    last_filter_start_i: Optional[int] = None,
    scale_vec: Optional[Tuple[float, float, float]] = None
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts and merges a sequence of labels into ultrack input format (foreground and contours)

    Parameters
    ----------
    labels : Union[ArrayLike, Sequence[ArrayLike]]
        List of labels with equal shape.
    sigma : Optional[Union[Sequence[float], float]], optional
        Contours smoothing parameter (gaussian blur), contours aren't smoothed when not provided.
    foreground_store_or_path : str, zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.NestedDirectoryStorage to save the output into disk.
        By default it loads the data into memory.
    contours_store_or_path : str, zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.NestedDirectoryStorage to save the output into disk.
        By default it loads the data into memory.
    overwrite : bool, optional
        Overwrite output output files if they already exist, by default False.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Combined foreground and edges arrays.
    """
    # ndi = import_module("scipy", "ndimage")
    # segm = import_module("skimage", "segmentation")

    # if not isinstance(labels, Sequence):
    #     labels = [labels]
    if isinstance(labels, Sequence):
        raise ValueError("Function is not yet compatible with multiple lablels per image")

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)


    shape = (len(write_indices),) + labels.shape[1:]
    # for lb in labels:
    #     if shape != lb.shape:
    #         raise ValueError(
    #             f"All labels must have the same shape. Found {shape} and {lb.shape}"
    #         )

    foreground = create_zarr(
        shape=shape,
        dtype=bool,
        store_or_path=foreground_store_or_path,
        overwrite=overwrite,
        default_store_type=zarr.TempStore,
    )
    contours = create_zarr(
        shape=shape,
        dtype=np.float32,
        store_or_path=contours_store_or_path,
        overwrite=overwrite,
        default_store_type=zarr.TempStore,
    )

    label_fun_run = partial(label_fun, labels=labels, foreground=foreground, contours=contours, shape=shape,
                            sigma=sigma, last_filter_start_i=last_filter_start_i, scale_vec=scale_vec,
                            write_indices=write_indices)

    if par_flag:
        print("Using parallel processing")
        process_map(label_fun_run, write_indices, max_workers=n_workers, chunksize=1)
    else:
        print("Using sequential processing")
        for t in [2300]: #tqdm(write_indices):
            label_fun_run(t)

    return foreground, contours

def build_sh_basis(L_max, phi, theta):
    basis_functions = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # Note: sph_harm takes (m, l, phi, theta)
            Y_lm = sph_harm(m, l, phi, theta)
            basis_functions.append(Y_lm.real)

    return np.column_stack(basis_functions).T

# Assuming vertices is an (N, 3) array and radial_distances is the corresponding data.
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # colatitude
    phi = np.arctan2(y, x)
    return r, theta, phi

def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def create_sphere_mesh(center, radius, resolution=50):
    # Create a grid of angles
    phi, theta = np.mgrid[0.0:np.pi:complex(0, resolution), 0.0:2.0 * np.pi:complex(0, resolution)]

    # Parametric equations for a sphere
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    # Create vertices array
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Create faces: two triangles for each square in the grid
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx0 = i * resolution + j
            idx1 = idx0 + 1
            idx2 = idx0 + resolution
            idx3 = idx2 + 1
            faces.append([idx0, idx2, idx1])
            faces.append([idx1, idx2, idx3])
    faces = np.array(faces)
    return vertices, faces

def fit_sphere(points, quantile=0.95):
    """
    Fits a sphere to the given 3D points.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        mode (str): 'inner', 'outer', or 'average'.
        quantile (float): quantile for inner/outer fitting (0-1).

    Returns:
        center (np.ndarray): sphere center coordinates.
        radius (float): radius of the fitted sphere.
    """
    # Initial guess for sphere center: centroid of points
    center_init = np.mean(points, axis=0)

    def residuals(c):
        r = np.linalg.norm(points - c, axis=1)
        # if mode == 'average':
        return r - r.mean()
        # elif mode == 'inner':
        #     target_radius = np.quantile(r, quantile)
        #     return r - target_radius
        # elif mode == 'outer':
        #     target_radius = np.quantile(r, quantile)
        #     return r - target_radius
        # else:
        #     raise ValueError("mode must be 'average', 'inner', or 'outer'.")

    result = least_squares(residuals, center_init)
    fitted_center = result.x

    # Compute final radius based on chosen mode
    final_distances = np.linalg.norm(points - fitted_center, axis=1)
    fitted_radius = final_distances.mean()
    inner_radius = np.quantile(final_distances, 1 - quantile)
    outer_radius = np.quantile(final_distances, quantile)

    return fitted_center, fitted_radius, inner_radius, outer_radius

def create_sh_mesh(coeffs, sphere_mesh):
    """
    Evaluates the spherical harmonics on the sphere mesh.

    Parameters:
        coeffs (np.ndarray): coefficients of the spherical harmonics.
        sphere_mesh (tuple): vertices and faces of the sphere mesh.
        radius (float): radius of the sphere.

    Returns:
        np.ndarray: evaluated values on the sphere mesh.
    """

    vertices, _ = sphere_mesh

    mesh_center = np.mean(vertices, axis=0)  # center of the sphere mesh
    vertices_c = vertices - mesh_center  # shift vertices to center

    # Convert Cartesian coordinates to spherical coordinates
    r, theta, phi = cart2sph(vertices_c[:, 0], vertices_c[:, 1], vertices_c[:, 2])

    L_max = int(np.sqrt(len(coeffs)) - 1)
    # Build the basis functions
    basis_functions = build_sh_basis(L_max, phi=phi, theta=theta)

    # Evaluate the spherical harmonics
    r_sh = coeffs[None, :] @ basis_functions

    # get new caresian points
    x, y, z = sph2cart(r_sh, theta, phi)
    vertices_sh = np.c_[x.T, y.T, z.T] + mesh_center  # combine x, y, z into a single array

    # define SH mesh
    sh_mesh = (vertices_sh, sphere_mesh[1])  # keep the same faces as the original sphere mesh
    # sh_mesh.vertices = vertices_sh  # update vertices with SH values

    return sh_mesh, r_sh


# write function to fit spherical harmoncs to deviations from sphere surface
def fit_sphere_and_sh(points, L_max=10, knn=3, k_thresh=50):
    """
    Fits spherical harmonics to the deviations of points from a sphere.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        radius (float): radius of the sphere.
        order (int): maximum order of spherical harmonics.

    Returns:
        coeffs (np.ndarray): coefficients of the fitted spherical harmonics.
    """
    # first, fit a sphere to the points
    fitted_center, fitted_radius, inner_radius, outer_radius = fit_sphere(points)

    # shift points to center
    points_c = points - fitted_center

    # Convert Cartesian coordinates to spherical coordinates
    r, theta, phi = cart2sph(points_c[:, 0], points_c[:, 1], points_c[:, 2])

    # Generate the mesh for the sphere:
    vertices, faces = create_sphere_mesh(np.asarray([0, 0, 0]), fitted_radius, resolution=100)
    r_v, theta_v, phi_v = cart2sph(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # map centroids to sphere vertices
    surf_dist_mat = distance_matrix(vertices, points_c)
    closest_indices = np.argsort(surf_dist_mat, axis=1)[:, :knn]
    closest_distances = np.sort(surf_dist_mat, axis=1)[:, :knn]
    r_dist_array = r[closest_indices]
    r_dist_array[closest_distances > k_thresh] = np.nan
    radial_distances = np.nanmean(r_dist_array, axis=1)

    # Compute spherical harmonics
    basis_functions = build_sh_basis(L_max, theta=theta_v, phi=phi_v)

    # Create the design matrix (N, num_basis)
    Y = np.column_stack(basis_functions)

    nan_filter = ~np.isnan(radial_distances)
    Y_filtered = Y[nan_filter, :]
    radial_distances_filtered = radial_distances[nan_filter]

    # Solve for coefficients using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(Y_filtered, radial_distances_filtered, rcond=None)

    # Evaluate the fitted function at the vertices
    # fitted_radial = Y @ coeffs

    return np.array(coeffs), fitted_center, fitted_radius


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

