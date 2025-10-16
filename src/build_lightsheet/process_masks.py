import zarr
import os
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, label
from scipy.sparse import csr_matrix, coo_matrix
from tqdm.contrib.concurrent import process_map
from functools import partial
import multiprocessing
from filelock import FileLock
from skimage.morphology import remove_small_objects
import math


def ellipsoid_axis_lengths(central_moments):
    """
    Compute ellipsoid major, intermediate, and minor axis lengths for multiple masks.

    Parameters
    ----------
    central_moments : ndarray
        Array of central moments for multiple masks, with shape (N, 3, 3, 3),
        as given by `moments_central` with order 2 for each mask.

    Returns
    -------
    radii : ndarray
        Array of ellipsoid semi-axis lengths in descending order for each mask.
        Shape is (N, 3); each row corresponds to (major, intermediate, minor) axis.
    S : ndarray
        Covariance matrices computed for each mask, with shape (N, 3, 3).
    """
    # Extract the zeroth moment for each mask (total mass/volume)
    m0 = central_moments[:, 0, 0, 0]

    # Compute normalized second-order moments
    sxx = central_moments[:, 2, 0, 0] / m0
    syy = central_moments[:, 0, 2, 0] / m0
    szz = central_moments[:, 0, 0, 2] / m0
    sxy = central_moments[:, 1, 1, 0] / m0
    sxz = central_moments[:, 1, 0, 1] / m0
    syz = central_moments[:, 0, 1, 1] / m0

    # Construct the covariance matrix S for each mask
    # S has shape (N, 3, 3)
    S = np.stack([
        np.stack([sxx, sxy, sxz], axis=1),
        np.stack([sxy, syy, syz], axis=1),
        np.stack([sxz, syz, szz], axis=1)
    ], axis=1)

    # Compute the eigenvalues of S for each mask
    # np.linalg.eigvalsh is vectorized and returns an array of shape (N, 3)
    eigvals = np.linalg.eigvalsh(S)
    # Sort eigenvalues in descending order for each mask
    eigvals = np.sort(eigvals, axis=1)[:, ::-1]

    # For a homogeneous ellipsoid, the variance along a principal axis is related to
    # the semi-axis length squared by a factor of 1/5.
    # Therefore, the semi-axis lengths are given by sqrt(5 * eigenvalue)
    radii = np.sqrt(5 * eigvals)

    return radii, S


def perform_mask_qc(t_int, mask_in, zarr_path, scale_vec, min_nucleus_vol, z_prox_thresh, max_eccentricity, min_overlap):

    mask = np.squeeze(mask_in[t_int])

    # regions = regionprops(mask, spacing=scale_vec)#, extra_properties=('area', 'label'))
    # regions = regionprops_table(mask, spacing=scale_vec, properties=('area', 'label'))
    ##################
    # step 1 size-based filtering
    # area_vec = np.asarray([rg["area"] for rg in regions])
    area_vec = np.bincount(mask.ravel()) * np.product(scale_vec)
    lb_vec = np.arange(len(area_vec))[1:]
    if len(lb_vec) > 20000:
        raise Exception(f"Too many masks found in dataset ({len(lb_vec)}). Revisit your li threshold")

    area_vec = area_vec[1:]
    # area_vec = np.array(regions['area'])
    # lb_vec = np.asarray([rg["label"] for rg in regions])
    # lb_vec = np.array(regions['label'])
    keep_labels = lb_vec[area_vec > min_nucleus_vol]
    size_mask = np.multiply(mask, np.isin(mask, keep_labels))
    # size_mask = remove_small_objects(mask, min_size=min_num_pixels
    #                                  )
    # del regions

    # generate new mask
    mask01 = label(size_mask)

    ##################
    # step 2: z projection-based filtering
    regions01 = regionprops(mask01)

    centroid_vec = np.asarray([rg["Centroid"] for rg in regions01])
    # euc_dist_mat = distance_matrix(centroid_vec, centroid_vec)
    z_dist_mat = (centroid_vec[:, 0][:, None] - centroid_vec[:, 0][None, :]) * scale_vec[0]

    # z_dir_filter = (np.divide(np.abs(z_dist_mat), euc_dist_mat)) >= 0.5
    z_shadow_candidates = (z_dist_mat < 0) & (z_dist_mat >= z_prox_thresh)

    # Assume masks_3d is your 3D labeled array (Z, Y, X)
    Z, Y, X = mask01.shape

    # Generate regionprops objects for each cell label (3D masks)
    n_cells = len(regions01)

    # Prepare a sparse projection array to minimize memory usage
    # proj_masks_flat = []
    #
    # #Loop through regionprops objects, project their smaller bounding box regions
    # for prop in regions01: #, "Scanning for refraction artifacts..."):
    #     # prop.image is a smaller boolean mask (cropped to bbox)
    #     zmin, ymin, xmin, zmax, ymax, xmax = prop.bbox
    #
    #     # Perform the projection within the smaller cropped region (along Z axis)
    #     proj_small = np.any(prop.image, axis=0)  # shape: (bbox_y, bbox_x)
    #
    #     # Now, embed proj_small back into the full-size flattened array (Y, X)
    #     proj_full = np.zeros((Y, X), dtype=bool)
    #     proj_full[ymin:ymax, xmin:xmax] = proj_small
    #
    #     # Flatten and store (as sparse vector if desired)
    #     proj_masks_flat.append(proj_full.ravel())

    # Stack all flat projections into a 2D sparse matrix for efficient intersection calculations
    # proj_matrix = csr_matrix(np.array(proj_masks_flat, dtype=np.uint16))  # shape: (n_cells, Y*X)

    # Compute intersection via sparse matrix multiplication (fast)
    # intersections = (proj_matrix @ proj_matrix.T).toarray()  # shape: (n_cells, n_cells)

    # --- Build sparse projection matrix in a memory‐efficient way ---
    # Instead of storing full flattened projections, we accumulate indices of True entries.
    row_indices = []
    col_indices = []
    data = []
    for i, prop in enumerate(regions01):
        zmin, ymin, xmin, zmax, ymax, xmax = prop.bbox
        # Create projection from the cropped image along Z.
        proj_small = np.any(prop.image, axis=0)  # shape: (bbox_y, bbox_x)
        # Create an empty full-size projection.
        # Instead of embedding the entire image, record only indices where True.
        if proj_small.any():
            # Find indices (local to the bbox) where projection is True.
            local_ys, local_xs = np.nonzero(proj_small)
            # Convert local indices to full-image indices.
            full_ys = local_ys + ymin
            full_xs = local_xs + xmin
            flat_indices = full_ys * X + full_xs
            row_indices.extend([i] * len(flat_indices))
            col_indices.extend(flat_indices)
            data.extend([1] * len(flat_indices))

    # Build sparse matrix directly (shape: n_cells x (Y*X))
    proj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n_cells, Y * X)).tocsr()
    #
    # # Compute intersections using sparse matrix multiplication
    intersections = (proj_matrix @ proj_matrix.T).toarray()
    # Compute area for each mask
    areas = proj_matrix.sum(axis=1).A1  # A1 gives a flattened array

    # Fraction of overlap (intersection / area of the first mask)
    fraction_overlap = intersections / areas[:, None]

    # now, start from z=0 and move forward, flagging potential artifacts along the way
    shadow_flag_vec = np.zeros((n_cells,), dtype=np.bool_)

    # import os
    # os.environ["QT_API"] = "pyqt5"
    for i in range(n_cells):
        z = centroid_vec[i, 0]
        # zi = z_order[i]
        if not shadow_flag_vec[i]:
            z_filter = centroid_vec[:, 0] > z  # will remove self interaction
            s_filter = z_shadow_candidates[i, :].copy()
            o_filter = fraction_overlap[:, i] >= min_overlap
            s_candidates = z_filter & s_filter & o_filter

            if np.any(s_candidates):
                shadow_flag_vec[s_candidates] = True

    keep_labels01 = np.asarray([regions01[n]["label"] for n in range(n_cells) if ~shadow_flag_vec[n]])

    # flag and remove masks that are too small along smallest dimension
    mask02 = label(np.isin(mask01, keep_labels01))
    props02 = regionprops(mask02, spacing=scale_vec)

    moment_array = np.stack([pr["moments_central"] for pr in props02], axis=0)
    radii, _ = ellipsoid_axis_lengths(moment_array)

    e_flags = np.divide(radii[:, 0], radii[:, 1]+1e-4) <= max_eccentricity
    e_flags = e_flags & (radii[:, 1] > 2)
    keep_labels02 = np.asarray([props02[p]["label"] for p in range(len(props02)) if e_flags[p]])

    # recover original labels
    keep_labels = np.unique(mask[np.isin(mask02, keep_labels02)])

    # Define a lock file based on your zarr path
    lock_path = zarr_path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        # Open the zarr group for update
        group = zarr.open(zarr_path, mode='a')
        # Get existing metadata dictionary or initialize one
        meta = group.attrs.get("mask_keep_ids", {})
        meta[str(t_int)] = keep_labels.tolist()  # update for current frame
        group.attrs["mask_keep_ids"] = meta

    # mask_out[t_int] = mask02

    # --- Clean up intermediate variables to free memory ---
    # del regions, area_vec, lb_vec, keep_labels, size_mask, regions01, centroid_vec
    # del z_dist_mat, z_shadow_candidates, proj_matrix
    # del intersections, areas, fraction_overlap, shadow_flag_vec   #, shadow_labels
    # gc.collect()

    return True


def mask_qc_wrapper(root, project, min_nucleus_vol=75, z_prox_thresh=-30, max_eccentricity=4.5, min_shadow_overlap=0.35,
                    last_i=None, overwrite=False, par_flag=False, n_workers=None):

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 2)

    # load mask zarr file
    mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_aff.zarr")
    mask_full = zarr.open(mpath, mode="a")

    if last_i is None:
        last_i = mask_full.shape[0]

    # get scale info
    scale_vec = tuple([mask_full.attrs['PhysicalSizeZ'],
                       mask_full.attrs['PhysicalSizeY'],
                       mask_full.attrs['PhysicalSizeX']])

    # get all indices
    all_indices = set(range(last_i))

    if "mask_keep_ids" in list(mask_full.attrs.keys()):
        meta = mask_full.attrs["mask_keep_ids"]
        # List files directly within zarr directory (recursive search):
        extant_indices = set([int(key) for key in list(meta.keys())])
    else:
        extant_indices = set([])

    empty_indices = np.asarray(sorted(all_indices - extant_indices))

    if overwrite:
        write_indices = np.asarray(list(all_indices))
    else:
        write_indices = empty_indices

    # iterate through time points
    mask_qc_run = partial(perform_mask_qc, mask_in=mask_full, zarr_path=mpath, scale_vec=scale_vec, min_overlap=min_shadow_overlap,
                          min_nucleus_vol=min_nucleus_vol, z_prox_thresh=z_prox_thresh, max_eccentricity=max_eccentricity)

    if par_flag:
        m_list = process_map(mask_qc_run, write_indices, max_workers=n_workers, chunksize=1)
    else:
        m_list = []
        for t in tqdm(write_indices):
            mt = mask_qc_run(t)
            m_list.append(mt)

    # for t, m_array in enumerate(m_list):
    #     mask_zarr_clean[t] = m_array

    return m_list


if __name__ == "__main__":

    # get filepaths
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project = "20250311_LCP1-NLSMSC_side2"

    mask_qc_wrapper(root, project)

