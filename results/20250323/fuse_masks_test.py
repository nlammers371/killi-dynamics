import zarr
import os
import pandas as pd
import scipy.ndimage as ndi
import numpy as np
from tqdm import tqdm
import napari
from skimage.measure import regionprops, label
from scipy.optimize import least_squares
from scipy.spatial import distance_matrix


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


if __name__ == "__main__":

    last_i = 2200

    # load masks
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_prefix = "20250311_LCP1-NLSMSC"

    project1 = project_prefix + "_side1"
    project2 = project_prefix + "_side2"

    # load mask zarr file
    mpath1 = os.path.join(root, "built_data", "mask_stacks", project1 + "_mask_aff_orig.zarr")
    mask1 = zarr.open(mpath1, mode="r")

    mpath2 = os.path.join(root, "built_data", "mask_stacks", project2 + "_mask_aff_orig.zarr")
    mask2 = zarr.open(mpath2, mode="r")

    if last_i is None:
        last_i = mask1.shape[0]

    # get scale info
    scale_vec = tuple([mask1.attrs['PhysicalSizeZ'],
                       mask1.attrs['PhysicalSizeY'],
                       mask1.attrs['PhysicalSizeX']])

    # load shift info
    data_path = os.path.join(root, "metadata", project1, "")
    half_shift_df = pd.read_csv(os.path.join(data_path, project2 + "_to_" + project1 + "_shift_df.csv"))
    time_shift_df = pd.read_csv(os.path.join(data_path, "frame_shift_df.csv"))

    # generate shift arrays
    side1_shifts = time_shift_df.copy()
    side2_shifts = time_shift_df.copy() + half_shift_df.copy()

    # subset frames
    frames_to_align = np.arange(2180, last_i)

    # extract ID dictionary
    # keep_dict1 = mask1.attrs["mask_keep_ids"]
    # keep_dict2 = mask2.attrs["mask_keep_ids"]

    # initialze 'full' array
    zdim1_orig = mask1.shape[1]
    zdim2_orig = mask2.shape[1]
    full_z = zdim1_orig + zdim2_orig  # int(np.ceil((zdim1_orig + zdim2_orig) / 10) * 10)
    full_shape = tuple([full_z]) + tuple(mask1.shape[2:])

    masks_fused = np.zeros(tuple([len(frames_to_align)]) + full_shape, dtype=np.uint16)

    # generate fused mask frames
    frame = 1800
    frame_i = frames_to_align == frame

    shift1 = side1_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()
    shift2 = side2_shifts.loc[frame, ["zs", "ys", "xs"]].to_numpy()

    # data_zyx1 = np.squeeze(mask1[frame_i])
    # data_zyx2 = np.squeeze(mask2[frame_i][::-1, :, ::-1])

    # extract masks
    m1 = np.squeeze(mask1[frame])
    m2 = np.squeeze(mask2[frame])

    # filter
    keep_ids1 = np.unique(m1)[1:]#keep_dict1[str(frame)]
    m1_binary = np.isin(m1, keep_ids1)
    keep_ids2 = np.unique(m2)[1:] #keep_dict2[str(frame)]
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
    props = regionprops(mask_fused, spacing=scale_vec)
    centroid_array = np.asarray([pr["Centroid"] for pr in props])
    fitted_center, fitted_radius, inner_radius, outer_radius = fit_sphere(centroid_array)

    # Generate the mesh for the sphere:
    vertices, faces = create_sphere_mesh(fitted_center, inner_radius, resolution=50)

    viewer = napari.Viewer()

    # viewer.add_image(im, scale=scale_vec, colormap="gray", contrast_limits=[0, 2500])
    viewer.add_labels(mask_fused, scale=scale_vec)

    radial_dist_vec = np.sqrt(np.sum((centroid_array - fitted_center) ** 2, axis=1))
    # map centroids to sphere vertices
    surf_dist_mat = distance_matrix(vertices, centroid_array)
    closest_indices = np.argsort(surf_dist_mat, axis=1)[:, :5]
    dist_vec = np.mean(radial_dist_vec[closest_indices], axis=1) - fitted_radius

    # use mean closest points to screen for outliers
    point_to_surf_mapping = np.argmin(surf_dist_mat.T, axis=1)
    ref_dist_vec = dist_vec[point_to_surf_mapping]
    surf_delta_vec = radial_dist_vec - (ref_dist_vec + fitted_radius)

    radius_mask = np.zeros(mask_fused.shape, dtype=np.float32)
    for i in tqdm(range(len(radial_dist_vec))):
        coords = props[i].coords
        radius_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = surf_delta_vec[i]

    outlier_mask = radius_mask > 30
    mask_filtered = label((radius_mask < 30) & (mask_fused > 0))

    # add surface
    viewer.add_surface((vertices, faces, dist_vec), name='Fitted Sphere', opacity=0.5)
    viewer.add_labels(mask_filtered, scale=scale_vec)