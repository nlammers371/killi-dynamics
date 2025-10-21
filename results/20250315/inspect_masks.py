import zarr
import napari
import os
import numpy as np
from src.segmentation import Lo

import os
os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project = "20250311_LCP1-NLSMSC_side2"
# project2 = "20250311_LCP1-NLSMSC_side2"
zpath = os.path.join(root, "built_data", "zarr_image_files", project + ".zarr")
mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_stacks.zarr")
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")
# load
im_full = zarr.open(zpath, mode="r")
mask_full = zarr.open(mpath, mode="r")

# generate frame indices
t_int = 2250
nucleus_channel = 1

# get scale info
scale_vec = tuple([im_full.attrs['PhysicalSizeZ'], im_full.attrs['PhysicalSizeY'], im_full.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
im = np.squeeze(im_full[t_int, nucleus_channel])
mask = np.squeeze(mask_full[t_int])

# remove small regions
# from skimage.measure import regionprops, label
# from scipy.spatial import distance_matrix
# from scipy.sparse import csr_matrix
#
# regions = regionprops(mask, spacing=scale_vec)
#
# ##################
# # step 1 size-based filtering
# area_vec = np.asarray([rg["Area"] for rg in regions])
# lb_vec = np.asarray([rg["label"] for rg in regions])
# keep_labels = lb_vec[area_vec > 250]
# size_mask = np.multiply(mask, np.isin(mask, keep_labels))
#
# # generate new mask
# mask01 = label(size_mask)
#
# ##################
# # step 2: z projection-based filtering
# regions01 = regionprops(mask01)
#
# centroid_vec = np.asarray([rg["Centroid"] for rg in regions01])
# # euc_dist_mat = distance_matrix(centroid_vec, centroid_vec)
# z_dist_mat = (centroid_vec[:, 0][:, None] - centroid_vec[:, 0][None, :]) * scale_vec[0]
#
# z_thresh = -30
# # z_dir_filter = (np.divide(np.abs(z_dist_mat), euc_dist_mat)) >= 0.5
# z_shadow_candidates = (z_dist_mat < 0) & (z_dist_mat >= z_thresh)
#
# # from scipy import sparse
# # Remove background if needed; here we assume labels > 0 are cells.
#
# # Assume masks_3d is your 3D labeled array (Z, Y, X)
# Z, Y, X = mask01.shape
#
# # Generate regionprops objects for each cell label (3D masks)
# n_cells = len(regions01)
#
# # Prepare a sparse projection array to minimize memory usage
# proj_masks_flat = []
#
# # Loop through regionprops objects, project their smaller bounding box regions
# for prop in tqdm(regions01, "Scanning for refraction artifacts..."):
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
#
# # Stack all flat projections into a 2D sparse matrix for efficient intersection calculations
# proj_matrix = csr_matrix(np.array(proj_masks_flat, dtype=np.uint16))  # shape: (n_cells, Y*X)
#
# # Compute intersection via sparse matrix multiplication (fast)
# intersections = (proj_matrix @ proj_matrix.T).toarray()  # shape: (n_cells, n_cells)
#
# # Compute area for each mask
# areas = proj_matrix.sum(axis=1).A1  # A1 gives a flattened array
#
# # Intersection over Union (IoU)
# union = areas[:, None] + areas[None, :] - intersections
# iou_matrix = intersections / union
#
# # Fraction of overlap (intersection / area of the first mask)
# fraction_overlap = intersections / areas[:, None]
#
# # now, start from z=0 and move forward, flagging potential artifacts along the way
# shadow_flag_vec = np.zeros((n_cells,), dtype=np.bool_)
# min_overlap = 0.25
# # import os
# # os.environ["QT_API"] = "pyqt5"
# for i in range(n_cells):
#     z = centroid_vec[i, 0]
#     # zi = z_order[i]
#     if not shadow_flag_vec[i]:
#         z_filter = centroid_vec[:, 0] > z  # will remove self interaction
#         s_filter = z_shadow_candidates[i, :].copy()
#         o_filter = fraction_overlap[:, i] >= min_overlap
#         s_candidates = z_filter & s_filter & o_filter
#
#         if np.any(s_candidates):
#             shadow_flag_vec[s_candidates] = True
#
# shadow_labels = np.asarray([regions01[n]["label"] for n in range(n_cells) if shadow_flag_vec[n]])
# shadow_mask = np.isin(mask01, shadow_labels)

viewer = napari.Viewer()

viewer.add_image(im, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
viewer.add_labels(mask, scale=scale_vec)
# viewer.add_labels(mask01, scale=scale_vec)
# viewer.add_labels(shadow_mask, scale=scale_vec)
napari.run()

