from src.geometry.sphere import fit_sphere
import numpy as np
from typing import Iterable
from skimage.measure import regionprops_table
from scipy.spatial import distance_matrix

def filter_by_surf_distance(mask: np.ndarray,
                            scale_vec: Iterable[float],
                            max_surf_dist: float = 50.0,
                            rad_pct: float = .25, rad_um: float = 15.0) -> tuple[np.ndarray, np.ndarray]:

        props = regionprops_table(mask, spacing=scale_vec, properties=("centroid", "label"))
        points_phys = np.column_stack((
                                        props["centroid-0"],
                                        props["centroid-1"],
                                        props["centroid-2"],
                                    ))
        labels = np.asarray(props["label"])
        center, rad, rad_inner = fit_sphere(
            points_phys,
            rad_quantile=rad_pct
        )
        dists = np.sqrt(np.sum((points_phys - center[None, :]) ** 2, axis=1))
        keep_labels = labels[np.abs(dists - rad_inner) <= max_surf_dist]
        filtered = np.where(np.isin(mask, keep_labels), mask, 0)

        return filtered.astype(mask.dtype, copy=False), keep_labels


# def sh_mask_filter(mask, scale_vec, L_max=15, mesh_res=100, max_surf_dist=30, area_thresh=500):
#
#     # get mask locations
#     props = regionprops_table(mask, spacing=scale_vec, properties=("Centroid", "label"))
#     points_phys = np.asarray(props["centroid-0"], props["centroid-1"], props["centroid-2"])
#     labels = np.asarray(props["label"])
#
#     # fit sphere and get SH info
#     coeffs, fitted_center, fitted_radius = fit_sphere_and_sh(points_phys, L_max=L_max)
#     sphere_mesh = create_sphere_mesh(fitted_center, fitted_radius, resolution=mesh_res)
#
#     # get sh mesh
#     sh_mesh, r_sh = create_sh_mesh(coeffs, sphere_mesh)
#
#     # get distances
#     surf_dist_mat = distance_matrix(points, sh_mesh[0])
#     surf_dist_vec = np.min(surf_dist_mat, axis=1)
#
#     outlier_filter = (surf_dist_vec > max_surf_dist) & (area_vec < area_thresh)
#     outlier_labels = label_vec[outlier_filter]  # threshold for outliers
#     inlier_mask = label(~np.isin(mask, outlier_labels) & (mask > 0))  # create mask for outliers
#
#     return inlier_mask