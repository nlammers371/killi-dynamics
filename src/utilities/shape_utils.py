import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
import os
import glob2 as glob
from skimage.measure import regionprops
from scipy.stats import multivariate_normal
import math
from sklearn.neighbors import KDTree
from scipy.ndimage import distance_transform_edt
from numpy import pad
import plotly.graph_objects as go
import trimesh
# import alphashape

# remeshing
import pyvista
import pyacvd
import pymeshfix

#
#
# def pyvista_to_mesh(mesh):
#     v = mesh.points
#     f = mesh.faces.reshape(-1, 4)[:, 1:4]
#     return v, f
#
#
# def mesh_to_pyvista(v, f):
#     n, m = f.shape
#     threes = np.full((n, 1), 3)
#     face_arr = np.hstack((threes, f)).flatten()
#     return pyvista.PolyData(v, face_arr)
#
#
# def mesh_cleanup(mesh_raw, target_verts=2500):
#     v, f = mesh_raw.vertices, mesh_raw.faces
#     # this can give a depreciation warning but it is fine
#     mesh = mesh_to_pyvista(v, f)
#
#     # target mesh resolution
#     # target_verts = 2500
#
#     clus = pyacvd.Clustering(mesh)
#     clus.subdivide(2)
#     clus.cluster(target_verts)
#
#     remesh = clus.create_mesh()
#
#     v2, f2 = pyvista_to_mesh(remesh)
#
#     # pymeshfix is often necessary here to get rid of non-manifold vertices
#     v2, f2 = pymeshfix.clean_from_arrays(v2, f2)
#
#     mesh_out = trimesh.Trimesh(vertices=v2, faces=f2)
#
#     return mesh_out
#
#
# def ellipsoid_axis_lengths(central_moments):
#     """Compute ellipsoid major, intermediate and minor axis length.
#
#     Parameters
#     ----------
#     central_moments : ndarray
#         Array of central moments as given by ``moments_central`` with order 2.
#
#     Returns
#     -------
#     axis_lengths: tuple of float
#         The ellipsoid axis lengths in descending order.
#     """
#     m0 = central_moments[0, 0, 0]
#     sxx = central_moments[2, 0, 0] / m0
#     syy = central_moments[0, 2, 0] / m0
#     szz = central_moments[0, 0, 2] / m0
#     sxy = central_moments[1, 1, 0] / m0
#     sxz = central_moments[1, 0, 1] / m0
#     syz = central_moments[0, 1, 1] / m0
#     S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
#     # determine eigenvalues in descending order
#     eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
#     radii = tuple([math.sqrt(5 * e) for e in eigvals])
#     return radii, S
#
#
# def process_fin_df(fin_object, fin_df=None, k_nn=20, d_lb=2, d_ub=15):
#     if fin_df is None:
#         fin_df = fin_object.full_point_data
#         fin_df = fin_df.loc[fin_df["fin_label_curr"] == 1, :]
#     fin_points = fin_df[["X", "Y", "Z"]].to_numpy()
#
#     # orient to biological axes
#     fin_axis_df = fin_object.axis_fin
#     fin_axes = fin_object.calculate_axis_array(fin_axis_df)
#     fin_points_pca = np.matmul(fin_points - np.mean(fin_points, axis=0), fin_axes.T)
#     fin_df.loc[:, ["XP", "YP", "ZP"]] = fin_points_pca
#
#     ###########################
#     # Get nearest-neighbor statistics
#     # quick pass to remove extreme outliers
#
#     tree = KDTree(fin_df[["X", "Y", "Z"]].to_numpy())
#     xyz_array = fin_df[["X", "Y", "Z"]].to_numpy()
#     if xyz_array.shape[0] > k_nn:
#         dist, ind = tree.query(xyz_array, k=k_nn + 1)
#
#         # get avg neighbor distance
#         nn_scale_vec = np.mean(dist[:, 1:3], axis=1)
#
#         # average NN dist to get spatially smoothed estimate for inter-nucleus spacing
#         nn_dist_array = nn_scale_vec[ind]
#         nn_dist_mean = np.percentile(nn_dist_array, 90, axis=1)
#         nn_dist_mean[nn_dist_mean < d_lb] = d_lb
#         nn_dist_mean[nn_dist_mean > d_ub] = d_ub
#
#         fin_df["nn_scale_um"] = nn_dist_mean
#     else:
#         fin_df["nn_scale_um"] = np.inf  # this will be ignored in favor of nucleus-based dims
#
#     return fin_df
#
#
# def get_fin_mask(point_name, fin_df, seg_model, root):
#     # load nuclear mask
#     well_ind = point_name.find("well")
#     date = point_name[:well_ind - 1]  # "20240711_01"
#     well_num = int(point_name[well_ind + 4:well_ind + 8])
#     path_string = os.path.join(root, "built_data", "mask_stacks", seg_model, date,
#                                date + f"_well{well_num:04}" + "*aff.zarr*")
#     mask_path = glob.glob(path_string)[0]
#
#     # load mask and filter for nuclei that are inside the fin
#     time_int = int(point_name[-4:])
#     mask_raw = zarr.open(mask_path, mode="r")
#     mask = np.squeeze(mask_raw[time_int])
#     mask[~np.isin(mask, fin_df.nucleus_id.to_numpy())] = 0
#     scale_vec = mask_raw.attrs["voxel_size_um"]
#
#     return mask, scale_vec
#
#
# def get_gaussian_masks(fin_df, mask, sample_res_um, z_factor, scale_vec, sample_sigma=1):
#     fin_points = fin_df.loc[:, ["X", "Y", "Z"]].to_numpy()
#
#     # get grid dimensions
#     z_min, y_min, x_min = np.floor(
#         [fin_points[:, 2].min(), fin_points[:, 1].min(), fin_points[:, 0].min()]) - 5
#     z_max, y_max, x_max = np.ceil(
#         [fin_points[:, 2].max(), fin_points[:, 1].max(), fin_points[:, 0].max()]) + 5
#
#     x_vec = np.arange(x_min, x_max, sample_res_um)
#     y_vec = np.arange(y_min, y_max, sample_res_um)
#     z_vec = np.arange(z_min, z_max, sample_res_um)
#
#     # make ref grids
#     z_grid, y_grid, x_grid = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")
#     # xyz_array = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
#
#     # use regionprops to get mask statistics
#     scale_vec_rs = scale_vec.copy()
#     scale_vec_rs[0] = scale_vec_rs[0] / z_factor
#
#     # set thresh value
#     thresh = np.exp(-sample_sigma ** 2 / 2)
#
#     # get mask info
#     regions = regionprops(mask, spacing=scale_vec_rs)
#
#     # initialize arrays to store mask IDs and Gaussian weights
#     nucleus_id_array = np.zeros(x_grid.shape, dtype=np.uint16)
#     nucleus_weight_array = np.zeros(x_grid.shape, dtype=np.float32)
#
#     # lb_vec = [rg.label for rg in regions]
#     buffer = 5 / sample_res_um
#
#     for rg in tqdm(regions):
#
#         # iterate through masks
#         mask_id = rg.label
#         # if mask_id == 6453:
#         #     print("check")
#         nn_scale = fin_df.loc[fin_df["nucleus_id"] == mask_id, "nn_scale_um"].to_numpy()[0]
#
#         # calculate ellipsoid dims
#         cm = rg["moments_central"]
#         MU = np.asarray(rg["Centroid"])
#         MU[0] = MU[0] * z_factor  # make sure to get correctly scaled z centroid locations
#         r_vals, CORR = ellipsoid_axis_lengths(cm)
#
#         # check for singular matrix
#         cc = rg.coords_scaled
#         if np.linalg.det(CORR) == 0:
#             v_vec = np.var(cc, axis=0)
#             v_vec[v_vec < 1] = 1
#             CORR[np.eye(3) == 1] = v_vec
#             # print("fix")
#
#         factor = np.min([np.max([nn_scale / r_vals[0], 1]), 1.5])
#         scale_factors = nn_scale / (factor * np.asarray(np.asarray(r_vals[::-1])))
#         scale_factors[scale_factors > 1] = 1
#         # scale_factors = np.asarray([factor, factor, factor])
#         COV0 = 5 * CORR * factor ** 2
#         # rescale
#         # Perform eigendecomposition
#         eigenvalues, eigenvectors = np.linalg.eigh(COV0)
#         # Create diagonal matrix of eigenvalues
#         Lambda = np.diag(eigenvalues)
#         Lambda_rs = np.multiply(Lambda, scale_factors)
#         # Recreate the covariance matrix
#         COV = eigenvectors @ Lambda_rs @ eigenvectors.T
#
#         # see here for why there is a factor of 5 https://forum.image.sc/t/scikit-image-regionprops-minor-axis-length-in-3d-gives-first-minor-radius-regardless-of-whether-it-is-actually-the-shortest/59273/2
#         # COV = CORR#np.matmul(CORR, CORR.T)
#
#         # Extract the region from the original 3D image using the bounding box
#
#         cc[:, 0] = cc[:, 0] * z_factor
#         min_z, min_y, min_x = np.asarray(np.min(cc, axis=0) - buffer)  # + np.asarray([z0, y0, x0])
#         max_z, max_y, max_x = np.asarray(np.max(cc, axis=0) + buffer)  # + np.asarray([z0, y0, x0])
#
#         min_row, min_col, min_depth = np.abs(z_vec - min_z).argmin(), np.abs(y_vec - min_y).argmin(), np.abs(
#             x_vec - min_x).argmin()
#         max_row, max_col, max_depth = np.abs(z_vec - max_z).argmin(), np.abs(y_vec - max_y).argmin(), np.abs(
#             x_vec - max_x).argmin()
#
#         zv = z_grid[min_row:max_row, min_col:max_col, min_depth:max_depth]
#         yv = y_grid[min_row:max_row, min_col:max_col, min_depth:max_depth]
#         xv = x_grid[min_row:max_row, min_col:max_col, min_depth:max_depth]
#
#         zyx = np.c_[zv.ravel(), yv.ravel(), xv.ravel()]
#
#         # calculate probabilities (modeling as multivariate gaussian)
#         try:
#             p = multivariate_normal.pdf(zyx, mean=MU, cov=COV)
#             # p = multivariate_normal.pdf(xyz_array, mean=MU, cov=COV[::-1, ::-1])
#
#             p = p / np.max(p)
#             P = np.reshape(p, zv.shape)  # x_grid.shape)
#         except:
#             print("why?")
#
#         # take high prob pixels
#         wv = nucleus_weight_array[min_row:max_row, min_col:max_col, min_depth:max_depth]
#         iv = nucleus_id_array[min_row:max_row, min_col:max_col, min_depth:max_depth]
#
#         sig_indices = np.where(P >= thresh)
#         p_curr = wv[sig_indices]
#         p_new = P[sig_indices]
#
#         # only update regions for which current mask is closest option
#         update_indices = ([s[p_new > p_curr] for s in sig_indices])
#
#         # update as appropriate
#         wv[update_indices[0], update_indices[1], update_indices[2]] = p_new[p_new > p_curr]
#         iv[update_indices[0], update_indices[1], update_indices[2]] = mask_id
#
#         nucleus_weight_array[min_row:max_row, min_col:max_col, min_depth:max_depth] = wv
#         nucleus_id_array[min_row:max_row, min_col:max_col, min_depth:max_depth] = iv
#
#     return nucleus_id_array, nucleus_weight_array
#
#
# def upsample_nucleus_points(nucleus_id_array, sample_res_um, points_per_nucleus):
#     regions = regionprops(nucleus_id_array, spacing=np.asarray([sample_res_um, sample_res_um, sample_res_um]))
#     df_vec = []
#
#     for rg in tqdm(regions):
#         nucleus_id = rg.label
#         im0 = rg.image.copy()
#         imp = pad(im0, pad_width=1)
#         imd = distance_transform_edt(imp, sampling=sample_res_um)
#         imd = imd[1:-1, 1:-1, 1:-1]
#         fg_indices = np.where(im0.ravel() == True)[0]
#
#         boundary_indices = np.where((imd.ravel() > 0) & (imd.ravel() <= 1))[0]
#         bound_options = np.where(np.isin(fg_indices, boundary_indices))[0]
#         bound_samples = np.random.choice(bound_options, points_per_nucleus)
#
#         cs = rg.coords_scaled
#         boundary_xyz = cs[bound_samples, ::]
#         # add rs for numerical reasons
#         jitter = (np.random.rand(boundary_xyz.shape[0], boundary_xyz.shape[1]) - 0.5) * sample_res_um
#         boundary_xyz += jitter
#
#         df_temp = pd.DataFrame([nucleus_id] * points_per_nucleus, columns=["nucleus_id"])
#         df_temp[["X", "Y", "Z"]] = boundary_xyz
#         df_vec.append(df_temp)
#
#     # combine
#     fin_df_new = pd.concat(df_vec, axis=0, ignore_index=True)
#     fin_df_new["nucleus_id"] = fin_df_new["nucleus_id"].astype(str)
#
#     return fin_df_new
#
#
# def upsample_fin_point_cloud(fin_object, fin_df=None, root=None, points_per_nucleus=25, z_factor=1.95,
#                              sample_res_um=0.25,
#                              seg_model="tdTom-bright-log-v5"):
#     # get dataset name
#     point_name = fin_object.name
#
#     if root is None:
#         root = fin_object.data_root
#
#     # extract data frame containing nucleus centroids
#     fin_df = process_fin_df(fin_object, fin_df=fin_df)
#
#     mask, scale_vec = get_fin_mask(point_name, fin_df, seg_model, root)
#
#     ###########################
#     # Gaussian up-sampling procedure
#     nucleus_id_array, nucleus_weight_array = get_gaussian_masks(fin_df, mask, sample_res_um, z_factor, scale_vec)
#
#     ###########################3
#     # Sample boundary points
#     fin_df_new = upsample_nucleus_points(nucleus_id_array, sample_res_um, points_per_nucleus)
#
#     # orient to biological axes
#     fin_points = fin_df_new[["Z", "Y", "X"]].to_numpy()
#     fin_axis_df = fin_object.axis_fin
#     fin_axes = fin_object.calculate_axis_array(fin_axis_df)
#     fin_points_pca = np.matmul(fin_points - np.mean(fin_points, axis=0), fin_axes.T)
#     fin_df_new.loc[:, ["XP", "YP", "ZP"]] = fin_points_pca
#
#     return fin_df_new, nucleus_id_array, nucleus_weight_array
#

def plot_mesh(plot_hull, surf_alpha=0.2):
    tri_points = plot_hull.vertices[plot_hull.faces]

    # extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])

    # define the trace for triangle sides
    fig = go.Figure()
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color='rgb(70,70,70, 0.5)', width=1))

    lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness=0.9, specular=0.9, fresnel=0.9)
    mesh = go.Mesh3d(x=plot_hull.vertices[:, 0], y=plot_hull.vertices[:, 1], z=plot_hull.vertices[:, 2],
                     opacity=surf_alpha, i=plot_hull.faces[:, 0], j=plot_hull.faces[:, 1], k=plot_hull.faces[:, 2],
                     lighting=lighting_effects)
    fig.add_trace(mesh)

    fig.add_trace(lines)
    fig.update_layout(template="plotly")

    return fig, lines, mesh

#
# def fit_fin_mesh(xyz_fin, alpha=20, n_faces=5000, smoothing_strength=5):
#     # normalize for alphshape fitting
#     mp = np.min(xyz_fin)
#     points = xyz_fin - mp
#     mmp = np.max(points)
#     points = points / mmp
#
#     meshing_error_flag = False
#     try:
#         raw_hull = alphashape.alphashape(points, alpha)
#     except:
#         meshing_error_flag = True
#
#     if not meshing_error_flag:
#         # copy
#         hull02_cc = raw_hull.copy()
#
#         # keep only largest component
#         hull02_cc = hull02_cc.split(only_watertight=False)
#         hull02_sm = max(hull02_cc, key=lambda m: m.area)
#
#         hull02_sm = trimesh.smoothing.filter_laplacian(hull02_sm, iterations=2)
#
#         # fill holes
#         hull02_sm = mesh_cleanup(hull02_sm)
#
#         # smooth
#         hull02_sm = trimesh.smoothing.filter_laplacian(hull02_sm, iterations=smoothing_strength)
#
#         # resample
#         n_faces = np.min([n_faces, hull02_sm.faces.shape[0] - 1])
#         hull02_rs = hull02_sm.simplify_quadric_decimation(face_count=n_faces)
#         hull02_rs = hull02_rs.split(only_watertight=False)
#         hull02_rs = max(hull02_rs, key=lambda m: m.area)
#         hull02_rs.fill_holes()
#         hull02_rs.fix_normals()
#
#         vt = hull02_rs.vertices
#         vt = vt * mmp
#         vt = vt + mp
#         hull02_rs.vertices = vt
#
#         # check
#         wt_flag = hull02_rs.is_watertight
#
#         return hull02_rs, raw_hull, wt_flag
#
#     else:
#         return None, None, False