import zarr
import napari
import numpy as np
import pandas as pd
import os
import dask.array as da
import ast

if __name__ == "__main__":

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    os.environ["QT_API"] = "pyqt5"
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
    os.environ["QT_API"] = "pyqt5"

    project_name = "20250311_LCP1-NLSMSC"
    mip_flag = False  # assume along z axis for now
    stitch_suffix = ""
    side = 0

    # load imaga dataset
    zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    mip_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_mip.zarr")
    fused_image_zarr = da.from_zarr(zpath)
    fz = zarr.open(zpath, mode="r")
    # get scale info
    if "voxel_size_um" not in fz.attrs.keys():
        ad = fz.attrs
        scale_vec = [ad["PhysicalSizeZ"], ad["PhysicalSizeY"], ad["PhysicalSizeX"]]
    else:
        scale_vec = fz.attrs["voxel_size_um"]

    # load full tracking dataset
    print("Loading tracking data for project:", project_name)
    nls_track_path = os.path.join(root, "tracking", project_name, "tracking_20250328_redux", "well0000", "track_0000_2339_cb", "")
    nls_track_zarr = zarr.open(os.path.join(nls_track_path, "segments.zarr"), mode="r")
    nls_tracks_df = pd.read_csv(os.path.join(nls_track_path, "tracks_fluo" + stitch_suffix + ".csv"))
    nucleus_class_df = pd.read_csv(os.path.join(nls_track_path, "track_class_df_full.csv"))
    # add class info to tracks
    nls_tracks_df = nls_tracks_df.merge(nucleus_class_df.loc[:, ["track_id", "t", "track_class", "frame_class"]],
                                        on=["track_id", "t"], how="left")

    lcp_track_path = os.path.join(root, "built_data", "tracking", project_name, "")
    # lcp_tracks_df = pd.read_csv(os.path.join(lcp_track_path, "lcp_tracks_df.csv"))
    lcp_curation_df = pd.read_csv(os.path.join(lcp_track_path, "20250311_lcp_track_curation.csv"))

    # get list of unique NLS tracks to link
    nls_nodes = []
    for i in range(len(lcp_curation_df)):
        nls_vec = np.asarray(ast.literal_eval(lcp_curation_df.loc[i, "lcp_lineage"]))
        nls_nodes.append(nls_vec[-1])
    nls_nodes = np.unique(nls_nodes)

    # get deep tracks
    deep_tracks_df = nls_tracks_df.loc[nls_tracks_df["track_class"] == 0, :].reset_index(drop=True)


    viewer = napari.Viewer()
    viewer.add_image(fused_image_zarr, channel_axis=1, scale=scale_vec, colormap=["cyan", "gray"], visible=False, contrast_limits=[(0, 500), (0, 2500)])


    viewer.add_tracks(
        nls_tracks_df[["track_id", "t", "z", "y", "x"]],
        name="nls tracks",
        scale=tuple(scale_vec),
        color_by="track_class",
        properties=nls_tracks_df[["track_class", "frame_class"]],
        translate=(0, 0, 0, 0),
        # features=nls_tracks_df[["fluo_mean", "nucleus_volume"]],
        visible=False,
        tail_width=3,
        tail_length=40
    )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    napari.run()

    print("check")

    # print("Extracting data frames...")
    # frame_range = np.arange(1820, 1840)
    # # get mask
    # mask_frame = np.squeeze(np.max(lcp_track_zarr[frame_range-1200], axis=1))
    # # get image
    # im_frame_lcp = np.squeeze(np.max(fused_image_zarr[frame_range, 0], axis=1))
    # im_frame_nls = np.squeeze(np.max(fused_image_zarr[frame_range, 1], axis=1))
    #
    # # fit sphere
    # from src.build_killi.build_utils import fit_sphere, create_sphere_mesh
    # from skimage.measure import regionprops
    #
    # print("Getting centroids...")
    # mf = np.asarray(lcp_track_zarr[2000-1200])
    # props = regionprops(mf, spacing=scale_vec)
    # centroids = np.asarray([props[i].centroid for i in range(len(props))])
    #
    # print("Fitting sphere...")
    # fitted_center, fitted_radius, inner_radius, outer_radius = fit_sphere(centroids)
    #
    # print("Generating mesh objects...")
    # thresh = 30
    # vertices_o, faces_o = create_sphere_mesh(fitted_center, fitted_radius+thresh, resolution=50)
    # vertices_i, faces_i = create_sphere_mesh(fitted_center, fitted_radius-thresh, resolution=50)
    #
    # print("Opening viewer...")
    # viewer = napari.Viewer(ndisplay=2)
    #
    # viewer.add_image(im_frame_nls, name="nls", contrast_limits=[0, 3000])
    # viewer.add_image(im_frame_lcp, name="lcp1", contrast_limits=[0, 500])
    #
    # viewer.add_surface((vertices_o, faces_o), name='inner', opacity=0.8, colormap='viridis')
    # viewer.add_surface((vertices_i, faces_i), name='outer', opacity=0.8, colormap='viridis')
    #
    # napari.run()
    #
    # print("Check")
    #
    # import numpy as np
    # import scipy.ndimage as ndi
    #
    # # Define sphere center C and parameters R and D.
    # C = fitted_center  # replace with your center coordinates
    # C_x, C_y, C_z = C[::-1]
    # R = fitted_radius  # sphere radius
    # D = 50  # allowed distance offset
    #
    # # Define your 3D image A (for example, loaded from a Zarr array)
    # # A.shape should be (Z, Y, X)
    # nls_test = np.squeeze(fused_image_zarr[2000, 1])
    # A = nls_test.copy()  # your 3D data
    #
    # Z, Y, X = np.indices(A.shape)
    # Z = Z*3
    #
    # r = np.sqrt((X - C_x) ** 2 + (Y - C_y) ** 2 + (Z - C_z) ** 2)
    # theta = np.arctan2(Y - C_y, X - C_x)
    # phi = np.arccos((Z - C_z) / (r + 1e-10))
    #
    # shell_mask = (r >= R - D) & (r <= R + D)
    #
    # # For each (phi, theta), sample along the radial direction.
    # nbins_theta = 512  # number of bins in theta
    # nbins_phi = 512  # number of bins in phi
    # theta_bins = np.linspace(-np.pi, np.pi, nbins_theta + 1)
    # phi_bins = np.linspace(0, np.pi, nbins_phi + 1)
    #
    # theta_idx = np.digitize(theta[shell_mask], theta_bins) - 1
    # phi_idx = np.digitize(phi[shell_mask], phi_bins) - 1
    # bin_idx = phi_idx * nbins_theta + theta_idx
    #
    # flat_intensity = A[shell_mask].compute().ravel()
    # # Create an array of zeros (or -inf) for bins:
    # # projection = np.full((nbins_phi * nbins_theta,), -np.inf)
    # # For each unique bin, compute the maximum value.
    # # for idx in tqdm(np.unique(bin_idx)):
    # #     projection[idx] = np.max(flat_intensity[bin_idx == idx])
    # # # Reshape the projection into (nbins_phi, nbins_theta)
    # # projection = projection.reshape((nbins_phi, nbins_theta))
    #
    # from scipy import ndimage as ndi
    # proj = ndi.maximum(flat_intensity, bin_idx, index=np.arange(nbins_phi*nbins_theta))
    #
    # proj = proj.reshape((nbins_phi, nbins_theta))