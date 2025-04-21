import numpy as np
from scipy import ndimage as ndi
import os
import zarr
import pandas as pd
from src.build_killi.build_utils import fit_sphere, create_sphere_mesh
from skimage.measure import regionprops
import napari

if __name__ == "__main__":

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"

    zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    mip_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_mip.zarr")
    fused_image_zarr = zarr.open(zpath, mode="r")

    tracking_config = "tracking_20250328_redux"
    nls_track_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000", "track_0000_2339_cb", "")
    nls_track_zarr = zarr.open(os.path.join(nls_track_path, "segments.zarr"), mode="r")
    nls_tracks_df = pd.read_csv(os.path.join(nls_track_path, "tracks_fluo" + ".csv"))

    # get scale info
    ad = fused_image_zarr.attrs
    scale_vec = [ad["PhysicalSizeZ"], ad["PhysicalSizeY"], ad["PhysicalSizeX"]]

    print("Extracting data frames...")
    target_frame = 1400

    print("Getting centroids...")
    mf = np.asarray(nls_track_zarr[target_frame])
    props = regionprops(mf, spacing=scale_vec)
    centroids = np.asarray([props[i].centroid for i in range(len(props))])

    print("Fitting sphere...")
    fitted_center, fitted_radius, inner_radius, outer_radius = fit_sphere(centroids)

    im_frame_nls = np.squeeze(fused_image_zarr[target_frame, 1])
    # im_frame_lcp = np.squeeze(fused_image_zarr[target_frame, 0])
    print("Generating mesh objects...")
    thresh = 30
    vertices_o, faces_o = create_sphere_mesh(fitted_center, fitted_radius+thresh, resolution=50)
    vertices_i, faces_i = create_sphere_mesh(fitted_center, fitted_radius*0.96, resolution=50)

    print("Opening viewer...")
    # viewer = napari.Viewer(ndisplay=3)

    # viewer.add_surface((vertices_i, faces_i), name='outer', opacity=0.8, colormap='viridis')

    # viewer.add_image(im_frame_nls, name="nls", scale=scale_vec, contrast_limits=[0, 2000])
    # viewer.add_image(im_frame_lcp, name="lcp1", contrast_limits=[0, 500])



    # napari.run()

    print("Check")

    # Define sphere center C and parameters R and D.
    C = fitted_center  # replace with your center coordinates
    C_x, C_y, C_z = C[::-1]
    R = fitted_radius  # sphere radius
    D = 50  # allowed distance offset

    # Define your 3D image A (for example, loaded from a Zarr array)
    # A.shape should be (Z, Y, X)
    nls_test = np.squeeze(fused_image_zarr[2000, 1])
    A = nls_test.copy()  # your 3D data

    Z, Y, X = np.indices(A.shape)
    Z = Z*3

    r = np.sqrt((X - C_x) ** 2 + (Y - C_y) ** 2 + (Z - C_z) ** 2)
    theta = np.arctan2(Y - C_y, X - C_x)
    phi = np.arccos((Z - C_z) / (r + 1e-10))

    shell_mask = (r >= R - D) & (r <= R + D)

    # For each (phi, theta), sample along the radial direction.
    nbins_theta = 512  # number of bins in theta
    nbins_phi = 512  # number of bins in phi
    theta_bins = np.linspace(-np.pi, np.pi, nbins_theta + 1)
    phi_bins = np.linspace(0, np.pi, nbins_phi + 1)

    theta_idx = np.digitize(theta[shell_mask], theta_bins) - 1
    phi_idx = np.digitize(phi[shell_mask], phi_bins) - 1
    bin_idx = phi_idx * nbins_theta + theta_idx

    flat_intensity = A[shell_mask].ravel()


    proj = ndi.maximum(flat_intensity, bin_idx, index=np.arange(nbins_phi*nbins_theta))

    proj = proj.reshape((nbins_phi, nbins_theta))

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # — your existing proj array of shape (nbins_phi, nbins_theta) —
    # proj[i,j] corresponds to φ in bin i and θ in bin j, where
    #   θ ∈ [–π … +π] (longitude),   φ ∈ [0 … π] (co‐latitude)

    # 1) build the lon/lat edges for "extent"
    theta_edges = np.linspace(-np.pi, np.pi, nbins_theta + 1)
    phi_edges = np.linspace(0, np.pi, nbins_phi + 1)

    # convert to degrees & to (lon_min, lon_max, lat_min, lat_max)
    lon_min = np.degrees(theta_edges[0])
    lon_max = np.degrees(theta_edges[-1])
    # latitude = 90° – φ*180/π
    lat_edges = 90 - np.degrees(phi_edges)
    lat_min, lat_max = lat_edges.min(), lat_edges.max()  # -> (–90, +90)

    extent = (lon_min, lon_max, lat_min, lat_max)  # (–180,180,–90,90)

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

    # ─────────────────────────────────────────────────────────────────────
    # 1) Set up a black‐background “theme” for this figure
    mpl.rcParams.update({
        # figure + axes face
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",
        # text & ticks in white
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    })

    # ─────────────────────────────────────────────────────────────────────
    # 2) build the figure & axis (explicitly give facecolor again for safety)
    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw=dict(projection=ccrs.Robinson()),
        facecolor="black",  # figure patch
    )
    ax.set_facecolor("black")  # axes patch

    # ─────────────────────────────────────────────────────────────────────
    # 3) display the equirectangular image as gray → white on black
    img = proj  # your 2D array of shape (nbins_phi, nbins_theta)
    ax.imshow(
        img,
        origin="lower",
        extent=extent,  # (lon_min, lon_max, lat_min, lat_max)
        transform=ccrs.PlateCarree(),
        cmap="gray",  # black→white
        interpolation="nearest",
        vmin=0,
        vmax=1000
    )

    # ─────────────────────────────────────────────────────────────────────
    # 4) add gridlines & (optional) coastlines in white
    gl = ax.gridlines(
        draw_labels=True,
        color="white",
        linestyle="--",
        linewidth=0.5,
        xformatter=LongitudeFormatter(),
        yformatter=LatitudeFormatter()
    )
    gl.top_labels = False
    gl.right_labels = False

    # if you want the coastlines:
    # ax.coastlines(color="white", linewidth=0.8)

    ax.set_global()  # force full‐globe view

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # 5) save to disk with black background baked in
    fig_path = os.path.join(root, "figures", project_name, "pipeline_figs", "deep_cell_density_robinson_black.png")
    # out_path =
    fig.savefig(
        fig_path,
        dpi=600,  # high resolution
        facecolor="black",  # ensure the saved file uses black
        edgecolor="black",
        bbox_inches="tight",
        pad_inches=0
    )
    print("Check")