import pandas as pd
import numpy as np
import napari
import os
from src.build_yx1.fit_embryo_surface import make_sphere_mesh, fit_sphere_with_percentile
from src.nucleus_dynamics.utilities.image_utils import calculate_LoG
from pathlib import Path
import zarr
import numpy as np

if __name__ == "__main__":

    project_name = "20250731"
    model_name = "tdTom-bright-log-v5"
    zarr_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/{project_name}/")
    mask_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/mask_stacks/{model_name}/{project_name}/")
    prob_path = Path(
        f"/media/nick/cluster/projects/data/killi_tracker/built_data/cellpose_output/{model_name}/{project_name}/")
    # image_list = sorted(list(zarr_path.glob("*.zarr")))
    # image_list = [im for im in image_list if "_z.zarr" not in str(im)]
    t_ind = 55
    well_ind = 4
    ch_ind = 0
    R_um = 550
    n_phi = 360
    n_theta = 180
    out_thresh = 75

    im_zarr = zarr.open(zarr_path / f"{project_name}_well{well_ind:04}.zarr", mode='r')
    mask_zarr = zarr.open(mask_path  / f"{project_name}_well{well_ind:04}_mask_aff.zarr", mode='r')
    prob_zarr = zarr.open(prob_path / f"{project_name}_well{well_ind:04}_probs.zarr", mode='r')
    im_plot = np.squeeze(im_zarr[ch_ind, t_ind])
    prob_plot = np.squeeze(prob_zarr[t_ind])
    mask_plot = np.squeeze(mask_zarr[t_ind])
    scale_vec = np.array(im_zarr.attrs['voxel_size_um'])

    # load sphere fit if it exists
    # dog = remove_background_dog(vol=im_plot,
    #                             scale_vec=scale_vec,
    #                             sigma_large_um=sigma_small_um,
    #                             sigma_small_um=sigma_large_um)
    # thresh = np.percentile(dog, seg_thresh)
    # mask = dog > thresh
    points_phys = np.array(np.nonzero(mask_plot)).T * scale_vec[None, :]
    #
    center_fit_raw, _, r_inner_raw = fit_sphere_with_percentile(
                                                                points_phys,
                                                                im_shape=np.multiply(scale_vec, mask_plot.shape),
                                                                pct=.25,
                                                                loss="linear",
                                                                R0=None)

    # remove outliers and refit
    dists = np.sqrt(np.sum((points_phys - center_fit_raw[None, :])**2, axis=1))
    inlier_mask = np.abs(dists - r_inner_raw) < out_thresh

    center_fit, radius_fit, radius_inner = fit_sphere_with_percentile(
                                                                points_phys[inlier_mask],
                                                                im_shape=np.multiply(scale_vec, mask_plot.shape),
                                                                pct=.25,
                                                                loss="linear",
                                                                R0=None)

    inner_mask = np.zeros_like(im_plot)
    zg, yg, xg = np.meshgrid(np.arange(im_plot.shape[0]), np.arange(im_plot.shape[1]), np.arange(im_plot.shape[2]), indexing='ij')
    dist_array = np.sqrt((zg*scale_vec[0]-center_fit[0])**2 +
                         (yg*scale_vec[1]-center_fit[1])**2 +
                         (xg*scale_vec[2]-center_fit[2])**2)
    inner_mask[dist_array <= radius_inner] = 1
    # # sphere_file = sphere_path / f"well{well_ind:04}_sphere_fits.csv"
    # # sphere_df = pd.read_csv(sphere_file)
    # # sphere_row = sphere_df[sphere_df["t"] == t_ind]
    # # center = sphere_row[["center_z_smooth", "center_y_smooth", "center_x_smooth"]].to_numpy()[0]
    # # radius = sphere_row["radius_smooth"].to_numpy()[0]
    # print(radius)
    verts, faces = make_sphere_mesh(n_phi, n_theta, center_fit, radius_inner)
    # verts0, faces0 = make_sphere_mesh(n_phi, n_theta, center, radius-75)
    #
    # im_log, im_bkg = calculate_LoG(data_zyx=im_plot, scale_vec=scale_vec, subtract_background=False,
    #                                make_isotropic=False)
    print(center_fit)
    viewer = napari.Viewer()
    viewer.add_image(im_plot, scale=scale_vec)
    viewer.add_image(prob_plot, scale=scale_vec)
    viewer.add_labels(inner_mask, name='inner_mask', scale=scale_vec)
    viewer.add_labels(mask_plot, scale=scale_vec)

    viewer.add_surface((verts, faces),
                       colormap="magma", opacity=1, name="spherical_projection")
    # viewer.add_surface((verts0, faces0),
    #                    colormap="magma", opacity=1, name="spherical_projection")

    napari.run()
    # viewer.add_points(points, size=15, face_color="Green", name="sphere centers")
    # viewer.add_surface(sphere_mesh)
    print("Saving sphere data...")
