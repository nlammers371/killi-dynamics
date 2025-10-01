import napari
import pandas as pd
import numpy as np
from pathlib import Path
import zarr
from src.killi_stats.surface_stats import make_sphere_mesh, smooth_spherical_grid, project_to_sphere, healpix_to_mesh, \
    project_to_healpix
import healpy as hp

if __name__ == "__main__":

    project_name = "20250621"
    well_ind = 3
    t_ind = 47
    ch_ind = 0
    R_um = 550
    n_phi = 360
    n_theta = 180
    proj_mode= "mean"
    data_root = Path(f"/media/nick/cluster/projects/data/killi_tracker/output_data/sphere_projections/{project_name}/")
    sphere_path = data_root / f"{well_ind:04}_sphere_fits.csv"
    dist_thresh = 75.0
    smooth_sigma_theta = 2.0
    smooth_sigma_phi = 2.0
    # field_path = data_root / f"{well_ind:04}_fields.zarr"
    #
    sphere_df = pd.read_csv(sphere_path)
    # field_zarr = zarr.open(field_path, mode="r")
    # raw_field = field_zarr["raw"]
    # sm_field = field_zarr["smoothed"]

    zarr_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/{project_name}/")
    image_list = sorted(list(zarr_path.glob("*.zarr")))
    image_list = [im for im in image_list if "_z.zarr" not in str(im)]

    im_zarr = zarr.open(image_list[well_ind], mode='r')
    im_plot = np.squeeze(im_zarr[ch_ind, t_ind])
    scale_vec = np.array(im_zarr.attrs['voxel_size_um'])

    # center_fitted = sphere_df[(sphere_df['t'] == t_ind) ][['center_z_smooth', 'center_y_smooth', 'center_x_smooth']].values[0]
    # radius = sphere_df[sphere_df['t'] == t_ind]['radius_smooth'].values[0]

    row = sphere_df.loc[sphere_df.t == t_ind].iloc[0]
    center = np.array([row.center_z_smooth, row.center_y_smooth, row.center_x_smooth])
    radius = row.radius_smooth

    raw_arr = np.zeros((n_theta, n_phi))


    im = np.squeeze(im_zarr[ch_ind, t_ind])
    values, _ = project_to_healpix(
        im,
        center=center,
        radius=radius,
        scale_vec=scale_vec,
        nside=64,
        mode=proj_mode,
        dist_thresh=dist_thresh,
    )
    smoothed = hp.smoothing(values, fwhm=np.radians(5))
    # values_grid = np.reshape(values, (n_theta, n_phi))
    # values_sm = smooth_spherical_grid(
    #     values_grid, counts=None,
    #     sigma_theta=smooth_sigma_theta,
    #     sigma_phi=smooth_sigma_phi,
    # )

    verts, faces, values = healpix_to_mesh(smoothed, radius, center)
    # verts0, faces0 = make_sphere_mesh(n_theta=n_theta, n_phi=n_phi, radius=radius - dist_thresh, center=center)
    # verts1, faces1 = make_sphere_mesh(n_theta=n_theta, n_phi=n_phi, radius=radius + dist_thresh, center=center)

    viewer = napari.Viewer()
    viewer.add_image(im_plot, scale=scale_vec)
    viewer.add_surface((verts, faces, values.ravel()), colormap="magma", opacity=0.7)
    napari.run()
    print("Check")


