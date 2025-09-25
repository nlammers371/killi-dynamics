import pandas as pd
import numpy as np
import napari
import os
from src.killi_stats.surface_stats import fit_sphere, remove_background_dog, project_to_sphere, smooth_spherical_grid

from pathlib import Path
import zarr
import numpy as np

if __name__ == "__main__":

    project_name = "20250716"

    zarr_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/{project_name}/")
    image_list = sorted(list(zarr_path.glob("*.zarr")))
    image_list = [im for im in image_list if "_z.zarr" not in str(im)]
    t_ind = -1
    well_ind = 7
    ch_ind = 1
    R_um = 550
    n_phi = 360
    n_theta = 180

    im_zarr = zarr.open(image_list[well_ind], mode='r')
    im_plot = np.squeeze(im_zarr[ch_ind, t_ind])
    scale_vec = np.array(im_zarr.attrs['voxel_size_um'])
    dog = remove_background_dog(im_plot, scale_vec=scale_vec, sigma_small_um=2.0, sigma_large_um=8.0)

    thresh = np.percentile(dog, 99) #threshold_multiotsu(dog, classes=3)
    mask = dog > thresh #[1]
    points_phys = np.array(np.nonzero(mask)).T * scale_vec[None, :]
    fitted_center, fitted_radius = fit_sphere(points_phys, R0=R_um)

    verts, faces, values =  project_to_sphere(im_plot, center=fitted_center, radius=R_um, scale_vec=scale_vec,
                                      n_theta=n_theta, n_phi=n_phi,
                                      mode="mean", dist_thresh=50.0)

    values_sm = smooth_spherical_grid(np.reshape(values, (n_theta, n_phi)), counts=None, sigma_theta=2.0, sigma_phi=2.0)

    viewer = napari.Viewer()
    viewer.add_image(im_plot, scale=scale_vec)
    viewer.add_image(dog, scale=scale_vec)
    viewer.add_labels(mask, scale=scale_vec)

    viewer.add_surface((verts, faces, values_sm.ravel()),
                       colormap="magma", opacity=0.7, name="spherical_projection")

    napari.run()
    # viewer.add_points(points, size=15, face_color="Green", name="sphere centers")
    # viewer.add_surface(sphere_mesh)
    print("Saving sphere data...")
