import zarr
import napari
import numpy as np
from pathlib import Path
import os
import pandas as pd
from skimage.measure import label, regionprops_table
from tqdm import tqdm

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = Path(r"/media/nick/cluster/projects/data/killi_dynamics/")
project = "20250716"
well_num_vec = [4, 11]
dist_thresh = 50
t_start = 20
t_stop = 25
nucleus_channel = 1
out_dir = root / 'shared_data' / "sample_yx1_masks" / project
out_dir.mkdir(parents=True, exist_ok=True)

for well_num in tqdm(well_num_vec, desc="Processing wells"):

    im_path = root / "built_data" / "zarr_image_files" / project / (project + f"_well{well_num:04}.zarr")
    mask_path = root / "built_data" / "mask_stacks" / "tdTom-bright-log-v5" / project / (project + f"_well{well_num:04}_mask_aff.zarr")
    sphere_path = root / "output_data" / "sphere_projections" / project / (f"well{well_num:04}_sphere_fits.csv")

    image_zarr = zarr.open(im_path, mode="r")
    mask_zarr = zarr.open(mask_path, mode="r")
    sphere_df = pd.read_csv(sphere_path)

    # generate frame indices
    frames = np.arange(t_start, t_stop)

    # get scale info
    scale_vec = tuple(image_zarr.attrs["voxel_size_um"])

    # extract relevant frames
    im_p = np.squeeze(image_zarr[nucleus_channel, t_start:t_stop])
    mask_p = mask_zarr[t_start:t_stop]
    mask_clean = np.zeros_like(mask_p)
    for i, t in enumerate(frames):
        row = sphere_df.loc[sphere_df.t == t].iloc[0]
        # use smoothed values if available
        if "center_z_smooth" in row:
            center = np.array([row.center_z_smooth, row.center_y_smooth, row.center_x_smooth])
            radius = row.radius_smooth
        else:
            center = np.array([row.center_z, row.center_y, row.center_x])
            radius = row.radius
        # get cell centroids in physical unit
        props = regionprops_table(mask_p[i], spacing=scale_vec, properties=("centroid", "label"))
        coords = np.column_stack([props["centroid-0"], props["centroid-1"], props["centroid-2"]])
        labels = props["label"]
        # restrict to spherical shell
        dR = np.linalg.norm(coords - center[None, :], axis=1) - radius
        mask = np.abs(dR) <= dist_thresh
        labels_clean = labels[mask]
        # create new mask
        new_mask = np.isin(mask_p[i], labels_clean) * mask_p[i]
        mask_clean[i] = new_mask


    # make output zarr store
    out_mask_path = out_dir / (project + f"_well{well_num:04}_mask.zarr")
    out_mask_zarr = zarr.open(out_mask_path, mode="w", shape=mask_clean.shape, dtype=mask_clean.dtype, chunks=(1,) + mask_clean.shape[1:])
    out_image_path = out_dir / (project + f"_well{well_num:04}_image.zarr")
    out_image_zarr = zarr.open(out_image_path, mode="w", shape=im_p.shape, dtype=im_p.dtype, chunks=(1,) + im_p.shape[1:])
    # transfer metadata
    for key in mask_zarr.attrs:
        out_image_zarr.attrs[key] = mask_zarr.attrs[key]
    for key in image_zarr.attrs:
        out_image_zarr.attrs[key] = image_zarr.attrs[key]

    # write data
    out_mask_zarr[:] = mask_clean
    out_image_zarr[:] = im_p




print("Check")

