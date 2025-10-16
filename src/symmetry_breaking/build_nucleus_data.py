import numpy as np
import pandas as pd
from pathlib import Path
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from typing import Union
import re
import healpy as hp
from skimage.measure import regionprops_table


def get_surface_nuclei(mask, center, radius, scale_vec, im=None, dist_thresh=50.0):
    """
    Project a binary mask to a Healpix sphere.
    """
    # get cell centroids in physical units
    if im is not None:
        props = regionprops_table(mask, intensity_image=im, spacing=scale_vec, properties=("centroid","area","mean_intensity"))
    else:
        props = regionprops_table(mask, intensity_image=im, spacing=scale_vec, properties=("centroid","area"))
    coords = np.column_stack([props["centroid-0"], props["centroid-1"], props["centroid-2"]])
    area_vec = props["area"]

    # restrict to spherical shell
    dR = np.linalg.norm(coords - center[None, :], axis=1) - radius
    mask = np.abs(dR) <= dist_thresh
    coords = coords[mask]
    area_vec = area_vec[mask]
    if im is not None:
        i_vec = props["mean_intensity"]
        i_vec = i_vec[mask]

    # spherical angles
    rel = coords - center[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arccos(np.clip(rel[:, 0] / r, -1, 1))
    phi = np.arctan2(rel[:, 1], rel[:, 2]) % (2 * np.pi)

    df = pd.DataFrame({
        "x": coords[:, 2],
        "y": coords[:, 1],
        "z": coords[:, 0],
        "r": r,
        "theta": theta,
        "phi": phi,
        "area": area_vec,
    })
    if im is not None:
        df["intensity"] = i_vec

    return df


def build_nucleus_dataset(
        w: int,
        mask_list: Path,
        out_root: Path,
        dT: float,
        dist_thresh: float,
        fluor_channel: Union[int, None] = None,
        image_list: Union[Path, None] = None,
):
    """
    Project a well's volumes into Healpix sphere using precomputed sphere fits.
    """
    mask_path = [m for m in mask_list if int(re.search(r"well(\d+)", str(m)).group(1)) == w][0]
    mask_zarr = zarr.open(mask_path, mode="r")
    n_t, *_ = mask_zarr.shape
    scale_vec = np.array(mask_zarr.attrs["voxel_size_um"])

    if image_list is not None:
        im_path = [im for im in image_list if int(re.search(r"well(\d+)", str(im)).group(1)) == w][0]
        im_zarr = zarr.open(im_path, mode="r")
        # im_zarr = im_zarr[fluor_channel]
    else:
        im_zarr = None

    # Load sphere fits if not passed in
    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if not sphere_csv.exists():
        raise FileNotFoundError(f"No sphere fits found for well {w}")
    sphere_df = pd.read_csv(sphere_csv)

    # Output zarr for projected fields
    df_path = out_root / f"well{w:04}_nucleus_df.csv"

    nucleus_df_list = []

    for t in range(n_t):
        row = sphere_df.loc[sphere_df.t == t].iloc[0]
        # use smoothed values if available
        if "center_z_smooth" in row:
            center = np.array([row.center_z_smooth, row.center_y_smooth, row.center_x_smooth])
            radius = row.radius_smooth
        else:
            center = np.array([row.center_z, row.center_y, row.center_x])
            radius = row.radius

        mask = np.squeeze(mask_zarr[t])
        temp_df = get_surface_nuclei(
            mask,
            center=center,
            radius=radius,
            scale_vec=scale_vec,
            dist_thresh=dist_thresh,
            im=im_zarr[fluor_channel, t] if im_zarr is not None else None,
        )
        temp_df["t_int"] = t
        temp_df["t"] = t * dT
        nucleus_df_list.append(temp_df)

    nucleus_df = pd.concat(nucleus_df_list, ignore_index=True)
    nucleus_df["well"] = w
    # save
    nucleus_df.to_csv(df_path, index=False)

    return nucleus_df


def build_nucleus_df_wrapper(
        root: Union[Path, str],
        project_name: str,
        model_name: str,
        dT: float,
        wells: Union[list[int], None] = None,
        dist_thresh: float = 50.0,
        n_jobs: int = 1,
        fluor_channel: Union[int, None] = None,
):
    """
    Parallel wrapper across wells.
    """

    # to masks
    zarr_path = Path(root) / "built_data" / "mask_stacks" / model_name / project_name
    mask_list = sorted(list(Path(zarr_path).glob("*_mask_aff.zarr")))

    if fluor_channel is not None:
        im_path = Path(root) / "built_data" / "zarr_image_files" / project_name
        image_list = sorted(list(Path(im_path).glob("*.zarr")))
        image_list = [im for im in image_list if "_z.zarr" not in str(im)]
    else:
        image_list = None
    # output path
    out_root = Path(root) / "output_data" / "sphere_projections" / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        wells = [int(re.search(r"well(\d+)", str(s)).group(1)) for s in mask_list]

    ##############################
    # Fit spheres
    run_count_calc = partial(build_nucleus_dataset,
                               mask_list=mask_list,
                               out_root=out_root,
                               dist_thresh=dist_thresh,
                               dT=dT,
                               fluor_channel=fluor_channel,
                               image_list=image_list,)

    if n_jobs == 1:
        _ = [run_count_calc(a) for a in tqdm(wells, desc="Calculating density fields")]
    else:
        _ = process_map(
            run_count_calc,
            wells,
            max_workers=n_jobs,
            chunksize=1,
            desc="Calculating density fields",
        )