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



def project_to_healpix(mask, center, radius, scale_vec, nside=64,  dist_thresh=50.0):

    """
    Project a binary mask to a Healpix sphere.
    """
    # get cell centroids in physical units
    props = regionprops_table(mask, spacing=scale_vec, properties=("centroid",))
    coords = np.column_stack([props["centroid-0"], props["centroid-1"], props["centroid-2"]])

    # restrict to spherical shell
    dR = np.linalg.norm(coords - center[None, :], axis=1) - radius
    mask = np.abs(dR) <= dist_thresh
    coords = coords[mask]

    # spherical angles
    rel = coords - center[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arccos(np.clip(rel[:, 0] / r, -1, 1))
    phi = np.arctan2(rel[:, 1], rel[:, 2]) % (2*np.pi)

    # map to healpix pixels
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, theta, phi)

    # counts per pixel
    counts = np.bincount(pix, minlength=npix)

    return counts


def project_masks_to_healpix(
    w: int,
    mask_list: Path,
    out_root: Path,
    model_name: str,
    nside: int,
    dist_thresh: float,
):
    """
    Project a well's volumes into Healpix sphere using precomputed sphere fits.
    """
    mask_path = [m for m in mask_list if int(re.search(r"well(\d+)", str(m)).group(1)) == w][0]
    mask_zarr = zarr.open(mask_path, mode="r")
    n_t, *_ = mask_zarr.shape
    scale_vec = np.array(mask_zarr.attrs["voxel_size_um"])

    # Load sphere fits if not passed in
    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if not sphere_csv.exists():
        raise FileNotFoundError(f"No sphere fits found for well {w}")
    sphere_df = pd.read_csv(sphere_csv)

    # Output zarr for projected fields
    field_path = out_root / f"well{w:04}_fields_{nside:04}.zarr"
    field_store = zarr.open(field_path, mode="a")

    # Delete existing subgroup if present
    if "density" in field_store:
        del field_store["density"]

    field_arr = field_store.create_dataset(
        "density",
        shape=(n_t, 12 * nside ** 2),
        chunks=(1, 12 * nside ** 2),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )
    
    # field_arr.attrs["mode"] = proj_mode
    field_arr.attrs["nside"] = nside
    field_arr.attrs["model_name"] = model_name
    field_arr.attrs["dist_thresh"] = dist_thresh

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
        cell_counts = project_to_healpix(
            mask,
            center=center,
            radius=radius,
            scale_vec=scale_vec,
            nside=nside,
            dist_thresh=dist_thresh,
        )
        field_arr[t] = cell_counts

    return field_arr



def density_projection_wrapper(
        root: Union[Path, str],
        project_name: str,
        model_name: str,
        wells: Union[list[int], None] = None,
        dist_thresh: float = 50.0,
        nside: int = 64,
        n_jobs: int = 1,
):
    """
    Parallel wrapper across wells.
    """

    zarr_path = Path(root) / "built_data" / "mask_stacks" / model_name / project_name
    mask_list = sorted(list(Path(zarr_path).glob("*_mask_aff.zarr")))

    # output path
    out_root = Path(root) / "output_data" / "sphere_projections" / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        wells = [int(re.search(r"well(\d+)", str(s)).group(1)) for s in mask_list]

    ##############################
    # Fit spheres

    run_density_calc = partial(project_masks_to_healpix,
                             mask_list=mask_list,
                             out_root=out_root,
                             model_name=model_name,
                             nside=nside,
                             dist_thresh=dist_thresh,)

    if n_jobs == 1:
        _ = [run_density_calc(a) for a in tqdm(wells, desc="Calculating density fields")]
    else:
        _ = process_map(
            run_density_calc,
            wells,
            max_workers=n_jobs,
            chunksize=1,
            desc="Calculating density fields",
        )