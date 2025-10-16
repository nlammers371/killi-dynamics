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



def project_to_healpix(vol, center, radius, scale_vec,
                       nside=64, mode="mean", dist_thresh=50.0):
    
    dz, dy, dx = scale_vec
    Z, Y, X = np.indices(vol.shape)
    coords = np.c_[Z.ravel()*dz, Y.ravel()*dy, X.ravel()*dx]
    vals = vol.ravel().astype(float)

    # restrict to spherical shell
    dR = np.linalg.norm(coords - center[None, :], axis=1) - radius
    mask = np.abs(dR) <= dist_thresh
    coords, vals = coords[mask], vals[mask]

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

    if mode == "sum":
        values = np.bincount(pix, weights=vals, minlength=npix)
    elif mode == "mean":
        sums = np.bincount(pix, weights=vals, minlength=npix)
        values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    elif mode == "max":
        # np.bincount can't do max; need groupby-style reduction
        values = np.full(npix, -np.inf)
        np.maximum.at(values, pix, vals)
        values[values == -np.inf] = 0.0
    else:
        raise ValueError(f"Unknown mode {mode}")

    return values, counts


def project_well_to_healpix(
    w: int,
    image_list: Path,
    out_root: Path,
    channels: list[int],
    nside: int,
    proj_mode: str,
    dist_thresh: float,
):
    """
    Project a well's volumes into Healpix sphere using precomputed sphere fits.
    """
    image_path = [m for m in image_list if int(re.search(r"well(\d+)", str(m)).group(1))==w][0]
    im_zarr = zarr.open(image_path, mode="r")
    n_ch, n_t, *_ = im_zarr.shape
    scale_vec = np.array(im_zarr.attrs["voxel_size_um"])

    # Load sphere fits if not passed in
    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if not sphere_csv.exists():
        raise FileNotFoundError(f"No sphere fits found for well {w}")
    sphere_df = pd.read_csv(sphere_csv)

    # Output zarr for projected fields
    field_path = out_root / f"well{w:04}_fields_{nside:04}.zarr"
    
    field_store = zarr.open(field_path, mode="a")

    # Delete existing subgroup if present
    if proj_mode in field_store:
        del field_store[proj_mode]

    field_arr = field_store.create_dataset(
        proj_mode,
        shape=(n_t, len(channels), 12 * nside ** 2),
        chunks=(1, 1, 12 * nside ** 2),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )
    
    field_arr.attrs["mode"] = proj_mode
    field_arr.attrs["nside"] = nside
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

        for i, ch in enumerate(channels):
            im = np.squeeze(im_zarr[ch, t])
            values, _ = project_to_healpix(
                im,
                center=center,
                radius=radius,
                scale_vec=scale_vec,
                nside=nside,
                mode=proj_mode,
                dist_thresh=dist_thresh,
            )
            field_arr[t, i] = values

    return field_arr



def field_projection_wrapper(
    root: Union[Path, str],
    project_name: str,
    wells: Union[list[int], None] = None,
    channels: Union[list[int], None] = None,
    nside: int = 64,
    proj_mode: str = "mean",
    dist_thresh: float = 50.0,
    n_jobs: int = 1,
    overwrite_fields: bool = False,
):
    """
    Parallel wrapper across wells.
    """

    zarr_path = Path(root) / "built_data" / "zarr_image_files" / project_name
    image_list = sorted(list(Path(zarr_path).glob("*.zarr")))
    image_list = [im for im in image_list if "_z.zarr" not in str(im)]

    if channels is None:
        test_zarr = zarr.open(image_list[0], mode='r')
        channels = [ch for ch in range(test_zarr.shape[0])]

    # output path
    out_root = Path(root) / "output_data" / "sphere_projections" / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        sphere_fits_list = sorted(list(out_root.glob("well????_sphere_fits.csv")))
        wells = [int(re.search(r"well(\d+)", str(s)).group(1)) for s in sphere_fits_list]

    ##############################
    # Project to Healpix spheres
    run_projection = partial(
                            project_well_to_healpix,
                            image_list=image_list,
                            out_root=out_root,
                            channels=channels,
                            nside=nside,
                            proj_mode=proj_mode,
                            dist_thresh=dist_thresh,
                        )

    if n_jobs == 1:
        proj_results = [
            run_projection(w=w)
            for w in tqdm(wells, desc="Doing projections...")
        ]
    else:
        proj_results = process_map(
            run_projection,
            wells,
            max_workers=n_jobs,
            chunksize=1,
            desc="Doing projections...",
        )

    return proj_results
