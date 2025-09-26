import numpy as np
import pandas as pd
from pathlib import Path
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor

from src.killi_stats.surface_stats import (
    fit_sphere,
    remove_background_dog,
    project_to_sphere,
    smooth_spherical_grid, project_to_healpix,
)

def _dispatch_process_well(args):
    return _process_well(*args)

def _process_well(
    w: int,
    image_path: Path,
    out_root: Path,
    channels: list[int],
    R_um: float,
    nside: int,
    sigma_small_um: float,
    sigma_large_um: float,
    proj_mode: str,
    dist_thresh: float,
    dog_thresh: float,
    smooth_centers: bool,
    center_smooth_window: int,
    outlier_thresh: float,
    sphere_fit_channel: int = 1,
):
    """
    Process a single well: sphere fitting, smoothing, projection, smoothing fields.
    Returns (sphere_df_well, field_db_well).
    """
    # make output path

    # open image
    im_zarr = zarr.open(image_path, mode="r")
    n_ch, n_t, *_ = im_zarr.shape
    scale_vec = np.array(im_zarr.attrs["voxel_size_um"])

    if channels is None:
        channels = list(range(n_ch))

    # pass 1: fit spheres
    sphere_records = []
    for t in range(n_t):
        ch_fit = channels[sphere_fit_channel]  # channel to use for sphere fitting
        im = np.squeeze(im_zarr[ch_fit, t])
        dog = remove_background_dog(
            im, scale_vec=scale_vec,
            sigma_small_um=sigma_small_um,
            sigma_large_um=sigma_large_um
        )
        thresh = np.percentile(dog, dog_thresh)
        mask = dog > thresh
        points_phys = np.array(np.nonzero(mask)).T * scale_vec[None, :]

        center, radius = fit_sphere(points_phys, im_shape=np.multiply(scale_vec, mask.shape), R0=R_um)
        if R_um is not None:
            radius = R_um

        sphere_records.append({
            "well": w,
            "t": t,
            "center_z": center[0],
            "center_y": center[1],
            "center_x": center[2],
            "radius": radius,
        })

    sphere_df_well = pd.DataFrame(sphere_records)

    # pass 2: smooth centers
    if smooth_centers:
        sub = sphere_df_well[["center_x", "center_y", "center_z", "radius"]]
        smoothed = sub.rolling(center_smooth_window, center=True, min_periods=1).mean()
        resid = sub[["center_x", "center_y", "center_z"]] - smoothed[["center_x", "center_y", "center_z"]]
        zscores = resid / resid.std(ddof=0)
        outliers = (zscores.abs() > outlier_thresh).any(axis=1)

        sphere_df_well["is_outlier"] = outliers
        sphere_df_well["center_z_smooth"] = smoothed["center_z"].values
        sphere_df_well["center_y_smooth"] = smoothed["center_y"].values
        sphere_df_well["center_x_smooth"] = smoothed["center_x"].values
        sphere_df_well["radius_smooth"]   = smoothed["radius"].values

    # save
    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    sphere_df_well.to_csv(sphere_csv, index=False)

    # pass 3: projection
    field_store = zarr.open(out_root / f"well{w:04}_fields.zarr", mode="w")

    # create datasets with known shape
    raw_arr = field_store.create_dataset(
        "raw",
        shape=(n_t, len(channels), 12*nside**2),
        chunks=(1, 1, 12*nside**2),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )

    for t in range(n_t):
        row = sphere_df_well.loc[sphere_df_well.t == t].iloc[0]
        center = np.array([row.center_x_smooth, row.center_y_smooth, row.center_z_smooth])
        radius = row.radius_smooth

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

            raw_arr[t, i] = values


    return {}


def process_dataset(
    root: Path | str,
    project_name: str,
    wells: list[int] | None = None,
    channels: list[int] | None = None,
    R_um: float | None = None,
    nside: int = 64,
    sigma_small_um: float = 2.0,
    sigma_large_um: float = 8.0,
    proj_mode: str = "mean",
    dist_thresh: float = 50.0,
    smooth_centers: bool = True,
    center_smooth_window: int = 3,
    outlier_thresh: float = 2.0,
    dog_thresh: float = 99.0,
    n_jobs: int = 1,
    sphere_fit_channel: int = 0,
):
    """
    Parallel wrapper across wells.
    """

    zarr_path = Path(root) / "built_data" / "zarr_image_files" / project_name
    image_list = sorted(list(Path(zarr_path).glob("*.zarr")))
    image_list = [im for im in image_list if "_z.zarr" not in str(im)]

    # output path
    out_root = Path(root) / "output_data" / "sphere_projections" / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        wells = list(range(len(image_list)))

    # prepare args for each well
    args = [
        (
            w,
            image_list[w],
            out_root,
            channels,
            R_um,
            nside,
            sigma_small_um,
            sigma_large_um,
            proj_mode,
            dist_thresh,
            dog_thresh,
            smooth_centers,
            center_smooth_window,
            outlier_thresh,
            sphere_fit_channel
        )
        for w in wells
    ]

    if n_jobs == 1:
        [_process_well(*a) for a in tqdm(args, desc="Processing wells")]
    else:
        process_map(
            _dispatch_process_well,
            args,
            max_workers=n_jobs,
            chunksize=1,
            desc="Processing wells",
        )

    # collect results
    # sphere_df = pd.concat([res[0] for res in results], ignore_index=True)
    # field_db = {w: res[1] for w, res in zip(wells, results)}

    return {}
