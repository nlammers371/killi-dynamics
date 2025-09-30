import numpy as np
import pandas as pd
from pathlib import Path
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src.killi_stats.surface_stats import (
    fit_sphere,
    remove_background_dog,
    project_to_sphere,
    smooth_spherical_grid, project_to_healpix,
)

def _dispatch_sphere_fit(args):
    return fit_spheres_for_well(*args)

def _sphere_projection(args):
    return project_to_sphere(*args)

def fit_spheres_for_well(
    w: int,
    image_list: Path,
    out_root: Path,
    R_um: float,
    sigma_small_um: float,
    sigma_large_um: float,
    dog_thresh: float,
    sphere_fit_channel: int = 1,
    smooth_centers: bool = True,
    center_smooth_window: int = 5,
    outlier_thresh: float = 3.0,
    overwrite: bool = False,
):
    """
    Fit spheres to a well and save sphere parameters to CSV.
    """
    image_path = image_list[w]
    im_zarr = zarr.open(image_path, mode="r")
    n_ch, n_t, *_ = im_zarr.shape
    scale_vec = np.array(im_zarr.attrs["voxel_size_um"])

    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if sphere_csv.exists() and not overwrite:
        sphere_df = pd.read_csv(sphere_csv)
        return sphere_df

    sphere_records = []
    for t in range(n_t):
        im = np.squeeze(im_zarr[sphere_fit_channel, t])
        dog = remove_background_dog(vol=im,
                                    scale_vec=scale_vec,
                                    sigma_large_um=sigma_small_um,
                                    sigma_small_um=sigma_large_um)
        thresh = np.percentile(dog, dog_thresh)
        mask = dog > thresh
        points_phys = np.array(np.nonzero(mask)).T * scale_vec[None, :]

        center, radius = fit_sphere(points_phys, im_shape=np.multiply(scale_vec, mask.shape), R0=R_um)
        if R_um is not None:
            radius = R_um

        sphere_records.append({
            "well": w, "t": t,
            "center_z": center[0], "center_y": center[1], "center_x": center[2],
            "radius": radius,
        })

    sphere_df = pd.DataFrame(sphere_records)

    # smoothing/outlier removal
    if smooth_centers:
        sub = sphere_df[["center_x", "center_y", "center_z", "radius"]]
        smoothed = sub.rolling(center_smooth_window, center=True, min_periods=1).mean()
        resid = sub[["center_x","center_y","center_z"]] - smoothed[["center_x","center_y","center_z"]]
        zscores = resid / resid.std(ddof=0)
        outliers = (zscores.abs() > outlier_thresh).any(axis=1)

        sphere_df["is_outlier"] = outliers
        sphere_df["center_z_smooth"] = smoothed["center_z"].values
        sphere_df["center_y_smooth"] = smoothed["center_y"].values
        sphere_df["center_x_smooth"] = smoothed["center_x"].values
        sphere_df["radius_smooth"]   = smoothed["radius"].values

    sphere_df.to_csv(sphere_csv, index=False)

    return sphere_df


def project_well_to_healpix(
    w: int,
    image_list: Path,
    out_root: Path,
    channels: list[int],
    nside: int,
    proj_mode: str,
    dist_thresh: float,
    # overwrite: bool = False,
):
    """
    Project a well's volumes into Healpix sphere using precomputed sphere fits.
    """
    image_path = image_list[w]
    im_zarr = zarr.open(image_path, mode="r")
    n_ch, n_t, *_ = im_zarr.shape
    scale_vec = np.array(im_zarr.attrs["voxel_size_um"])

    # Load sphere fits if not passed in
    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if not sphere_csv.exists():
        raise FileNotFoundError(f"No sphere fits found for well {w}")
    sphere_df = pd.read_csv(sphere_csv)

    # Output zarr for projected fields
    field_path = out_root / f"well{w:04}_fields.zarr"
    # if field_path.exists() and not overwrite:
    #     return zarr.open(field_path, mode="r")

    field_store = zarr.open(field_path, mode="w")
    raw_arr = field_store.create_dataset(
        "raw",
        shape=(n_t, len(channels), 12*nside**2),
        chunks=(1, 1, 12*nside**2),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )
    raw_arr.attrs["mode"] = proj_mode
    raw_arr.attrs["nside"] = nside
    raw_arr.attrs["dist_thresh"] = dist_thresh

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
            raw_arr[t, i] = values

    return field_store



def project_fields_to_sphere(
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
    overwrite_sphere_centers: bool = False,
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
        wells = list(range(len(image_list)))

    ##############################
    # Fit spheres

    run_sphere_fit = partial(fit_spheres_for_well,
                             image_list=image_list,
                             R_um=R_um,
                             sphere_fit_channel=sphere_fit_channel,
                             out_root=out_root,
                             sigma_small_um=sigma_small_um,
                             sigma_large_um=sigma_large_um,
                             smooth_centers=smooth_centers,
                             center_smooth_window=center_smooth_window,
                             outlier_thresh=outlier_thresh,
                             dog_thresh=dog_thresh,
                             overwrite=overwrite_sphere_centers)

    if n_jobs == 1:
        _ = [run_sphere_fit(a) for a in tqdm(wells, desc="Fitting spheres")]
    else:
        _ = process_map(
                                run_sphere_fit,
                       wells,
                                max_workers=n_jobs,
                                chunksize=1,
                                desc="Processing wells",
                            )

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
            for w in tqdm(wells, desc="Projection")
        ]
    else:
        proj_results = process_map(
            run_projection,
            wells,
            max_workers=n_jobs,
            chunksize=1,
            desc="Projection",
        )

    return proj_results
