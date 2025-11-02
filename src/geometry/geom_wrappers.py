from src.geometry.sphere import fit_sphere
import numpy as np
from typing import Iterable
from skimage.measure import regionprops_table
from src.geometry.spherical_harmonics import fit_sh_healpix
from src.data_io.zarr_utils import open_mask_array
from tqdm import tqdm
import pandas as pd
import zarr
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
from src.registration.virtual_fusion import VirtualFuseArray
import logging
import json

def _worker_init():
    logging.getLogger().handlers.clear()  # disable inherited handlers
    logging.disable(logging.CRITICAL)

def call_sphere_fit(t: int,
                    root: Path,
                    project_name: str,
                    seg_type: str = "li_segmentation",
                    side_key: str | None = "fused",
                    rad_quantile: float = .25,
                    well=None):

    if well is not None:
        mask_store = zarr.open(root / "segmentation" / seg_type / f"{project_name}_well{well:04}_masks.zarr", mode="r")
    else:
        mask_store = zarr.open(root / "segmentation" / seg_type / f"{project_name}_masks.zarr", mode="r")
    scale_vec = np.array(mask_store[side_key].attrs["voxel_size_um"])
    mask_zarr, _, _ = open_mask_array(root=root, project_name=project_name, side=side_key)
    mask = mask_zarr[t]
    props = regionprops_table(mask, spacing=scale_vec, properties=("centroid",))
    points_phys = np.column_stack((
        props["centroid-0"],
        props["centroid-1"],
        props["centroid-2"],
    ))
    center, rad, rad_inner = fit_sphere(
                                            points_phys
                                        )

    records = {
            "well": well,
            "t": t,
            "rad_quantile": rad_quantile,
            "center_z": center[0],
            "center_y": center[1],
            "center_x": center[2],
            "radius": rad_inner,
        }
    return records

def sphere_from_mask(mask: np.ndarray,
                     scale_vec: Iterable[float],
                     rad_pct: float = .25, rad_um: float = 15.0) -> tuple[np.ndarray, np.ndarray]:

    props = regionprops_table(mask, spacing=scale_vec, properties=("centroid",))
    points_phys = np.column_stack((
        props["centroid-0"],
        props["centroid-1"],
        props["centroid-2"],
    ))

    center, rad, rad_inner = fit_sphere(
        points_phys,
        rad_quantile=rad_pct
    )

    return rad_inner, center


def _fit_sh_single(
        t: int,
        root: Path,
        project_name: str,
        seg_type: str,
        L_max: int,
        nside: int,
        sh_ridge: float,
        sphere_df: pd.DataFrame,
        scale_vec: np.ndarray,
        field_path: Path,
        verbose_load: bool = False,
):
    """Worker function: fit SH surface for one timepoint.
       Returns (t, coeffs) tuple so main process can write a single JSON."""

    # Silence logging in worker (if running in parallel)
    import logging
    logging.getLogger().handlers.clear()
    logging.disable(logging.CRITICAL)

    # Re-open mask for process isolation
    mask_zarr, _, _ = open_mask_array(
        root=root, project_name=project_name, seg_type=seg_type, verbose=verbose_load
    )
    mask = mask_zarr[t]

    # Get sphere params
    row = sphere_df.loc[sphere_df["t"] == t]
    center = np.array([
        row["center_z_smooth"].values[0],
        row["center_y_smooth"].values[0],
        row["center_x_smooth"].values[0],
    ])
    radius = row["radius_smooth"].values[0]

    # Extract centroids
    props = regionprops_table(mask, spacing=scale_vec, properties=("centroid",))
    points = np.column_stack((
        props["centroid-0"],
        props["centroid-1"],
        props["centroid-2"],
    ))

    # Fit SH surface
    coeffs, r_fit = fit_sh_healpix(
        points=points[:, ::-1],
        center=center[::-1],
        radius=radius,
        L_max=L_max,
        nside=nside,
        ridge=sh_ridge,
    )

    # Write to Zarr radius dataset
    field_store = zarr.open(field_path, mode="a")
    field_store["sh_radius"][t, :] = r_fit.astype(np.float32)

    # Return coeffs for JSON write later
    return t, coeffs.tolist()


def fit_sh_trend(
        root: Path | str,
        project_name: str,
        seg_type: str = "li_segmentation",
        L_max: int = 10,
        sh_ridge: float = 0.0,
        nside: int | None = 16,
        well: int | None = None,
        overwrite: bool = False,
        n_workers: int = 1,
):
    """Compute spherical harmonic surface trends across timepoints."""

    root = Path(root)
    out_root = root / "surf_stats"
    out_root.mkdir(parents=True, exist_ok=True)

    if nside is None:
        nside = max(1, 2 * int(np.ceil((L_max + 1) / 2)))

    if well is not None:
        field_path = out_root / f"{project_name}_well{well:04}_surf_stats.zarr"
        mask_path = root / "segmentation" / seg_type / f"{project_name}_well{well:04}_masks.zarr"
    else:
        field_path = out_root / f"{project_name}_surf_stats.zarr"
        mask_path = root / "segmentation" / seg_type / f"{project_name}_masks.zarr"

    # ---- load inputs ----
    surf_fits_dir = field_path / "surf_fits"
    surf_fits_dir.mkdir(parents=True, exist_ok=True)
    sphere_df = pd.read_csv(surf_fits_dir / "sphere_fits.csv")

    mask_store = zarr.open(mask_path, mode="r")
    scale_vec = np.array(mask_store["side_00"].attrs["voxel_size_um"])
    mask_zarr, _, _ = open_mask_array(root=root, project_name=project_name)
    n_t = mask_zarr.shape[0]

    # ---- initialize zarr outputs ----
    field_store = zarr.open(field_path, mode="a")
    if "sh_radius" in field_store and overwrite:
        del field_store["sh_radius"]

    npix = 12 * nside ** 2
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    sh_zarr = field_store.require_dataset(
        "sh_radius",
        shape=(n_t, npix),
        dtype=np.float32,
        chunks=(1, npix),
        compressor=compressor,
    )
    sh_zarr.attrs.update({
        "nside": nside,
        "healpix_order": "nested",
        "L_max": L_max,
    })

    # ---- build worker ----
    run_fit = partial(
        _fit_sh_single,
        root=root,
        project_name=project_name,
        seg_type=seg_type,
        L_max=L_max,
        nside=nside,
        sh_ridge=sh_ridge,
        sphere_df=sphere_df,
        scale_vec=scale_vec,
        field_path=field_path,
        verbose_load=False,
    )

    # ---- run in parallel or sequential ----
    if n_workers > 1:
        results = process_map(run_fit, range(n_t), max_workers=n_workers, chunksize=1,
                              desc="Fitting spherical harmonic surface trends...")
    else:
        results = []
        for t in tqdm(range(n_t), desc="Fitting spherical harmonic surface trends..."):
            results.append(run_fit(t))

    # ---- merge coefficients and save JSON ----
    coeff_dict = {str(t): coeffs for t, coeffs in results if coeffs is not None}
    json_path = surf_fits_dir / "surf_sh_coeffs.json"
    with open(json_path, "w") as f:
        json.dump(coeff_dict, f, indent=2)

    print(f"[fit_sh_trend] Completed SH fits written to {field_path}")
    print(f"[fit_sh_trend] Coefficients saved to {json_path}")

    return field_path




def fit_surf_sphere_trend(
        root: Path | str,
        project_name: str,
        seg_type: str = "li_segmentation",
        well: int | None = None,
        rad_quantile: float = 0.25,
        center_smooth_window: int = 5,
        sm_outlier_thresh: float = 3.0,
        overwrite: bool = False,
        n_workers: int = 1,
) -> pd.DataFrame:

    par_flag = n_workers is not None and n_workers > 1

    # set up path
    root = Path(root)
    out_root = root / "surf_stats"
    out_root.mkdir(parents=True, exist_ok=True)
    if well is not None:
        # mask_path = root / "segmentation" / seg_type / f"{project_name}_well{well:04}_masks.zarr"
        sphere_fit_path = out_root / f"{project_name}_well{well:04}_surf_stats.zarr"
    else:
        # mask_path = root / "segmentation" / seg_type / f"{project_name}_masks.zarr"
        sphere_fit_path = out_root / f"{project_name}_surf_stats.zarr"
    sphere_fit_path.mkdir(parents=True, exist_ok=True)
    # get basic stats
    mask_zarr, _, _ = open_mask_array(root=root, project_name=project_name)
    n_t = mask_zarr.shape[0]

    # set up output paths
    if sphere_fit_path.exists() and not overwrite:
        print(f"Loading existing sphere fits from {sphere_fit_path}")
        return pd.read_csv(sphere_fit_path)


    run_sphere_fit = partial(call_sphere_fit,
                             root=root,
                             project_name=project_name,
                             seg_type=seg_type,
                             rad_quantile=rad_quantile,)
    if par_flag:
        records = process_map(run_sphere_fit, range(n_t), max_workers=n_workers, chunksize=1,
                              desc="Fitting spheres to timepoints...")
    else:
        records = []
        for t in tqdm(range(n_t), "Fitting spheres to timepoints..."):
            rec = run_sphere_fit(t)
            records.append(rec)

    sphere_df = pd.DataFrame(records)

    sub = sphere_df[["center_x", "center_y", "center_z", "radius"]]
    smoothed = sub.rolling(center_smooth_window, center=True, min_periods=1).mean()
    resid = sub[["center_x", "center_y", "center_z"]] - smoothed[["center_x", "center_y", "center_z"]]
    zscores = resid / resid.std(ddof=0)
    outliers = (zscores.abs() > sm_outlier_thresh).any(axis=1)

    sphere_df["is_outlier"] = outliers
    sphere_df["center_z_smooth"] = smoothed["center_z"].values
    sphere_df["center_y_smooth"] = smoothed["center_y"].values
    sphere_df["center_x_smooth"] = smoothed["center_x"].values
    sphere_df["radius_smooth"] = smoothed["radius"].values

    sphere_df.to_csv(sphere_fit_path / "sphere_fits.csv", index=False)

    return sphere_df

