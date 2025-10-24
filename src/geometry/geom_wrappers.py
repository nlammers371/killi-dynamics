from src.geometry.sphere import fit_sphere
import numpy as np
from typing import Iterable
from skimage.measure import regionprops_table
from src.geometry.spherical_harmonics import fit_sh_healpix
from tqdm import tqdm
import pandas as pd
import zarr
from pathlib import Path

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

def fit_sh_trend(root: Path | str,
                 project_name: str,
                 seg_type: str = "li_segmentation",
                 L_max: int = 10,
                 sh_ridge: float = 0.0,
                 nside: int | None = 16,
                 well: int | None = None,):

    if nside is None:
        nside = max(1, 2*int(np.ceil((L_max + 1) / 2)))

    # set up path
    root = Path(root)
    out_root = root / "output_data" / "sphere_projections" / "surf_fields"
    out_root.mkdir(parents=True, exist_ok=True)
    if well is not None:
        field_path = out_root / f"{project_name}_well{well:04}_sphere_field.zarr"
        mask_path = root / "built_data" / "mask_stacks" / seg_type / f"{project_name}_well{well:04}_masks.zarr"
        sphere_fit_path = out_root / f"{project_name}_well{well:04}_sphere_fits.zarr"
    else:
        field_path = out_root / f"{project_name}_sphere_field.zarr"
        mask_path = root / "built_data" / "mask_stacks" / seg_type / f"{project_name}_masks.zarr"
        sphere_fit_path = out_root / f"{project_name}_sphere_fits.zarr"

    # load sphere fits
    sphere_df = pd.read_csv(sphere_fit_path)
    mask_store = zarr.open(mask_path, mode="r")
    mask_zarr = mask_store["clean"]
    n_t = mask_zarr.shape[0]
    scale_vec = np.array(mask_store.attrs["voxel_size_um"])

    # initialize output zarr store
    field_store = zarr.open(field_path, mode="a")
    if "sh_radius" in field_store:
        del field_store["sh_radius"]

    npix = 12 * nside ** 2
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    sh_zarr = field_store.create_dataset(
        "sh_radius",
        shape=(n_t, npix),
        dtype=np.float32,
        chunks=(1, npix),
        compression=compressor,
    )
    sh_zarr.attrs["nside"] = nside
    sh_zarr.attrs["healpix_order"] = "nested"
    sh_zarr.attrs["L_max"] = L_max
    sh_zarr.attrs["sh_coeffs"] = dict({})
    # iterate through time points and fit SH surface
    for t in tqdm(range(n_t), desc="Fitting spherical harmonic surface trends (beta phase)..."):
        mask = mask_zarr[t]
        center = np.array([
            sphere_df.loc[sphere_df["t"] == t, "center_z_smooth"].values[0],
            sphere_df.loc[sphere_df["t"] == t, "center_y_smooth"].values[0],
            sphere_df.loc[sphere_df["t"] == t, "center_x_smooth"].values[0],
        ])
        # get points
        props = regionprops_table(mask, spacing=scale_vec, properties=("centroid",))
        points = np.column_stack((
            props["centroid-0"],
            props["centroid-1"],
            props["centroid-2"],
        ))
        radius = sphere_df.loc[sphere_df["t"] == t, "radius_smooth"].values[0]
        coeffs, r_fit = fit_sh_healpix(
            points=points[:, ::-1],
            center=center[::-1],
            radius=radius,
            L_max=L_max,
            nside=nside,
            ridge=sh_ridge
        )
        sh_zarr[t, :] = r_fit.astype(np.float32)
        # Fetch safely with a default empty dict
        sh_dict = dict(sh_zarr.attrs.get("sh_coeffs", {}))

        # Update current timepoint (always stringify keys for JSON compatibility)
        sh_dict[t] = coeffs.tolist()

        # Overwrite full dict back to the root attrs
        sh_zarr.attrs["sh_coeffs"] = sh_dict

def fit_surf_sphere_trend(
        root: Path | str,
        project_name: str,
        seg_type: str = "li_segmentation",
        well: int | None = None,
        rad_pct: float = 0.25,
        center_smooth_window: int = 5,
        sm_outlier_thresh: float = 3.0,
        overwrite: bool = False,
) -> pd.DataFrame:

    # set up path
    root = Path(root)
    out_root = root / "geometry"
    out_root.mkdir(parents=True, exist_ok=True)
    if well is not None:
        mask_path = root / "built_data" / "mask_stacks" / seg_type / f"{project_name}_well{well:04}_masks.zarr"
        sphere_fit_path = out_root / f"{project_name}_well{well:04}_sphere_fits.csv"
    else:
        mask_path = root / "built_data" / "mask_stacks" / seg_type / f"{project_name}_masks.zarr"
        sphere_fit_path = out_root / f"{project_name}_sphere_fits.csv"

    # get basic stats
    mask_store = zarr.open(mask_path, mode="r")
    mask_zarr = mask_store["clean"]
    n_t = mask_zarr.shape[0]
    scale_vec = np.array(mask_store.attrs["voxel_size_um"])

    # set up output paths
    if sphere_fit_path.exists() and not overwrite:
        print(f"Loading existing sphere fits from {sphere_fit_path}")
        return pd.read_csv(sphere_fit_path)

    records = []
    for t in tqdm(range(n_t), "Fitting spheres to timepoints..."):
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

        records.append(
            {
                "well": well,
                "t": t,
                "rad_pct": rad_pct,
                "center_z": center[0],
                "center_y": center[1],
                "center_x": center[2],
                "radius": rad_inner,
            }
        )

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

    sphere_df.to_csv(sphere_fit_path, index=False)

    return sphere_df

