"""Sphere fitting utilities."""
from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import least_squares
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def fit_sphere_with_percentile(
    points_phys: np.ndarray,
    im_shape: Sequence[float],
    R0: float | None = None,
    weights: Iterable[float] | None = None,
    loss: str = "huber",
    max_nfev: int = 1000,
    pct: float = 0.9,
) -> tuple[np.ndarray, float, float]:
    """Fit a sphere center (and optional radius) to 3D points."""
    pts = np.asarray(points_phys, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_phys must be (N,3)")

    weights_arr = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float).clip(min=0)

    c0 = np.array([im_shape[0] + 50, im_shape[1] / 2, im_shape[2] / 2], dtype=float)
    if R0 is None:
        R0 = np.mean(np.linalg.norm(pts - c0[None, :], axis=1))
        f_scale = 1.2 * R0
        fit_radius = True
    else:
        fit_radius = False
        f_scale = R0

    def residuals(params):
        center = params[:3]
        radius = params[3] if fit_radius else R0
        d = np.linalg.norm(pts - center[None, :], axis=1)
        return (d - radius) * np.sqrt(weights_arr)

    p0 = np.hstack([c0, R0])
    lb = [0, im_shape[1] // 3, im_shape[2] // 3, 400]
    ub = [2000, 2 * im_shape[1] // 3, 2 * im_shape[2] // 3, 700]
    p0 = np.clip(p0, lb, ub)

    res = least_squares(residuals, p0, loss=loss, f_scale=f_scale, bounds=(lb, ub), max_nfev=max_nfev)
    center_fit = res.x[:3]
    radius_fit = res.x[3] if fit_radius else R0

    d_all = np.linalg.norm(pts - center_fit[None, :], axis=1)
    radius_pct = np.quantile(d_all, pct)
    return center_fit, radius_fit, radius_pct


def create_sphere_mesh(center: Iterable[float], radius: float, resolution: int = 50):
    """Return vertices and faces for a sphere surface."""
    center = np.asarray(center, dtype=float)
    phi, theta = np.mgrid[0.0:np.pi:complex(0, resolution), 0.0:2.0 * np.pi:complex(0, resolution)]
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx0 = i * resolution + j
            idx1 = idx0 + 1
            idx2 = idx0 + resolution
            idx3 = idx2 + 1
            faces.append([idx0, idx2, idx1])
            faces.append([idx1, idx2, idx3])
    return vertices, np.array(faces, dtype=np.int32)


def make_sphere_mesh(n_phi: int, n_theta: int, center: Iterable[float], radius: float):
    """Compatibility wrapper matching the legacy ``make_sphere_mesh`` signature."""
    center = np.asarray(center, dtype=float)
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    Theta, Phi = np.meshgrid(thetas, phis, indexing="ij")
    Xs = radius * np.sin(Theta) * np.cos(Phi) + center[2]
    Ys = radius * np.sin(Theta) * np.sin(Phi) + center[1]
    Zs = radius * np.cos(Theta) + center[0]
    verts = np.stack([Zs.ravel(), Ys.ravel(), Xs.ravel()], axis=1)

    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            p0 = i * n_phi + j
            p1 = i * n_phi + (j + 1) % n_phi
            p2 = (i + 1) * n_phi + j
            p3 = (i + 1) * n_phi + (j + 1) % n_phi
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])
    return verts, np.array(faces, dtype=np.int32)


def fit_sphere(points: np.ndarray, quantile: float = 0.95):
    """Fit a sphere via least squares and report inner/outer radii."""
    points = np.asarray(points, dtype=float)
    center_init = np.mean(points, axis=0)

    def residuals(c):
        r = np.linalg.norm(points - c, axis=1)
        return r - r.mean()

    result = least_squares(residuals, center_init)
    center = result.x
    distances = np.linalg.norm(points - center, axis=1)
    radius = distances.mean()
    inner_radius = np.quantile(distances, 1 - quantile)
    outer_radius = np.quantile(distances, quantile)
    return center, radius, inner_radius, outer_radius


def fit_spheres_for_well(
    w: int,
    mask_list: Sequence[Path],
    out_root: Path,
    rad_um: float | None = None,
    rad_pct: float = 0.25,
    outlier_thresh: float = 75.0,
    smooth_centers: bool = True,
    center_smooth_window: int = 5,
    sm_outlier_thresh: float = 3.0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Fit spheres across time points for a single well."""
    mask_path = [m for m in mask_list if int(re.search(r"well(\d+)", str(m)).group(1)) == w][0]
    mask_zarr = zarr.open(mask_path, mode="r")
    n_t = mask_zarr.shape[0]
    scale_vec = np.array(mask_zarr.attrs["voxel_size_um"])

    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if sphere_csv.exists() and not overwrite:
        return pd.read_csv(sphere_csv)

    records = []
    for t in range(n_t):
        mask = mask_zarr[t]
        points_phys = np.array(np.nonzero(mask)).T * scale_vec[None, :]
        center_raw, rad_raw, rad_inner_raw = fit_sphere_with_percentile(
            points_phys,
            im_shape=np.multiply(scale_vec, mask.shape),
            pct=rad_pct,
            loss="linear",
            R0=rad_um,
        )
        dists = np.sqrt(np.sum((points_phys - center_raw[None, :]) ** 2, axis=1))
        inlier_mask = np.abs(dists - rad_inner_raw) < outlier_thresh

        center, radius_fit, radius_inner = fit_sphere_with_percentile(
            points_phys[inlier_mask],
            im_shape=np.multiply(scale_vec, mask.shape),
            pct=0.25,
            loss="linear",
            R0=None,
        )

        records.append(
            {
                "well": w,
                "t": t,
                "rad_pct": rad_pct,
                "center_z": center[0],
                "center_y": center[1],
                "center_x": center[2],
                "radius": radius_inner,
            }
        )

    sphere_df = pd.DataFrame(records)

    if smooth_centers:
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

    sphere_df.to_csv(sphere_csv, index=False)
    return sphere_df


def sphere_fit_wrapper(
    root: Union[Path, str],
    project_name: str,
    model_name: str,
    wells: Sequence[int] | None = None,
    rad_um: float | None = None,
    rad_outlier_thresh: float = 50.0,
    smooth_centers: bool = True,
    center_smooth_window: int = 3,
    sm_outlier_thresh: float = 2.0,
    n_jobs: int = 1,
    overwrite_sphere_centers: bool = False,
) -> None:
    """Parallel wrapper to fit spheres for all wells in a project."""
    zarr_path = Path(root) / "built_data" / "mask_stacks" / model_name / project_name
    mask_list = sorted(Path(zarr_path).glob("*_mask_aff.zarr"))

    out_root = Path(root) / "output_data" / "sphere_projections" / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        wells = [int(re.search(r"well(\d+)", str(s)).group(1)) for s in mask_list]

    run = partial(
        fit_spheres_for_well,
        mask_list=mask_list,
        rad_um=rad_um,
        out_root=out_root,
        rad_pct=0.25,
        smooth_centers=smooth_centers,
        center_smooth_window=center_smooth_window,
        outlier_thresh=rad_outlier_thresh,
        sm_outlier_thresh=sm_outlier_thresh,
        overwrite=overwrite_sphere_centers,
    )

    if n_jobs == 1:
        for w in tqdm(wells, desc="Fitting spheres"):
            run(w)
    else:
        process_map(run, wells, max_workers=n_jobs, chunksize=1, desc="Fitting spheres")
