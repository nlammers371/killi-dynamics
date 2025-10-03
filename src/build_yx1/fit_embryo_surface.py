import numpy as np
from scipy.optimize import least_squares
import pandas as pd
from pathlib import Path
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Union
from functools import partial

def fit_sphere_with_percentile(points_phys, im_shape, R0=None, weights=None,
                               loss="huber", max_nfev=1000, pct=0.9):
    """
    Fit a sphere center (and optionally radius) to 3D points.
    Also compute a robust percentile-based radius.

    Parameters
    ----------
    points_phys : (N,3) array
        Candidate shell points in physical units (z,y,x).
    im_shape : tuple
        Shape of the 3D image (z,y,x), used for bounds/initial guess.
    R0 : float or None
        Known/prior radius. If None, radius is fit; otherwise radius is fixed.
    weights : (N,) array or None
        Optional nonnegative weights (e.g. DoG intensities).
    loss : str
        Robust loss for least_squares ("linear", "huber", "soft_l1", ...).
    max_nfev : int
        Max number of function evaluations for optimizer.
    pct : float
        Percentile of distances to report (0<pct<=1).
    f_scale : float
        Transition scale for robust loss (Huber, soft_l1, etc.).

    Returns
    -------
    c_fit : (3,) array
        Estimated center in physical units.
    R_fit : float
        Estimated radius in physical units (least-squares).
    R_pct : float
        Percentile-based radius (enclosing pct fraction of points).
    """
    pts = np.asarray(points_phys, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_phys must be (N,3)")

    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float).clip(min=0)

    # crude center guess
    c0 = np.array([im_shape[0] + 50, im_shape[1]/2, im_shape[2]/2], dtype=float)

    # crude radius guess
    if R0 is None:
        R0 = np.mean(np.linalg.norm(pts - c0[None, :], axis=1))
        f_scale = 1.2 * R0
        fit_radius = True
    else:
        fit_radius = False
        f_scale = R0

    def residuals(params):
        c = params[:3]
        R = params[3] if fit_radius else R0
        d = np.linalg.norm(pts - c[None, :], axis=1)
        return (d - R) * np.sqrt(w)

    # initial vector
    p0 = np.hstack([c0, R0])
    # bounds
    lb = [0, im_shape[1]//3, im_shape[2]//3, 0]
    ub = [2*im_shape[0], 2*im_shape[1]//3, 2*im_shape[2]//3, 2*R0]

    res = least_squares(residuals, p0, loss=loss, f_scale=f_scale,
                        bounds=(lb, ub), max_nfev=max_nfev)

    c_fit = res.x[:3]
    R_fit = res.x[3] if fit_radius else R0

    # percentile-based radius
    d_all = np.linalg.norm(pts - c_fit[None, :], axis=1)
    R_pct = np.quantile(d_all, pct)

    return c_fit, R_fit, R_pct


def make_sphere_mesh(n_phi, n_theta, center, radius):
    """
    Create a spherical mesh (vertices, faces) suitable for napari.add_surface.

    Parameters
    ----------
    n_phi, n_theta : int
        Angular resolution of spherical mesh.
    radius : float
        Sphere radius.

    Returns
    -------
    verts : (N,3) array
        Vertex coordinates (z,y,x).
    faces : (M,3) array
        Mesh faces as indices into verts.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    Theta, Phi = np.meshgrid(thetas, phis, indexing="ij")
    Xs = radius * np.sin(Theta) * np.cos(Phi) + center[2]
    Ys = radius * np.sin(Theta) * np.sin(Phi) + center[1]
    Zs = radius * np.cos(Theta) + center[0]
    verts = np.stack([Zs.ravel(), Ys.ravel(), Xs.ravel()], axis=1)

    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            p0 = i * n_phi + j
            p1 = i * n_phi + (j+1) % n_phi   # wrap around φ
            p2 = (i+1) * n_phi + j
            p3 = (i+1) * n_phi + (j+1) % n_phi
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])
    faces = np.array(faces, dtype=np.int32)

    return verts, faces


def fit_spheres_for_well(
    w: int,
    mask_list: Path,
    out_root: Path,
    rad_um: float = None,
    rad_pct: float = 0.25,
    outlier_thresh: float = 75.0,
    smooth_centers: bool = True,
    center_smooth_window: int = 5,
    sm_outlier_thresh: float = 3.0,
    overwrite: bool = False,
):
    """
    Fit spheres to a well and save sphere parameters to CSV.
    """
    mask_path = mask_list[w]
    mask_zarr = zarr.open(mask_path, mode="r")
    n_t, *_ = mask_zarr.shape
    scale_vec = np.array(mask_zarr.attrs["voxel_size_um"])

    sphere_csv = out_root / f"well{w:04}_sphere_fits.csv"
    if sphere_csv.exists() and not overwrite:
        sphere_df = pd.read_csv(sphere_csv)
        return sphere_df

    sphere_records = []
    for t in range(n_t):
        mask = mask_zarr[t]
        points_phys = np.array(np.nonzero(mask)).T * scale_vec[None, :]
        # first fit to screen out outliers
        center_raw, rad_raw, rad_inner_raw = \
                                        fit_sphere_with_percentile(
                                                        points_phys,
                                                        im_shape=np.multiply(scale_vec, mask.shape),
                                                        pct=rad_pct,
                                                        loss="linear",
                                                        R0=rad_um)
        # remove outliers and refit
        dists = np.sqrt(np.sum((points_phys - center_raw[None, :]) ** 2, axis=1))
        inlier_mask = np.abs(dists - rad_inner_raw) < outlier_thresh

        center, radius_fit, radius_inner = fit_sphere_with_percentile(
                                                        points_phys[inlier_mask],
                                                        im_shape=np.multiply(scale_vec, mask.shape),
                                                        pct=.25,
                                                        loss="linear",
                                                        R0=None)

        sphere_records.append({
            "well": w, "t": t, "rad_pct": rad_pct,
            "center_z": center[0], "center_y": center[1], "center_x": center[2],
            "radius": radius_inner,
        })

    sphere_df = pd.DataFrame(sphere_records)

    # smoothing/outlier removal
    if smooth_centers:
        sub = sphere_df[["center_x", "center_y", "center_z", "radius"]]
        smoothed = sub.rolling(center_smooth_window, center=True, min_periods=1).mean()
        resid = sub[["center_x","center_y","center_z"]] - smoothed[["center_x","center_y","center_z"]]
        zscores = resid / resid.std(ddof=0)
        outliers = (zscores.abs() > sm_outlier_thresh).any(axis=1)

        sphere_df["is_outlier"] = outliers
        sphere_df["center_z_smooth"] = smoothed["center_z"].values
        sphere_df["center_y_smooth"] = smoothed["center_y"].values
        sphere_df["center_x_smooth"] = smoothed["center_x"].values
        sphere_df["radius_smooth"]   = smoothed["radius"].values

    sphere_df.to_csv(sphere_csv, index=False)

    return sphere_df

def sphere_fit_wrapper(
    root: Union[Path, str],
    project_name: str,
    model_name: str,
    wells: Union[list[int], None] = None,
    rad_um: Union[float, None] = None,
    rad_outlier_thresh: float = 50.0,
    smooth_centers: bool = True,
    center_smooth_window: int = 3,
    sm_outlier_thresh: float = 2.0,
    n_jobs: int = 1,
    overwrite_sphere_centers: bool = False,
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
        wells = list(range(len(mask_list)))

    ##############################
    # Fit spheres

    run_sphere_fit = partial(fit_spheres_for_well,
                             mask_list=mask_list,
                             rad_um=rad_um,
                             out_root=out_root,
                             rad_pct=.25,
                             smooth_centers=smooth_centers,
                             center_smooth_window=center_smooth_window,
                             outlier_thresh=rad_outlier_thresh,
                             sm_outlier_thresh=sm_outlier_thresh,
                             overwrite=overwrite_sphere_centers)

    if n_jobs == 1:
        _ = [run_sphere_fit(a) for a in tqdm(wells, desc="Fitting spheres")]
    else:
        _ = process_map(
                            run_sphere_fit,
                   wells,
                            max_workers=n_jobs,
                            chunksize=1,
                            desc="Fitting spheres",
                        )