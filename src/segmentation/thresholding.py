"""Threshold estimation utilities for volumetric segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage as ski
import zarr
from scipy.interpolate import interp1d
import statsmodels.api as sm


def extract_random_quadrant(vol: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Return a random quadrant in the YX plane of a ZYX stack."""
    if seed is not None:
        np.random.seed(seed)

    z, y, x = vol.shape
    half_y, half_x = y // 2, x // 2

    quadrant = np.random.choice(4)

    if quadrant == 0:  # top-left
        return vol[:, :half_y, :half_x]
    if quadrant == 1:  # top-right
        return vol[:, :half_y, half_x:]
    if quadrant == 2:  # bottom-left
        return vol[:, half_y:, :half_x]
    return vol[:, half_y:, half_x:]


def calculate_li_thresh(
    image: np.ndarray,
    use_subsample: bool = False,
    tol: Optional[float] = None,
    initial_guess: Optional[float] = None,
    LoG_sigma: float = 1,
    gauss_sigma: Optional[Sequence[float]] = None,
    thresh_li: Optional[float] = None,
):
    """Estimate a Li threshold for a denoised volume."""
    if gauss_sigma is None:
        gauss_sigma = (1.33, 4, 4)

    gaussian_background = ski.filters.gaussian(image, sigma=gauss_sigma, preserve_range=True)
    data_bkg = image - gaussian_background

    data_log = sitk.GetArrayFromImage(
        sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data_bkg), sigma=LoG_sigma)
    )
    data_log_i = ski.util.invert(data_log)

    if thresh_li is None:
        working = data_log_i
        if use_subsample:
            working = extract_random_quadrant(working)  # downsample
        if initial_guess is None:
            initial_guess = ski.filters.threshold_otsu(working)
        if tol is not None:
            thresh_li = ski.filters.threshold_li(working, tolerance=tol, initial_guess=initial_guess)
        else:
            thresh_li = ski.filters.threshold_li(working, initial_guess=initial_guess)
    return data_log_i, thresh_li


def calculate_li_trend(
    root: Path | str,
    project_prefix: str,
    first_i: int = 0,
    last_i: Optional[int] = None,
    multiside_experiment: bool = True,
) -> pd.DataFrame:
    """Smooth Li-threshold estimates across time for a project."""
    root = Path(root)
    mask_root = root / "built_data" / "mask_stacks"

    if not multiside_experiment:
        manual_path = mask_root / f"{project_prefix}_li_df_manual.csv"
        if manual_path.exists():
            li_df = pd.read_csv(manual_path)
        else:
            li_df = pd.read_csv(manual_path.with_name(manual_path.name.replace("_manual", "")))
    else:
        manual_path1 = mask_root / f"{project_prefix}_side1_li_df_manual.csv"
        manual_path2 = mask_root / f"{project_prefix}_side2_li_df_manual.csv"

        def _load(path: Path) -> pd.DataFrame:
            if path.exists():
                return pd.read_csv(path)
            return pd.read_csv(path.with_name(path.name.replace("_manual", "")))

        li_df = pd.concat([_load(manual_path1), _load(manual_path2)], axis=0, ignore_index=True)

    if last_i is None:
        zarr_name = f"{project_prefix}.zarr" if not multiside_experiment else f"{project_prefix}_side1.zarr"
        zarr_path = root / "built_data" / "zarr_image_files" / zarr_name
        image_store = zarr.open(zarr_path, mode="r")
        last_i = image_store.shape[0]

    x = li_df["frame"].to_numpy()
    y = li_df["li_thresh"].to_numpy()
    y_thresh = np.percentile(y, 95) / 20
    outlier_filter = y > y_thresh
    x = x[outlier_filter]
    y = y[outlier_filter]
    si = np.argsort(x)
    x = x[si] + np.random.rand(len(x)) * 0.01
    y = y[si]

    lowess_result = sm.nonparametric.lowess(y, x, frac=0.3, it=3)
    x_lowess = lowess_result[:, 0]
    y_lowess = lowess_result[:, 1]

    frames_full = np.arange(first_i, last_i)
    thresh_interp = interp1d(x_lowess, y_lowess, kind="linear", fill_value="extrapolate")
    thresh_predictions = thresh_interp(frames_full)

    li_df_full = pd.DataFrame(frames_full, columns=["frame"])
    li_df_full["li_thresh"] = thresh_predictions

    if not multiside_experiment:
        out_path = mask_root / f"{project_prefix}_li_thresh_trend.csv"
        li_df_full.to_csv(out_path, index=False)
    else:
        for side in ("side1", "side2"):
            out_path = mask_root / f"{project_prefix}_{side}_li_thresh_trend.csv"
            li_df_full.to_csv(out_path, index=False)

    return li_df_full


__all__ = ["extract_random_quadrant", "calculate_li_thresh", "calculate_li_trend"]

