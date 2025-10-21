"""Threshold estimation utilities for volumetric segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from func_timeout import FunctionTimedOut, func_timeout
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage as ski
from scipy.interpolate import interp1d
import statsmodels.api as sm
from tqdm import tqdm

from src.data_io.zarr_utils import open_experiment_array

def estimate_li_thresh(
    root: Path | str,
    project_name: str,
    interval: int = 125,
    nuclear_channel: int | None = None,
    tol: float | None = None,
    initial_guess: float | None = None,
    start_i: int = 0,
    last_i: int | None = None,
    timeout: float = 60 * 6,
    use_subsample: bool = False,
):
    """Estimate Li thresholds on a coarse grid for a project."""
    root = Path(root)
    out_directory = root / "built_data" / "mask_stacks" / "li_segmentation"
    out_directory.mkdir(exist_ok=True, parents=True)

    image_zarr, _store_path, _resolved_side = open_experiment_array(root, project_name)
    channel_list = image_zarr.attrs["channels"]
    multichannel_flag = len(channel_list) > 1
    if nuclear_channel is None:
        if multichannel_flag:
            nuclear_channel = [
                i for i in range(len(channel_list)) if ("H2B" in channel_list[i]) or ("nls" in channel_list[i])
            ][0]
        else:
            nuclear_channel = 0

    if last_i is None:
        last_i = image_zarr.shape[0]

    thresh_frames = np.arange(start_i, last_i, interval)
    if len(thresh_frames) > 0:
        thresh_frames[-1] = last_i - 1

    li_vec = []
    frame_vec = []
    for time_int in tqdm(thresh_frames, "Estimating Li thresholds..."):
        if multichannel_flag:
            image_array = np.squeeze(image_zarr[time_int, nuclear_channel, :, :, :]).copy()
        else:
            image_array = np.squeeze(image_zarr[time_int, :, :, :]).copy()
        try:
            _, li_thresh = func_timeout(
                timeout,
                calculate_li_thresh,
                args=(image_array, use_subsample, tol, initial_guess),
            )
            li_vec.append(li_thresh)
            frame_vec.append(time_int)
        except FunctionTimedOut:
            print(f"Function timed out for time: {time_int}")

    li_df_raw = pd.DataFrame(frame_vec, columns=["frame"])
    li_df_raw["li_thresh"] = li_vec

    li_df_raw.to_csv(out_directory / f"{project_name}_li_df.csv", index=False)

    return li_df_raw


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
    mask_root = root / "built_data" / "mask_stacks" / "li_segmentation"
    thresh_files = list(mask_root.glob(f"{project_prefix}*_li_df.csv"))
    s1_flag = np.any(["side1" in f.name for f in thresh_files])
    s2_flag = np.any(["side2" in f.name for f in thresh_files])
    multiside_experiment = s1_flag | s2_flag
    manual_flag = np.any(["_manual" in f.name for f in thresh_files])

    # first the base case
    if not multiside_experiment and not manual_flag:
        li_df = pd.read_csv(mask_root / f"{project_prefix}_li_df.csv")
    elif not multiside_experiment and manual_flag:
        manual_path = mask_root / f"{project_prefix}_li_df_manual.csv"
        li_df = pd.read_csv(manual_path)
    elif multiside_experiment and not manual_flag:
        li_df1 = pd.read_csv(mask_root / f"{project_prefix}_side1_li_df.csv")
        li_df2 = pd.read_csv(mask_root / f"{project_prefix}_side2_li_df.csv")
        li_df = pd.concat([li_df1, li_df2], axis=0, ignore_index=True).sort_values(by="frame")
    elif multiside_experiment and manual_flag:
        manual_path1 = mask_root / f"{project_prefix}_side1_li_df_manual.csv"
        manual_path2 = mask_root / f"{project_prefix}_side2_li_df_manual.csv"
        li_df1 = pd.read_csv(manual_path1)
        li_df2 = pd.read_csv(manual_path2)
        li_df = pd.concat([li_df1, li_df2], axis=0, ignore_index=True).sort_values(by="frame")

    if last_i is None:
        side_hint: Optional[str | Sequence[str]] = None
        if multiside_experiment:
            side_hint = ["side1", "side_00", "side0"]
        image_store, _store_path, _ = open_experiment_array(root, project_prefix, side=side_hint)
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

