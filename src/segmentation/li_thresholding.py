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
import zarr
from types import SimpleNamespace
from src.data_io.zarr_utils import open_experiment_array


# ------------------------------------------------------------------------- #
# Core Li threshold estimation
# ------------------------------------------------------------------------- #
def estimate_li_thresholds_over_time(
    image_zarr: zarr.Array,
    interval: int = 25,
    nuclear_channel: int | None = None,
    tol: float | None = None,
    initial_guess: float | None = None,
    start_i: int = 0,
    last_i: int | None = None,
    timeout: float = 60 * 6,
    use_subsample: bool = False,
) -> pd.DataFrame:
    """Estimate Li thresholds across timepoints on a coarse grid."""
    channel_list = image_zarr.attrs["channels"]
    multichannel_flag = len(channel_list) > 1
    if nuclear_channel is None:
        if multichannel_flag:
            nuclear_channel = [
                i
                for i in range(len(channel_list))
                if ("H2B" in channel_list[i].upper()) or ("NLS" in channel_list[i].upper())
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
                compute_li_threshold_single_frame,
                args=(image_array, use_subsample, tol, initial_guess),
            )
            li_vec.append(li_thresh)
            frame_vec.append(time_int)
        except FunctionTimedOut:
            print(f"Function timed out for time: {time_int}")

    li_df_raw = pd.DataFrame(frame_vec, columns=["frame"])
    li_df_raw["li_thresh"] = li_vec
    return li_df_raw


# ------------------------------------------------------------------------- #
# Utility functions
# ------------------------------------------------------------------------- #
def sample_random_quadrant(vol: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
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


# ------------------------------------------------------------------------- #
# Per-frame Li threshold computation
# ------------------------------------------------------------------------- #
def compute_li_threshold_single_frame(
    image: np.ndarray,
    use_subsample: bool = False,
    tol: Optional[float] = None,
    initial_guess: Optional[float] = None,
    LoG_sigma: float = 1,
    gauss_sigma: Optional[Sequence[float]] = None,
    thresh_li: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Estimate a Li threshold for a single 3D volume."""
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
            working = sample_random_quadrant(working)
        if initial_guess is None:
            initial_guess = ski.filters.threshold_otsu(working)
        if tol is not None:
            thresh_li = ski.filters.threshold_li(working, tolerance=tol, initial_guess=initial_guess)
        else:
            thresh_li = ski.filters.threshold_li(working, initial_guess=initial_guess)
    return data_log_i, thresh_li


# ------------------------------------------------------------------------- #
# Temporal smoothing and interpolation
# ------------------------------------------------------------------------- #
def smooth_li_threshold_trend(
    mask_store_path: Path,
    image_store_path: Path,
    first_i: int = 0,
    last_i: Optional[int] = None,
) -> pd.DataFrame:
    """Smooth Li-threshold estimates across time using LOWESS interpolation."""
    thresh_dir = mask_store_path / "thresholds"
    thresh_files = list(thresh_dir.glob("*li_df.csv"))
    manual_flag = np.any(["_manual" in f.name for f in thresh_files])

    # base case
    if not manual_flag:
        li_df = pd.read_csv(thresh_dir / "li_df.csv")
    else:
        manual_path = thresh_dir / "li_df_manual.csv"
        li_df = pd.read_csv(manual_path)

    if last_i is None:
        image_store = zarr.open_group(image_store_path, mode="r")
        image_zarr = image_store["side_00"]
        last_i = image_zarr.shape[0]

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
    li_df_full.to_csv(thresh_dir / "li_thresh_trend.csv", index=False)
    return li_df_full


# ------------------------------------------------------------------------- #
# Main entry point
# ------------------------------------------------------------------------- #
def run_li_threshold_pipeline(root: Path | str, project_name: str, overwrite_flag: bool = False) -> pd.DataFrame:
    """Run full Li threshold estimation + smoothing pipeline for a project."""
    root = Path(root)
    mask_store_path = root / "built_data" / "mask_stacks" / "li_segmentation" / f"{project_name}masks.zarr"
    image_store_path = root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"

    image_array, _store_path, resolved_side = open_experiment_array(root, project_name)
    side_specs = [SimpleNamespace(**spec) for spec in image_array.attrs["sides"]]

    # initialize mask store
    mode = "a"
    root_group = zarr.open_group(str(mask_store_path), mode=mode)
    root_group.attrs.update(
        {
            "project_name": project_name,
            "sides": [
                {
                    "name": spec.name,
                    "file_prefix": spec.file_prefix,
                    "scene_index": spec.scene_index,
                    "source_type": spec.source_type,
                }
                for spec in side_specs
            ],
        }
    )

    # ensure thresholds directory exists
    thresh_dir = Path(mask_store_path) / "thresholds"
    thresh_dir.mkdir(exist_ok=True)

    # coarse Li estimation
    li_df_raw = estimate_li_thresholds_over_time(
        image_array,
        interval=25,
        nuclear_channel=None,
        tol=None,
        initial_guess=None,
        start_i=0,
        last_i=None,
        timeout=60 * 6,
        use_subsample=True,
    )

    li_df_raw.to_csv(thresh_dir / "li_thresh_raw.csv", index=False)
    li_df_smoothed = smooth_li_threshold_trend(mask_store_path, image_store_path)
    return li_df_smoothed


__all__ = [
    "sample_random_quadrant",
    "compute_li_threshold_single_frame",
    "smooth_li_threshold_trend",
    "estimate_li_thresholds_over_time",
    "run_li_threshold_pipeline",
]
