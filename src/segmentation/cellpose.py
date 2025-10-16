"""Cellpose-based segmentation helpers.

These utilities centralise the ad-hoc cellpose orchestration that previously
lived in the `src.nucleus_dynamics.build` scripts so that the refactored
segmentation package exposes both the Li-threshold and Cellpose pipelines.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal

import numpy as np
import pandas as pd
import zarr
from cellpose import models
from cellpose.core import use_gpu
from skimage.transform import resize

from src.nucleus_dynamics.utilities.image_utils import calculate_LoG

LOGGER = logging.getLogger(__name__)


def segment_fov(
    column: np.ndarray,
    *,
    model: models.CellposeModel,
    do_3d: bool = True,
    anisotropy: float | None = None,
    diameter: float = 15.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size: int | None = None,
    label_dtype: np.dtype | type[np.uint32] = np.uint32,
    pretrain_flag: bool = False,
):
    """Run Cellpose on a single ZYX volume.

    Parameters mirror the historical implementation so that existing pipelines
    can pass through configuration without change.
    """

    LOGGER.info(
        "[segment_fov] START Cellpose | column: %s %s | do_3d: %s | "
        "diameter: %s | flow threshold: %s",
        type(column),
        column.shape,
        do_3d,
        diameter,
        flow_threshold,
    )

    start = time.perf_counter()
    eval_kwargs: Dict[str, Any] = {
        "channels": [0, 0],
        "do_3D": do_3d,
        "diameter": diameter,
        "anisotropy": anisotropy,
        "cellprob_threshold": cellprob_threshold,
    }

    if pretrain_flag:
        eval_kwargs.update({"min_size": min_size, "augment": False})
    else:
        eval_kwargs.update({"net_avg": False, "augment": False, "flow_threshold": flow_threshold})

    mask, flows, _ = model.eval(column, **eval_kwargs)

    if not do_3d:
        mask = np.expand_dims(mask, axis=0)

    elapsed = time.perf_counter() - start
    LOGGER.info(
        "[segment_fov] END Cellpose | elapsed: %.4fs | mask shape: %s | max(mask): %s",
        elapsed,
        mask.shape,
        np.max(mask) if mask.size else "<empty>",
    )

    label_dtype = np.dtype(label_dtype)
    probs = flows[2]
    grads = flows[1]

    return mask.astype(label_dtype), probs, grads


def _iter_well_indices(image_paths: Iterable[Path]) -> Iterable[int]:
    pattern = re.compile(r"well(\d+)\.zarr$")
    for path in image_paths:
        match = pattern.search(path.name)
        if match:
            yield int(match.group(1))


def cellpose_segmentation(
    *,
    root: str | Path,
    experiment_date: str,
    cell_diameter: float = 30.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
    pretrained_model: str | Path | None = None,
    overwrite: bool = False,
    xy_ds_factor: float = 1.0,
    nuclear_channel: int = 0,
    well_list: Iterable[int] | None = None,
    preproc_sigma: tuple[float, float, float] | None = None,
    min_size: int = 5,
) -> Dict[str, Any]:
    """Run Cellpose on all wells for a given experiment.

    This mirrors the previous behaviour used by the build scripts so callers
    can continue orchestrating segmentation via the refactored package.
    """

    root = Path(root)
    data_directory = root / "built_data" / "zarr_image_files" / experiment_date
    model_name = Path(pretrained_model).name if pretrained_model else model_type
    save_directory = root / "built_data" / "cellpose_output" / model_name / experiment_date
    save_directory.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_directory.glob("*.zarr"))
    if well_list is None:
        well_list = list(_iter_well_indices(image_paths))

    curation_path = root / "metadata" / "curation" / f"{experiment_date}_curation_info.csv"
    has_curation_info = curation_path.exists()
    if has_curation_info:
        curation_df = pd.read_csv(curation_path)
        curation_df_long = pd.melt(
            curation_df,
            id_vars=["series_number", "notes", "tbx5a_flag", "follow_up_flag"],
            var_name="time_index",
            value_name="qc_flag",
        )
        curation_df_long["time_index"] = curation_df_long["time_index"].astype(int)

    for well_index in well_list:
        zarr_path = data_directory / f"{experiment_date}_well{well_index:04}.zarr"
        if not zarr_path.exists():
            LOGGER.warning("Skipping %s: file does not exist", zarr_path)
            continue

        LOGGER.info("Processing %s", zarr_path.name)
        data_tzyx = zarr.open(zarr_path.as_posix(), mode="r")
        pixel_res_raw = data_tzyx.attrs.get("voxel_size_um")
        meta_keys = list(data_tzyx.attrs.keys())
        meta_dict = dict(data_tzyx.attrs)

        if len(data_tzyx.shape) == 5:
            data_tzyx = data_tzyx[nuclear_channel]

        anisotropy_raw = None
        if pixel_res_raw is not None:
            anisotropy_raw = pixel_res_raw[0] / pixel_res_raw[1]

        file_prefix = f"{experiment_date}_well{well_index:04}"
        mask_zarr_path = save_directory / f"{file_prefix}_labels.zarr"
        grad_zarr_path = save_directory / f"{file_prefix}_grads.zarr"
        prob_zarr_path = save_directory / f"{file_prefix}_probs.zarr"

        mask_zarr = zarr.open(
            mask_zarr_path.as_posix(),
            mode="a",
            shape=data_tzyx.shape,
            dtype=np.uint16,
            chunks=(1,) + data_tzyx.shape[1:],
        )
        grad_zarr = zarr.open(
            grad_zarr_path.as_posix(),
            mode="a",
            shape=(data_tzyx.shape[0], 3, data_tzyx.shape[1], data_tzyx.shape[2], data_tzyx.shape[3]),
            dtype=np.float32,
            chunks=(1, 3) + data_tzyx.shape[1:],
        )
        prob_zarr = zarr.open(
            prob_zarr_path.as_posix(),
            mode="a",
            shape=data_tzyx.shape,
            dtype=np.float32,
            chunks=(1,) + data_tzyx.shape[1:],
        )

        for meta_key in meta_keys:
            value = meta_dict[meta_key]
            prob_zarr.attrs[meta_key] = value
            mask_zarr.attrs[meta_key] = value
            grad_zarr.attrs[meta_key] = value

        if pretrained_model:
            model_path = Path(pretrained_model).as_posix()
            prob_zarr.attrs["model_path"] = model_path
            mask_zarr.attrs["model_path"] = model_path
            grad_zarr.attrs["model_path"] = model_path

        time_indices = range(data_tzyx.shape[0])
        if not overwrite:
            time_indices = [
                t
                for t in time_indices
                if not np.any(prob_zarr[t, :, :, :] != 0)
            ]

        for t in reversed(list(time_indices)):
            data_zyx_raw = data_tzyx[t]
            if not np.any(data_zyx_raw > 0):
                LOGGER.info("No image data found for t=%s; skipping", t)
                continue

            dims_orig = data_zyx_raw.shape
            if xy_ds_factor > 1.0:
                dims_new = np.round(
                    [dims_orig[0], dims_orig[1] / xy_ds_factor, dims_orig[2] / xy_ds_factor]
                ).astype(int)
                data_zyx = resize(data_zyx_raw, dims_new, order=1)
            else:
                dims_new = dims_orig
                data_zyx = data_zyx_raw.copy()

            if (pretrained_model and ("log" in model_name or "bkg" in model_name)) and preproc_sigma is not None:
                im_log, im_bkg = calculate_LoG(data_zyx=data_zyx, scale_vec=pixel_res_raw, sigma_dims=preproc_sigma)
                if "log" in model_name:
                    data_zyx = im_log
                elif "bkg" in model_name:
                    data_zyx = im_bkg

            anisotropy = None
            if anisotropy_raw is not None:
                anisotropy = anisotropy_raw * dims_new[1] / dims_orig[1]

            do_3d = data_zyx.shape[0] > 1
            if pretrained_model is not None and not Path(pretrained_model).exists():
                raise ValueError(f"pretrained_model path does not exist: {pretrained_model}")

            gpu = use_gpu()
            if pretrained_model:
                model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
            else:
                model = models.CellposeModel(gpu=gpu, model_type=model_type)

            mask, probs, grads = segment_fov(
                data_zyx,
                model=model,
                do_3d=do_3d,
                anisotropy=anisotropy,
                label_dtype=np.uint32,
                diameter=cell_diameter / xy_ds_factor,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                min_size=min_size,
                pretrain_flag=pretrained_model is not None,
            )

            if xy_ds_factor > 1.0:
                image_mask_out = resize(mask, dims_orig, order=0, anti_aliasing=False, preserve_range=True)
                image_probs_out = resize(probs, dims_orig, order=1)
                image_grads_out = resize(grads, (3,) + dims_orig, order=1)
            else:
                image_mask_out = mask
                image_probs_out = probs
                image_grads_out = grads

            mask_zarr[t] = image_mask_out
            prob_zarr[t] = image_probs_out
            grad_zarr[t] = image_grads_out

    return {}


# Backwards-compatible alias matching the historical CamelCase helper name.
def segment_FOV(*args, **kwargs):  # noqa: N802 - preserve public API
    return segment_fov(*args, **kwargs)


__all__ = ["cellpose_segmentation", "segment_FOV", "segment_fov"]

