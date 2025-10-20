"""High-level quality-control pipeline for segmentation masks."""
from __future__ import annotations

import multiprocessing
import os
from functools import partial
from typing import Dict, Iterable, Mapping, Sequence
from pathlib import Path
import numpy as np
import zarr
from filelock import FileLock
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .morphology import filter_by_eccentricity
from .shadows import filter_shadowed_labels
from .volumes import filter_by_minimum_volume


def compute_qc_keep_labels(
    mask: np.ndarray,
    scale_vec: Sequence[float],
    min_nucleus_vol: float,
    z_prox_thresh: float,
    max_eccentricity: float,
    min_overlap: float,
    min_minor_radius: float = 2.0,
) -> np.ndarray:
    """Return the labels that survive volume, shadow, and morphology filters."""
    filtered, keep = filter_by_minimum_volume(mask, scale_vec, min_nucleus_vol)
    if keep.size == 0:
        return np.array([], dtype=np.int32)

    filtered, keep = filter_shadowed_labels(filtered, scale_vec, z_prox_thresh, min_overlap)
    if keep.size == 0:
        return np.array([], dtype=np.int32)

    filtered, keep = filter_by_eccentricity(filtered, scale_vec, max_eccentricity, min_minor_radius)
    return keep


def persist_keep_labels(zarr_path: str, frame: int, keep_labels: Sequence[int]) -> None:
    """Store keep labels for ``frame`` inside ``mask_keep_ids`` attribute."""
    lock = FileLock(f"{zarr_path}.lock")
    with lock:
        group = zarr.open(zarr_path, mode="a")
        meta = dict(group.attrs.get("mask_keep_ids", {}))
        meta[str(int(frame))] = list(map(int, keep_labels))
        group.attrs["mask_keep_ids"] = meta


# ---------------------------------------------------------------------
# 2. Perform QC for a single frame and write clean mask
# ---------------------------------------------------------------------
def perform_mask_qc(
    t_int: int,
    mask_in: Mapping[int, np.ndarray],
    clean_zarr: zarr.Array,
    scale_vec: Iterable[float],
    min_nucleus_vol: float,
    z_prox_thresh: float,
    max_eccentricity: float,
    min_overlap: float,
) -> np.ndarray:
    """Run QC for a single time point and write cleaned mask to Zarr."""
    mask = np.asarray(mask_in[t_int]).squeeze()
    keep_labels = compute_qc_keep_labels(
        mask=mask,
        scale_vec=scale_vec,
        min_nucleus_vol=min_nucleus_vol,
        z_prox_thresh=z_prox_thresh,
        max_eccentricity=max_eccentricity,
        min_overlap=min_overlap,
    )

    if keep_labels.size == 0:
        clean_zarr[t_int] = np.zeros_like(mask, dtype=np.uint16)
    else:
        # keep only good labels
        clean_mask = np.isin(mask, keep_labels) * mask
        clean_zarr[t_int] = clean_mask.astype(np.uint16)

    return keep_labels

# ---------------------------------------------------------------------
# 3. Wrapper for full QC pipeline
# ---------------------------------------------------------------------
def mask_qc_wrapper(
    root: Path | str,
    project: str,
    mask_type: str = "li_segmentation",
    min_nucleus_vol: float = 75.0,
    z_prox_thresh: float = -30.0,
    max_eccentricity: float = 4.5,
    min_shadow_overlap: float = 0.35,
    last_i: int | None = None,
    overwrite: bool = False,
    n_workers: int = 1,
) -> Dict[int, np.ndarray]:
    """Process all frames of the fused mask Zarr and create a clean sub-dataset."""
    root = Path(root)
    mask_path = root / "built_data" / "mask_stacks" / mask_type / f"{project}_masks.zarr"
    mask_store = zarr.open(mask_path, mode="a")

    # choose source mask (e.g., 'aff')
    mask_in = mask_store["stitched"]
    if last_i is None:
        last_i = mask_in.shape[0]

    par_flag = n_workers is not None and n_workers > 1
    scale_vec = (
        mask_store["stitched"].attrs.get("voxel_size_um")
        or mask_store["stitched"].attrs.get("pixel_size_um")
    )
    if scale_vec is None:
        raise KeyError("Missing 'voxel_size_um' or 'pixel_size_um' in attrs.")

    # ---------------------------------------------------------
    # Create or reset "clean" dataset
    # ---------------------------------------------------------
    if "clean" in mask_store:
        if overwrite:
            del mask_store["clean"]
        else:
            clean_zarr = mask_store["clean"]
    if "clean" not in mask_store:
        clean_zarr = mask_store.create_dataset(
            "clean",
            shape=mask_in.shape,
            dtype=np.uint16,
            chunks=mask_in.chunks,
        )

    # ---------------------------------------------------------
    # Indices to process
    # ---------------------------------------------------------
    all_indices = set(range(last_i))
    existing = (
        set(np.nonzero([np.any(mask_store["clean"][i]) for i in range(last_i)])[0])
        if "clean" in mask_store and not overwrite
        else set()
    )
    write_indices = sorted(all_indices if overwrite else (all_indices - existing))

    # ---------------------------------------------------------
    # Parallel/serial QC
    # ---------------------------------------------------------
    run = partial(
        perform_mask_qc,
        mask_in=mask_in,
        clean_zarr=clean_zarr,
        scale_vec=scale_vec,
        min_nucleus_vol=min_nucleus_vol,
        z_prox_thresh=z_prox_thresh,
        max_eccentricity=max_eccentricity,
        min_overlap=min_shadow_overlap,
    )

    results: Dict[int, np.ndarray] = {}
    if write_indices:
        if par_flag:
            keep_lists = process_map(run, write_indices, max_workers=n_workers, chunksize=1)
            results.update({idx: np.asarray(keep) for idx, keep in zip(write_indices, keep_lists)})
        else:
            for idx in tqdm(write_indices, desc="Mask QC"):
                results[int(idx)] = run(idx)

    return results
