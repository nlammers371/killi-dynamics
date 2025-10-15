"""High-level quality-control pipeline for segmentation masks."""
from __future__ import annotations

import multiprocessing
import os
from functools import partial
from typing import Dict, Iterable, Mapping, Sequence

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


def perform_mask_qc(
    t_int: int,
    mask_in: Mapping[int, np.ndarray],
    zarr_path: str,
    scale_vec: Iterable[float],
    min_nucleus_vol: float,
    z_prox_thresh: float,
    max_eccentricity: float,
    min_overlap: float,
) -> np.ndarray:
    """Run QC for a single time point and persist keep labels."""
    mask = np.asarray(mask_in[t_int]).squeeze()
    keep_labels = compute_qc_keep_labels(
        mask=mask,
        scale_vec=scale_vec,
        min_nucleus_vol=min_nucleus_vol,
        z_prox_thresh=z_prox_thresh,
        max_eccentricity=max_eccentricity,
        min_overlap=min_overlap,
    )
    persist_keep_labels(zarr_path, t_int, keep_labels)
    return keep_labels


def mask_qc_wrapper(
    root: str,
    project: str,
    min_nucleus_vol: float = 75.0,
    z_prox_thresh: float = -30.0,
    max_eccentricity: float = 4.5,
    min_shadow_overlap: float = 0.35,
    last_i: int | None = None,
    overwrite: bool = False,
    par_flag: bool = False,
    n_workers: int | None = None,
) -> Dict[int, np.ndarray]:
    """Process all frames of the fused mask Zarr and record keep labels."""
    mask_path = os.path.join(root, "built_data", "mask_stacks", f"{project}_mask_aff.zarr")
    mask_store = zarr.open(mask_path, mode="a")

    if last_i is None:
        last_i = mask_store.shape[0]

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        n_workers = max(1, total_cpus // 2)

    scale_vec = (
        mask_store.attrs["PhysicalSizeZ"],
        mask_store.attrs["PhysicalSizeY"],
        mask_store.attrs["PhysicalSizeX"],
    )

    all_indices = set(range(last_i))
    existing = set(map(int, mask_store.attrs.get("mask_keep_ids", {}).keys()))
    write_indices = sorted(all_indices if overwrite else (all_indices - existing))

    run = partial(
        perform_mask_qc,
        mask_in=mask_store,
        zarr_path=mask_path,
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
