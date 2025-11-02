"""High-level quality-control pipeline for segmentation masks."""
from __future__ import annotations

import multiprocessing
import os
import warnings
from functools import partial
from typing import Dict, Iterable, Mapping, Sequence
from pathlib import Path
import numpy as np
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.qc.morphology import filter_by_eccentricity
from src.qc.shadows import filter_shadowed_labels
from src.qc.volumes import filter_by_minimum_volume
from src.qc.surf import filter_by_surf_distance


# def compute_qc_keep_labels(
#     mask: np.ndarray,
#     scale_vec: Sequence[float],
#     min_nucleus_vol: float,
#     z_prox_thresh: float,
#     max_eccentricity: float,
#     min_overlap: float,
#     max_surf_dist: float = 50.0,
# ) -> np.ndarray:
#
#     """Return the labels that survive volume, shadow, and morphology filters."""
#
#     # Volume-based filter
#     filtered, keep = filter_by_minimum_volume(mask, scale_vec, min_nucleus_vol)
#     if keep.size == 0:
#         return np.array([], dtype=np.int32)
#
#     # Refraction "shadow" filter
#     filtered, keep = filter_shadowed_labels(filtered, scale_vec, z_prox_thresh, min_overlap)
#     if keep.size == 0:
#         return np.array([], dtype=np.int32)
#
#     # Filter out extreme shape outliers
#     filtered, keep = filter_by_eccentricity(filtered, scale_vec, max_eccentricity)
#
#     # Filter by distance from spherical surface
#     filtered, keep = filter_by_surf_distance(filtered, scale_vec, max_surf_dist)
#
#     # TO DO: add refined SH-based surface distance filter
#
#     return keep

def get_existing_timepoints(group: zarr.Group, dataset_name="clean") -> set[int]:
    """Return set of time indices that already have chunks on disk."""
    if dataset_name not in group:
        return set()
    arr = group[dataset_name]
    store_keys = getattr(arr.store, "keys", lambda: [])()
    # Extract first coordinate (time) from chunk key names like "12.0.0"
    time_indices = set(int(k.split(".")[0]) for k in store_keys if "." in k and k.split(".")[0].isdigit())
    return time_indices

# ---------------------------------------------------------------------
# 2. Perform QC for a single frame and write clean mask
# ---------------------------------------------------------------------
def perform_mask_qc(
    t_int: int,
    store_path: Path,
    side_key: str,
    scale_vec: Iterable[float],
    min_nucleus_vol: float,
    z_prox_thresh: float,
    max_eccentricity: float,
    max_surf_dist: float,
    min_overlap: float,
) -> np.ndarray:

    """Run QC for a single time point and write cleaned mask to Zarr."""
    mask_store = zarr.open(store_path, mode="a")
    side_group = mask_store.require_group(side_key)
    mask = side_group["stitched"][t_int]
    clean_zarr = side_group["clean"]

    # Volume-based filter
    filtered, keep = filter_by_minimum_volume(mask, scale_vec, min_nucleus_vol)
    if keep.size == 0:
        return np.array([], dtype=np.int32)

    # Refraction "shadow" filter
    filtered, keep = filter_shadowed_labels(filtered, scale_vec, z_prox_thresh, min_overlap)
    # if keep.size == 0:
    #     return np.array([], dtype=np.int32)

    # Filter out extreme shape outliers
    filtered, keep = filter_by_eccentricity(filtered, scale_vec, max_eccentricity)

    # Filter by distance from spherical surface
    if len(keep) < 100:
        warnings.warn(f"{len(keep)} cells fount. Surface spheres may be unreliable if cells are sparse.", UserWarning)
    filtered, keep = filter_by_surf_distance(filtered, scale_vec, max_surf_dist)

    if keep.size == 0:
        clean_zarr[t_int] = np.zeros_like(mask, dtype=np.uint16)
    else:
        # keep only good labels
        clean_mask = np.isin(mask, keep) * mask
        clean_zarr[t_int] = clean_mask.astype(np.uint16)

    return keep

# ---------------------------------------------------------------------
# 3. Wrapper for full QC pipeline
# ---------------------------------------------------------------------
def mask_qc_wrapper(
    root: Path | str,
    project: str,
    mask_type: str = "li_segmentation",
    min_nucleus_vol: float = 25.0,
    z_prox_thresh: float = -30.0,
    max_eccentricity: float = 4.5,
    min_shadow_overlap: float = 0.35,
    max_surf_dist: float = 50.0,
    last_i: int | None = None,
    overwrite: bool = False,
    n_workers: int = 1,
) -> Dict[int, np.ndarray]:

    par_flag = n_workers is not None and n_workers > 1

    """Process all frames of the fused mask Zarr and create a clean sub-dataset."""
    root = Path(root)
    mask_path = root / "segmentation" / mask_type / f"{project}_masks.zarr"
    mask_store = zarr.open(mask_path, mode="a")
    side_keys = sorted([k["name"] for k in mask_store.attrs["sides"]])

    for side_key in side_keys:

        # choose source mask (e.g., 'aff')
        side_group = mask_store.require_group(side_key)
        mask_in = side_group["stitched"]
        if last_i is None:
            last_i = mask_in.shape[0]

        scale_vec = (
            side_group["stitched"].attrs.get("voxel_size_um")
            or side_group["stitched"].attrs.get("pixel_size_um")
        )
        if scale_vec is None:
            raise KeyError("Missing 'voxel_size_um' or 'pixel_size_um' in attrs.")

        # ---------------------------------------------------------
        # Create or reset "clean" dataset
        # ---------------------------------------------------------
        if "clean" in side_group:
            if overwrite:
                del side_group["clean"]
            else:
                clean_zarr = side_group["clean"]
        if "clean" not in side_group:
            clean_zarr = side_group.create_dataset(
                "clean",
                shape=mask_in.shape,
                dtype=np.uint16,
                chunks=mask_in.chunks,
            )

        # ---------------------------------------------------------
        # Indices to process
        # ---------------------------------------------------------
        # all_indices = set(range(last_i))
        # existing = (
        #     set(np.nonzero([np.any(side_group["clean"][i]) for i in range(last_i)])[0])
        #     if "clean" in side_group and not overwrite
        #     else set()
        # )
        # write_indices = sorted(all_indices if overwrite else (all_indices - existing))

        all_indices = set(range(last_i))
        existing = get_existing_timepoints(side_group, dataset_name="clean") if not overwrite else set()
        write_indices = sorted(all_indices if overwrite else (all_indices - existing))

        # ---------------------------------------------------------
        # Parallel/serial QC
        # ---------------------------------------------------------
        run = partial(
            perform_mask_qc,
            store_path=mask_path,
            side_key=side_key,
            scale_vec=scale_vec,
            min_nucleus_vol=min_nucleus_vol,
            z_prox_thresh=z_prox_thresh,
            max_eccentricity=max_eccentricity,
            min_overlap=min_shadow_overlap,
            max_surf_dist=max_surf_dist,
        )

        results: Dict[int, np.ndarray] = {}
        if write_indices:
            if par_flag:
                keep_lists = process_map(run, write_indices, max_workers=n_workers, chunksize=1)
                results.update({idx: np.asarray(keep) for idx, keep in zip(write_indices, keep_lists)})
            else:
                for idx in tqdm(write_indices, desc="Running mask QC..."):
                    results[int(idx)] = run(idx)

    return {}
