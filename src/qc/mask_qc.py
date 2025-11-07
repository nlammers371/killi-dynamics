"""High-level quality-control pipeline for segmentation masks."""
from __future__ import annotations

import multiprocessing
import os
import warnings
from functools import partial
from typing import Dict, Iterable, Mapping, Sequence, Literal
from pathlib import Path
import numpy as np
import zarr
from tqdm import tqdm
from skimage.segmentation import relabel_sequential
from tqdm.contrib.concurrent import process_map
from src.registration.virtual_fusion import VirtualFuseArray
from src.qc.morphology import filter_by_eccentricity
from src.qc.shadows import filter_shadowed_labels
from src.qc.volumes import filter_by_minimum_volume
from src.qc.surf import filter_by_surf_distance

# def get_existing_timepoints(group: zarr.Group, dataset_name="clean") -> set[int]:
#     """Return set of time indices that already have chunks on disk."""
#     if dataset_name not in group:
#         return set()
#     arr = group[dataset_name]
#     store_keys = getattr(arr.store, "keys", lambda: [])()
#     # Extract first coordinate (time) from chunk key names like "12.0.0"
#     time_indices = set(int(k.split(".")[0]) for k in store_keys if "." in k and k.split(".")[0].isdigit())
#     return time_indices
def get_time_indices(arr) -> set[int]:
    """
    Return the set of written time indices for a zarr.Array.
    Works for any Zarr store backend (DirectoryStore, FSStore, etc.).
    """
    store = arr.store

    # Get all logical keys for this array
    if hasattr(store, "keys"):
        prefix = arr.path.lstrip("/") + "/"
        keys = [k for k in store.keys() if k.startswith(prefix)]
    else:
        # Fallback: filesystem-based enumeration
        try:
            path = Path(store.path) / arr.path.lstrip("/")
            keys = [str(p.relative_to(path.parent)) for p in path.glob("*.*")]
        except Exception:
            keys = []

    # Extract time indices from chunk key names like '12.0.0'
    time_indices = {
        int(Path(k).name.split(".")[0])
        for k in keys
        if "." in Path(k).name and Path(k).name.split(".")[0].isdigit()
    }
    return time_indices


def is_zarr_array_empty(arr) -> bool:
    """Return True if the zarr.Array has no stored chunks."""
    return len(get_time_indices(arr)) == 0


# ---------------------------------------------------------------------
# 2. Perform QC for a single frame and write clean mask
# ---------------------------------------------------------------------

def perform_mask_qc(
    t_int: int,
    store_path: Path,
    side_key: str,
    in_mask_field: str,
    out_mask_field: str,
    scale_vec: Iterable[float],
    min_nucleus_vol: float | None,
    z_prox_thresh: float | None,
    max_eccentricity: float | None,
    max_surf_dist: float | None,
    min_overlap: float | None,
) -> None:
    """
    Run quality control filters on a single 3D segmentation mask frame
    and write the cleaned, relabeled mask back to the Zarr store.

    Filters applied sequentially:
        1. Minimum volume threshold
        2. Shadow proximity (refraction)
        3. Eccentricity cutoff
        4. Surface distance threshold

    Parameters
    ----------
    t_int : int
        Time index to process.
    store_path : Path
        Path to Zarr store containing masks.
    side_key : str
        Group key within store (e.g. "side_00").
    in_mask_field : str
        Input mask dataset name (e.g. "stitched").
    out_mask_field : str
        Output dataset name (e.g. "clean").
    scale_vec : Iterable[float]
        Physical voxel scaling [z, y, x].
    min_nucleus_vol, z_prox_thresh, max_eccentricity, max_surf_dist, min_overlap : float | None
        Filter thresholds; if None, step is skipped.
    """

    mask_store = zarr.open(store_path, mode="a")
    side_group = mask_store.require_group(side_key)
    mask = side_group[in_mask_field][t_int]
    clean_prezarr = side_group[out_mask_field]

    filtered = mask.copy()

    # --- Volume-based filter ---
    if min_nucleus_vol is not None:
        filtered, _ = filter_by_minimum_volume(filtered, scale_vec, min_nucleus_vol)
        if np.max(filtered) == 0:
            clean_prezarr[t_int] = np.zeros_like(mask, dtype=np.uint16)
            return

    # --- Shadow / refraction filter ---
    if z_prox_thresh is not None and min_overlap is not None:
        filtered, _ = filter_shadowed_labels(filtered, scale_vec, z_prox_thresh, min_overlap)

    # --- Eccentricity filter ---
    if max_eccentricity is not None:
        filtered, _ = filter_by_eccentricity(filtered, scale_vec, max_eccentricity)

    # --- Surface distance filter ---
    if max_surf_dist is not None:
        n_labels = len(np.unique(filtered)) - 1  # exclude background
        if n_labels < 100:
            warnings.warn(
                f"Only {n_labels} nuclei found; surface fits may be unreliable for sparse frames.",
                UserWarning,
            )
        filtered, _ = filter_by_surf_distance(filtered, scale_vec, max_surf_dist)

    # --- Relabel clean mask from 1..N ---
    relabeled, _, _ = relabel_sequential(filtered.astype(np.int32))
    clean_prezarr[t_int] = relabeled.astype(np.uint16)


def mask_qc_loop(side_keys: Sequence[str],
                 mask_path: Path,
                 mask_field_in: str,
                 mask_field_out: str,
                 overwrite: bool = False,
                 min_nucleus_vol: float | None = None,
                 z_prox_thresh: float | None = None,
                 max_eccentricity: float | None = None,
                 min_shadow_overlap: float | None = None,
                 max_surf_dist: float | None = None,
                 n_workers: int = 1,
                 last_i: int | None = None,):
    
    mask_store = zarr.open(mask_path, mode="a")
    for side_key in side_keys:

        # choose source mask (e.g., 'aff')
        side_group = mask_store.require_group(side_key)
        mask_in = side_group[mask_field_in]
        if last_i is None:
            last_i = mask_in.shape[0]

        scale_vec = (
                side_group[mask_field_in].attrs.get("voxel_size_um")
                or side_group[mask_field_in].attrs.get("pixel_size_um")
        )
        if scale_vec is None:
            raise KeyError("Missing 'voxel_size_um' or 'pixel_size_um' in attrs.")

        # ---------------------------------------------------------
        # Create or reset "clean" dataset
        # ---------------------------------------------------------
        if mask_field_out in side_group:
            if overwrite:
                del side_group[mask_field_out]
            else:
                clean_prezarr = side_group[mask_field_out]
        if mask_field_out not in side_group:
            clean_prezarr = side_group.create_dataset(
                mask_field_out,
                shape=mask_in.shape,
                dtype=np.uint16,
                chunks=mask_in.chunks,
            )
            # transfer met adata from stitched to clean
            for k, v in mask_in.attrs.items():
                clean_prezarr.attrs[k] = v

        # ---------------------------------------------------------
        # Indices to process
        # ---------------------------------------------------------
        all_indices = set(range(last_i))
        # existing = get_existing_timepoints(side_group, dataset_name=mask_field_out) if not overwrite else set()
        existing = get_time_indices(side_group[mask_field_out]) if not overwrite else set()
        write_indices = sorted(all_indices if overwrite else (all_indices - existing))

        # ---------------------------------------------------------
        # Parallel/serial QC
        # ---------------------------------------------------------
        run = partial(
            perform_mask_qc,
            store_path=mask_path,
            side_key=side_key,
            in_mask_field=mask_field_in,
            out_mask_field=mask_field_out,
            scale_vec=scale_vec,
            min_nucleus_vol=min_nucleus_vol,
            z_prox_thresh=z_prox_thresh,
            max_eccentricity=max_eccentricity,
            min_overlap=min_shadow_overlap,
            max_surf_dist=max_surf_dist,
        )

        if len(write_indices) > 0:
            if mask_field_out == "clean":
                desc_text = "Running mask QC (surface distance filter)..."
            else:
                desc_text = "Running mask QC (Mask shape and refraction artifacts)..."
            if n_workers > 1:
                process_map(run, write_indices, max_workers=n_workers, chunksize=1, desc=desc_text)
            else:
                for idx in tqdm(write_indices, desc=desc_text):
                    run(idx)
        return len(write_indices) > 0
                    
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
    overwrite00: bool = False,
    overwrite01: bool = False,
    skip_surf_filtering: bool = False,
    n_workers: int = 1,
    delete_intermediate: bool = True,
) -> Dict[int, np.ndarray]:



    """Process all frames of the fused mask Zarr and create a clean sub-dataset."""
    root = Path(root)
    mask_path = root / "segmentation" / mask_type / f"{project}_masks.zarr"
    mask_store = zarr.open(mask_path, mode="a")
    
    # ---------------------------------------------------------
    # First QC loop!
    # if qc_iter == "initial":
    side_keys = sorted([k["name"] for k in mask_store.attrs["sides"] if "side_" in k["name"]])
    if len(side_keys) == 0:
        raise ValueError("No valid side keys found in mask Zarr store.")
    elif len(side_keys) > 2:
        raise ValueError("More than two sides found in mask Zarr store; expected one or two.")
    
    # call first loop
    updated_flag = mask_qc_loop(side_keys=side_keys,
                         mask_path=mask_path,
                         mask_field_in="stitched",
                         mask_field_out="clean_pre",
                         overwrite=overwrite00,
                         min_nucleus_vol=min_nucleus_vol,
                         z_prox_thresh=z_prox_thresh,
                         max_eccentricity=max_eccentricity,
                         min_shadow_overlap=min_shadow_overlap,
                         max_surf_dist=None,
                         last_i=last_i,
                         n_workers=n_workers)


    # fuse masks across sides if needed
    side_key_re = ["side_00"]
    if len(side_keys) == 2:
        # vf = VirtualFuseArray(
        #     store_path=mask_path,
        #     is_mask=True,
        #     subgroup_key="clean_pre",
        #     use_gpu=False,
        # )
        # overwrite = updated_flag
        # vf.write_fused(
        #     subgroup="clean_pre",  # writes to fused/clean
        #     overwrite=overwrite,
        #     n_workers=n_workers
        # )
        side_key_re = ["fused"]
            
    # ---------------------------------------------------------
    # Second round of QC using spherical embryo surface fits
    # ---------------------------------------------------------
    if not skip_surf_filtering:
        overwrite01 = overwrite00 or overwrite01
        mask_qc_loop(side_keys=side_key_re,
                     mask_path=mask_path,
                     mask_field_in="clean_pre",
                     mask_field_out="clean",
                     overwrite=overwrite01,
                     min_nucleus_vol=None,
                     z_prox_thresh=None,
                     max_eccentricity=None,
                     min_shadow_overlap=None,
                     max_surf_dist=max_surf_dist if side_key_re[0] == "fused" else 1.5 * max_surf_dist,
                     last_i=last_i,
                     n_workers=n_workers)

        if delete_intermediate:
            # delete clean_pre datasets
            side_keys = np.unique(side_keys + side_key_re).tolist()
            for side_key in side_keys:
                side_group = mask_store.require_group(side_key)
                if "clean_pre" in side_group:
                    del side_group["clean_pre"]
        

    return {}
