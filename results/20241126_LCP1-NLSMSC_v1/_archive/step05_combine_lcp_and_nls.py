import numpy as np
from pathlib import Path
import zarr
from tqdm import tqdm
import numpy as np
import dask.array as da
import zarr
from dask.diagnostics import ProgressBar
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_frame(t, path, min_size):
    arr = zarr.open(path, mode="r+")
    vol = arr[t][...]   # force load into memory
    vol = remove_small_labels(vol, min_size)
    arr[t] = vol
    return t

def remove_small_labels(label_arr, min_size):
    # Count voxel counts for each label
    counts = np.bincount(label_arr.ravel())
    # Identify labels to remove
    remove_ids = np.where(counts < min_size)[0]
    if 0 in remove_ids:  # never remove background
        remove_ids = remove_ids[remove_ids != 0]
    # Make a mask of which pixels to kill
    mask = np.isin(label_arr, remove_ids)
    out = label_arr.copy()
    out[mask] = 0
    return out


if __name__ == "__main__":

    root = Path("E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\")
    project = "20241126_LCP1-NLSMSC_v0"

    # src_nls = root / "built_data" / "mask_stacks" / (project + "_mask_aff_nls.zarr")
    # src_lcp = root / "built_data" / "mask_stacks" / (project + "_mask_aff_lcp.zarr")
    dst_path = root / "built_data" / "mask_stacks" / (project + "_mask_aff.zarr")

    # Lazy Zarr readers (no load)
    # a = da.from_zarr(src_nls)  # dtype uint16 (labels)
    # b = da.from_zarr(src_lcp)  # dtype uint16 (mask)
    #
    # # Sanity check shapes
    # if a.shape != b.shape:
    #     raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    #
    # # Keep NLS labels where LCP is nonzero, else 0
    # out = da.where(b != 0, a, 0).astype(a.dtype)
    #
    # # Keep chunking identical to source to avoid rechunk cost
    # out = out.rechunk(a.chunks)
    #
    # # Write lazily, chunk-by-chunk, with a simple progress bar
    # with ProgressBar():
    #     out.to_zarr(dst_path, overwrite=True, compute=True)
    #
    # # Copy attrs if you want a faithful header
    # src = zarr.open(src_nls, mode="r")
    # dst = zarr.open(dst_path, mode="r+")
    # dst.attrs.update(dict(src.attrs))
    #
    # ---- NEW: filter small regions per time frame (label images) ----
    dst_arr = zarr.open(dst_path, mode="r+")
    T = dst_arr.shape[0]
    MAX_WORKERS = 12  # <- adjust based on your CPU cores
    MIN_SIZE = 25  # <- your threshold in voxels

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(tqdm(
            pool.map(partial(process_frame, path=dst_path, min_size=MIN_SIZE), range(T)),
            total=T
        ))