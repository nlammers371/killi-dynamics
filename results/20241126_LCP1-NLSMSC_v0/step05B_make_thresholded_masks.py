import pandas as pd
import zarr
import napari  # if you still need it loaded elsewhere
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["QT_API"] = "pyqt5"

# --- paths & params ---
root = Path(r"E:\Nick\killi_immuno_paper")
project = "20241126_LCP1-NLSMSC_v0"
fluo_thresh = 175  #215

mpath = root / "built_data" / "mask_stacks" / f"{project}_mask_aff_raw.zarr"
mpath_out = root / "built_data" / "mask_stacks" / f"{project}_mask_aff.zarr"
fluo_dir = root / "built_data" / "fluorescence_data" / project
zpath = root / "built_data" / "zarr_image_files" / f"{project}.zarr"

# --- load inputs ---
mask = zarr.open(mpath, mode="r")
im = zarr.open(zpath, mode="r")  # only used for shape/consistency here; remove if unused

# build output (use same shape/chunks/dtype); add a synchronizer for safe concurrent metadata ops
sync = zarr.ThreadSynchronizer()
out_zarr = zarr.open(
    mpath_out,
    mode="a",
    shape=mask.shape,
    chunks=mask.chunks,
    dtype=mask.dtype,
    synchronizer=sync,
)

for key in mask.attrs.keys():
    out_zarr.attrs[key] = mask.attrs[key]

# load fluorescence tables
fluo_frames_list = sorted(fluo_dir.glob("*.csv"))
fluo_df = pd.concat((pd.read_csv(fp) for fp in tqdm(fluo_frames_list, desc="Read CSVs")),
                    axis=0, ignore_index=True)

# pre-split nucleus IDs by frame (faster than filtering per-frame repeatedly)
# store as python sets for O(1) membership checks
frame_to_ids = {
    int(f): set(sub.loc[sub["mean_fluo"] > fluo_thresh, "nucleus_id"].astype(np.int64).tolist())
    for f, sub in fluo_df.groupby("frame", sort=True)
}

frames = sorted(frame_to_ids.keys())

def write_one(frame: int) -> int:
    # read one frame
    mask_frame = mask[frame]  # (Z, Y, X) or whatever your chunks are
    keep_ids = frame_to_ids.get(frame, set())

    if not keep_ids:
        out_zarr[frame] = np.zeros_like(mask_frame, dtype=mask.dtype)
        return frame

    # np.isin with a 1D array of IDs is fast; set -> array
    ids_arr = np.fromiter(keep_ids, dtype=mask.dtype, count=len(keep_ids))
    keep = np.isin(mask_frame, ids_arr)

    out = np.zeros_like(mask_frame, dtype=mask.dtype)
    # copy labels where they are in keep_ids
    out[keep] = mask_frame[keep]

    # write this frame (distinct chunk slice) – safe with ThreadSynchronizer
    out_zarr[frame] = out
    return frame

# tune workers to your storage/CPU; start low (e.g., 4–8) for spinning disks; higher for fast SSD/NVMe
num_workers = 24 #os.cpu_count() // 2 or 4

with ThreadPoolExecutor(max_workers=num_workers) as ex:
    futures = {ex.submit(write_one, f): f for f in frames}
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Writing"):
        pass
