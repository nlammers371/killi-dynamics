import pandas as pd
import zarr
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.segmentation import expand_labels
from skimage.measure import regionprops

os.environ["QT_API"] = "pyqt5"

# -------------------------
# Paths & params
# -------------------------
root = Path(r"E:\Nick\killi_immuno_paper")
project = "20241126_LCP1-NLSMSC"
fluo_thresh = 175

MIN_SIZE = 10  # minimum voxel count allowed for a nucleus (important!)
SAFE_MIN_SIZE = 15   # minimum voxel count after dilation

mpath = root / "built_data" / "mask_stacks" / f"{project}_mask_aff_raw.zarr"
mpath_out = root / "built_data" / "mask_stacks" / f"{project}_mask_aff.zarr"
fluo_dir = root / "built_data" / "fluorescence_data" / project
zpath = root / "built_data" / "zarr_image_files" / f"{project}.zarr"

# -------------------------
# Load inputs
# -------------------------
mask = zarr.open(mpath, mode="r")
im = zarr.open(zpath, mode="r")

scale_vec = np.asarray([
    im.attrs["PhysicalSizeZ"],
    im.attrs["PhysicalSizeY"],
    im.attrs["PhysicalSizeX"],
])

sync = zarr.ThreadSynchronizer()
out_zarr = zarr.open(
    mpath_out,
    mode="a",
    shape=mask.shape,
    chunks=mask.chunks,
    dtype=mask.dtype,
    synchronizer=sync
)

for key, val in mask.attrs.items():
    out_zarr.attrs[key] = val

# -------------------------
# Fluorescence table
# -------------------------
fluo_frames_list = sorted(fluo_dir.glob("*.csv"))
fluo_df = pd.concat(
    (pd.read_csv(fp) for fp in tqdm(fluo_frames_list, desc="Read CSVs")),
    axis=0,
    ignore_index=True,
)

frame_to_ids = {
    int(f): set(sub.loc[sub["mean_fluo"] > fluo_thresh, "nucleus_id"].astype(np.int64))
    for f, sub in fluo_df.groupby("frame", sort=True)
}

frames = sorted(frame_to_ids.keys())


# -------------------------
# Helper: Validate nucleus region
# -------------------------
def is_valid_region(region):
    """Return True only for nuclei safe for Ultrack."""
    if region.area < MIN_SIZE:
        return False
    c = region.centroid
    if len(c) not in (3,):  # should always be z,y,x
        return False
    if not np.isfinite(c).all():
        return False
    return True


def post_expand_check(label_image, ids):
    """
    Fast QC after dilation using a single regionprops call.

    Returns:
        final_image  - cleaned label image
        valid_ids    - list of labels that passed QC
    """
    final = np.zeros_like(label_image, dtype=label_image.dtype)

    # Collect only the labels we care about (ignore everything else)
    ids_set = set(ids)

    # Compute props once — this is cheap and fast
    props = regionprops(label_image)

    out_ids = []

    for r in props:
        lbl = r.label
        if lbl not in ids_set:
            continue

        coords = r.coords   # (N, 3) array: z,y,x for N voxels

        # --- area check ---
        area = coords.shape[0]
        if area < SAFE_MIN_SIZE:
            continue

        # --- centroid check (manual, fast) ---
        centroid = coords.mean(axis=0)
        if not np.isfinite(centroid).all():
            continue

        # Passed all filters
        final[coords[:, 0], coords[:, 1], coords[:, 2]] = lbl
        out_ids.append(lbl)

    return final, out_ids


# -------------------------
# Worker
# -------------------------
def write_one(frame: int) -> int:
    mask_frame = mask[frame]
    keep_ids = frame_to_ids.get(frame, set())

    if not keep_ids:
        out_zarr[frame] = np.zeros_like(mask_frame)
        return frame

    ids_arr = np.fromiter(keep_ids, dtype=mask.dtype, count=len(keep_ids))

    # (1) keep only selected nuclei
    keep = np.isin(mask_frame, ids_arr)
    filtered = np.zeros_like(mask_frame, dtype=mask.dtype)
    filtered[keep] = mask_frame[keep]

    # (2) remove degenerate pre-dilation nuclei
    safe_ids = []
    for r in regionprops(filtered):
        if is_valid_region(r):
            safe_ids.append(r.label)

    if not safe_ids:
        out_zarr[frame] = np.zeros_like(mask_frame)
        return frame

    # keep only safe pre-dilation nuclei
    pre = np.zeros_like(filtered)
    for lbl in safe_ids:
        pre[filtered == lbl] = lbl

    # (3) dilation
    expanded = expand_labels(pre, distance=3, spacing=scale_vec)

    # (4) remove degenerate post-dilation objects
    final_frame, _ = post_expand_check(expanded, safe_ids)

    out_zarr[frame] = final_frame
    return frame


# -------------------------
# Parallel execution
# -------------------------
num_workers = 24

with ThreadPoolExecutor(max_workers=num_workers) as ex:
    futures = {ex.submit(write_one, f): f for f in frames}
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Writing"):
        pass
