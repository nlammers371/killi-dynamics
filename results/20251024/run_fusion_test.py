from pathlib import Path
from time import time

import numpy as np
import cupy as cp
import zarr
from tqdm import tqdm

from src.registration.virtual_fusion import VirtualFuseArray


# ------------------------ Setup ------------------------
zarr_path = Path(r"E:/Nick/killi_dynamics/built_data/zarr_image_files/20250419_BC1-NLSMSC.zarr")
store = zarr.open(zarr_path, mode="r")

vf = VirtualFuseArray(
    zarr_path,
    overlap_z=30,
    use_gpu=True,      # make sure this is True
    interp="linear",
)

print(vf)

# quick sanity check
arr = vf[0]
print(f"Array type: {type(arr)} | shape: {arr.shape} | dtype: {arr.dtype}")

# ------------------------ Benchmark ------------------------
N = 2

# --- (1) Baseline: direct zarr read ---
start = time()
for i in tqdm(range(N), desc="Fused zarr indexing"):
    arr0 = np.asarray(store["side_00"][i])
    arr1 = np.asarray(store["side_01"][i])
    _ = arr0.mean()
    _ = arr1.mean()
end = time()
print(f"Fused zarr indexing time for {N} frames: {end - start:.2f} seconds")


# --- (2) Virtual fused (GPU or CPU) ---
start = time()
for i in tqdm(range(N), desc="Virtual fused indexing"):
    arr = vf[i]
    # ensure compute actually runs on device before timing ends
    if isinstance(arr, cp.ndarray):
        _ = arr.mean()
        cp.cuda.Stream.null.synchronize()
    else:
        _ = arr.mean()
end = time()
print(f"Virtual fused indexing time for {N} frames: {end - start:.2f} seconds")
