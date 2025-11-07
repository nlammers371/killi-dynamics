import numpy as np
from pathlib import Path
from src.fluorescence.get_mask_fluorescence import _mean_fluo_foreground_single
from src.registration.virtual_fusion import VirtualFuseArray
import zarr
from skimage.measure import regionprops_table
import pandas as pd
from time import time


if __name__ == "__main__":

    frame = 20
    ch = 0
    # load the fused array
    zarr_path = Path(r"Y:/killi_dynamics/built_data/zarr_image_files/20251019_BC1-NLS_52-80hpf.zarr")
    mask_path = Path(r"Y:/killi_dynamics/segmentation/li_segmentation/20251019_BC1-NLS_52-80hpf_masks.zarr")
    vi = VirtualFuseArray(
        zarr_path,
        use_gpu=True,
        interp="nearest",
        eval_mode=False,
    )
    vm = zarr.open(mask_path, mode="r")["fused"]["clean"]

    # use sparse estimation first
    print("performing sparse estimation...")
    start = time()
    mask_zarr_path = mask_path / "fused" / "clean"
    fg_group_path = mask_zarr_path.parent / f"foreground_clean"
    df_sparse = _mean_fluo_foreground_single(
                        t=frame,
                        mask_zarr_path=mask_zarr_path,
                        fg_group_path=fg_group_path,
                        )

    df_sparse = df_sparse[df_sparse["channel"] == ch]

    print("Sparse estimation done in %.2f seconds." % (time() - start))
    print("Performing dense estimation...")
    start = time()
    # now do it the old way for comparison
    im_frame = vi[frame, ch]
    im_mask = vm[frame, :, :, :]
    df_dense = regionprops_table(im_mask, intensity_image=im_frame, properties=['label', 'mean_intensity'])
    df_dense = pd.DataFrame.from_dict(df_dense).rename(columns={"label": "mask_id", "mean_intensity": "mean_fluo"})
    # print("check")
    df_merge = df_dense.merge(df_sparse, on="mask_id", how="inner")
    df_merge["diff"] = df_merge["mean_fluo_x"] - df_merge["mean_fluo_y"]
    print("Dense estimation done in %.2f seconds." % (time() - start))

    print(f"Maximum difference between methods: {np.max(df_merge['diff'])}")
