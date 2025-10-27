import zarr
import napari
import numpy as np
from pathlib import Path
from src.registration.virtual_fusion import VirtualFuseArray
# zarr_path = Path("E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/killi_tracker/built_data/zarr_image_files/20241126_LCP1-NLSMSC.zarr")
# date = "20250716"
# zarr_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/{date}/")
# image_list = sorted(list(zarr_path.glob("*.zarr")))
# zarr_path = Path(r"E:/Nick/killi_dynamics/built_data/zarr_image_files/20250419_BC1-NLSMSC.zarr")
zarr_path = Path(r"Y:/killi_dynamics/built_data/zarr_image_files/20251019_BC1-NLS_52-80hpf.zarr")
store = zarr.open(zarr_path, mode="r")
side = "side_00"
im_zarr = store[side]
# vf = VirtualFuseArray(
#     zarr_path,
#     use_gpu=True,      # make sure this is True
#     interp="linear",
# )
# zarr_path = "/media/nick/cluster/projects/data/killi_tracker/built_data/cellpose_output/tdTom-bright-log-v5/20250716/20250716_well0000_probs.zarr"
# mask_path = "/media/nick/cluster/projects/data/killi_tracker/built_data/mask_stacks/tdTom-bright-log-v5/20250716/20250716_well0000_mask_aff.zarr"
# image_ind = 0
# im_zarr = zarr.open(image_list[image_ind], mode='r')
# mask_zarr = zarr.open(mask_path, mode='r')
scale_vec = tuple(im_zarr.attrs['voxel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(im_zarr[500])  # Assuming channel 0 is the nuclear channel
# warm up
# vf[0]
viewer = napari.Viewer()
viewer.add_image(im_plot, channel_axis=0, scale=scale_vec)

if __name__ == '__main__':
    napari.run()
    # import time
    # import numpy as np
    # import pandas as pd
    #
    # # ---- define test sizes (Z, Y, X slices) ----
    # roi_sizes = [
    #     (50, 200, 200),
    #     (100, 400, 400),
    #     (200, 600, 600),
    #     (300, 900, 900),
    #     (400, 1200, 1200),
    # ]
    #
    # t_index, c_index = 2, 1  # adjust as needed
    # timings = []
    #
    # print(f"[Timing VirtualFuseArray] use_gpu={vf.use_gpu}")
    #
    # for (z_len, y_len, x_len) in roi_sizes:
    #     z0, y0, x0 = 100, 127, 100
    #     z1, y1, x1 = z0 + z_len, y0 + y_len, x0 + x_len
    #
    #     print(f"\n→ ROI size: Z={z_len}, Y={y_len}, X={x_len}")
    #     start = time.perf_counter()
    #
    #     # trigger actual read + compute
    #     arr = np.squeeze(vf[t_index, c_index, z0:z1, y0:y1, x0:x1])
    #
    #     xp = vf.xp
    #     if vf.use_gpu:
    #         xp.cuda.Stream.null.synchronize()  # make sure GPU finished before timing
    #         arr = arr.get()  # pull to host so we can measure array size in bytes
    #
    #     elapsed = time.perf_counter() - start
    #     n_voxels = arr.size
    #     mb_read = arr.nbytes / 1e6
    #     rate = mb_read / elapsed if elapsed > 0 else np.nan
    #
    #     print(f"    time: {elapsed:6.3f}s | size: {mb_read:7.1f} MB | rate: {rate:6.1f} MB/s")
    #
    #     timings.append(dict(
    #         Z=z_len, Y=y_len, X=x_len,
    #         seconds=elapsed,
    #         mb=mb_read,
    #         mb_per_s=rate,
    #     ))
    #
    # timing_df = pd.DataFrame(timings)
    # print("\n--- Summary ---")
    # print(timing_df)

    # napari.run()
    print("Check")