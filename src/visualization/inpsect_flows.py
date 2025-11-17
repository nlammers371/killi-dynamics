import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pathlib import Path
import zarr
import napari

from src.registration.virtual_fusion import VirtualFuseArray



if __name__ == "__main__":


    # -------------------

    # load the fused array
    zarr_path = Path(r"/media/nick/hdd011/killi_dynamics/optical_flow/20251019_BC1-NLS_52-80hpf_optical_flow.zarr")
    # vf = VirtualFuseArray(
    #     zarr_path,
    #     use_gpu=True,
    #     interp="nearest",
    #     eval_mode=False,
    # )
    #
    # # extract voxel scale and target plane
    # scale_vec = tuple(vf.attrs["voxel_size_um"])  # (z, y, x)
    # im_plot = np.squeeze(vf[718, :])             # single timepoint, channel 1
    # print(im_plot.shape)
    flow_zarr = zarr.open(zarr_path, mode="r")

    # visualize
    viewer = napari.Viewer()
    viewer.add_image(flow_zarr, channel_axis=0, scale=tuple([3.0, 0.85, 0.85]))
    napari.run()
    print("Check")
