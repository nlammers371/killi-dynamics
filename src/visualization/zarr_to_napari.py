import numpy as np
from pathlib import Path
from skimage.measure import label
from scipy.ndimage import rotate
import napari

from src.registration.virtual_fusion import VirtualFuseArray



if __name__ == "__main__":


    # -------------------

    # load the fused array
    zarr_path = Path(r"Y:/killi_dynamics/built_data/zarr_image_files/20250611_UTR-NLS_52hpf-76hpf.zarr")
    vf = VirtualFuseArray(
        zarr_path,
        use_gpu=True,
        interp="nearest",
        eval_mode=False,
    )

    # extract voxel scale and target plane
    scale_vec = tuple(vf.attrs["voxel_size_um"])  # (z, y, x)
    im_plot = np.squeeze(vf[718, :])             # single timepoint, channel 1
    print(im_plot.shape)

    # visualize
    viewer = napari.Viewer()
    viewer.add_image(im_plot, channel_axis=0, scale=scale_vec)
    napari.run()
    print("Check")
